"""同花顺（iFinD / Fuyao）适配器：经 MCP 接入的 A 股与基金/期货/期权数据（docs 03.5）。

**它补齐了什么。** akshare 覆盖 A 股行情与一份中文宽表式财报摘要，但拿不到结构化
的三表字段、按报告期的完整财务指标、同花顺概念指数成分股、横截面估值快照，以及
涨停池/龙虎榜/热股榜这类特色数据。本适配器把这些接到同一条适配器链上——因此
``get_quote``/``get_kline``/``get_financials``/``get_indicators`` 这些既有工具
一行不改就获得了新的字段与覆盖范围。

**两条入口。** 五个语义方法（quote/kline/financials/indicators/index_constituents）
是 typed 的：它们把结果归一化到本项目既有的表格契约，因此下游的渲染、裁剪、计算
与引用都能直接复用。此外 ``fetch_dataset`` 是一条通用通道，用来触达长尾端点
（特色数据、基金/期货/期权），其数据集名与参数来自服务端 ``tools/list``，而不是
本地手抄的 schema。

**归一化契约。** 行情统一为 ``date/open/high/low/close/volume/amount``（
``MARKET_COLUMNS``）；财报统一为「指标行 × ``YYYYMMDD`` 报告期列」的宽表，与
akshare 的摘要表同形，使裁剪时的年度优先排序与计算工具都无需特判；指标统一为
「``date`` 列 + 中文指标名列」。列名、标签、报告期规则全部来自 ``data/mapping.py``，
本模块不含任何同花顺专有的名称字面量。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.adapters.mcp_client import McpHttpClient, McpToolError
from finharness.data.adapters.tavily_adapter import system_proxy
from finharness.data.mapping import (
    FUYAO_ADJUST_MAP,
    FUYAO_COLUMN_MAP,
    FUYAO_DEFAULT_ADJUST,
    FUYAO_ENDPOINTS,
    FUYAO_FINANCIAL_LABELS,
    FUYAO_INDICATOR_LABELS,
    FUYAO_KLINE_PERIODS,
    FUYAO_MAX_WINDOW_YEARS,
    FUYAO_MS_DATE_COLUMNS,
    FUYAO_SERVICE_PATHS,
    FUYAO_SERVICES,
    fuyao_thscode,
    select_indicator_columns,
)

_SHANGHAI = timezone(timedelta(hours=8))

# 语义方法 -> 该请求落在哪个 MCP 服务上。
_METHOD_SERVICE: dict[str, str] = {
    "quote": "a-share",
    "kline": "a-share",
    "financials": "a-share",
    "indicators": "a-share",
    "index_constituents": "a-share-index",
}

# ``fetch_financials`` 的 statement 参数 -> ``FUYAO_ENDPOINTS`` 的键。
_STATEMENT_KEYS: dict[str, str] = {
    "利润": "financials:利润",
    "利润表": "financials:利润",
    "income": "financials:利润",
    "资产": "financials:资产",
    "资产负债表": "financials:资产",
    "balance": "financials:资产",
    "现金流": "financials:现金流",
    "现金流量表": "financials:现金流",
    "cashflow": "financials:现金流",
}

# 同花顺的业务错误码 -> 是否值得重试（docs 03.5「错误码」）。
# 1001-1004 是参数错误、2001/2003 是凭据、3001/3004 是标的问题：重试一次结果完全
# 一样，因此不可重试；4001 是限流，5001-5003 是服务端/上游故障，值得再试。
_RETRYABLE_CODES: frozenset[int] = frozenset({4001, 5001, 5002, 5003})
# 3001「标的不存在」与 3004「类型不支持」是确定的"这个源没有"，交给回退链下一环。
_NO_DATA_CODES: frozenset[int] = frozenset({3001, 3004})
# 3002「数据未就绪」是**歧义**的：既可能是"该期尚未披露"（确定的缺席），也可能是
# 快照类资源的**预热**——实测 ``get_fund_market_snapshot`` 连发三次，首次 3002、
# 后两次成功。把它归入"确定没有"会让一次上游冷启动抖动被当作"该源无此数据"直接
# 回退，明明重试一次就能拿到。
#
# 因此它单独成一类：在**本适配器内**做一次有界重试（而不是交给编排器，因为编排器
# 的重试会连带影响按报告期逐期取数的语义），重试后仍失败才按"未就绪"上报。
_READINESS_CODE = 3002
_READINESS_RETRY_ATTEMPTS = 2
_READINESS_RETRY_DELAY_S = 1.5


class FuyaoDataNotReady(AdapterError):
    """3002：数据未就绪（重试预算已用尽）。

    单独成一个类型，是为了让调用方（如按报告期逐期取财务指标）能区分"这一期还没
    披露"与"这次调用失败"：前者应当跳过该期、继续取其余期间，而不是丢掉整条序列。
    ``retryable`` 为 False——适配器内部已经试过了，再让编排器重试同一请求没有意义。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def _next_day_ms(base: date) -> int:
    """把日期转成 Asia/Shanghai 午夜的毫秒时间戳（同花顺的时间戳口径）。"""
    moment = datetime(base.year, base.month, base.day, tzinfo=_SHANGHAI)
    return int(moment.timestamp() * 1000)


def _to_ms_datetime(series: pd.Series) -> pd.Series:
    """毫秒 Unix 时间戳 -> 无时区的本地日期。

    同花顺的每个时间字段都是 Asia/Shanghai 午夜的毫秒时间戳。不做这一步的话，
    pandas 会把数字原样保留，渲染出来是一串整数——读者无从判断那是哪一天。
    """
    return (
        pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
        .dt.tz_convert(_SHANGHAI)
        .dt.tz_localize(None)
    )


def _flatten_record(record: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """把一条嵌套记录压成平面列。

    嵌套 dict 展开为点号列名（``a.b``）；列表序列化为 JSON 字符串而非展开——
    列表长度逐行不同，展开会让列数取决于数据，而表格的列必须是稳定的。保留原文
    好过丢弃：``concept_list`` 这类字段本身就是答案的一部分。
    """
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_record(value, prefix=f"{name}."))
        elif isinstance(value, list):
            flat[name] = (
                json.dumps(value, ensure_ascii=False)
                if value and isinstance(value[0], (dict, list))
                else "、".join(str(item) for item in value)
            )
        else:
            flat[name] = value
    return flat


def _records_frame(records: list[Any]) -> pd.DataFrame:
    """记录列表 -> 数据帧，嵌套字段已压平。"""
    rows = [
        _flatten_record(record) if isinstance(record, dict) else {"value": record}
        for record in records
    ]
    return pd.DataFrame(rows)


def _fold_financial_indicators(data: Any) -> dict[str, Any]:
    """把「五大类分组」的财务指标折成单行 ``{中文指标名: 数值}``。

    同花顺的指标端点是 ``abilities[]`` → ``indicators[]`` 的两层结构，直接展平会
    得到 ``abilities.0.indicators.0.value`` 之类的列名，模型无法据此判断哪个数字
    是 ROE。折成"指标名为列"既符合本项目指标表的既有形态，也让字段过滤
    （``fields=["ROE"]`` 走中文子串匹配）继续成立。
    """
    folded: dict[str, Any] = {}
    if not isinstance(data, dict):
        return folded
    for ability in data.get("abilities") or []:
        if not isinstance(ability, dict):
            continue
        for indicator in ability.get("indicators") or []:
            if not isinstance(indicator, dict):
                continue
            index_id = str(indicator.get("index_id") or "")
            label = FUYAO_INDICATOR_LABELS.get(index_id, index_id)
            folded[label] = _numeric_or_raw(indicator.get("value"))
    return folded


def _numeric_or_raw(value: Any) -> Any:
    """尽量转成数值；转不动就保留原文。

    数值化让分位、同比这类计算得以进行，而同花顺的指标值本身是字符串。但把它
    无条件转成 ``NaN`` 会静默丢掉信息（``"—"``、``"不适用"`` 都会变成空），
    因此转不动时原样保留。``None`` 保持 ``None``：上游以 null 表示"该期未披露"，
    填 0 会把它变成一条看起来真实的记录。
    """
    if value is None:
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return value
    return numeric


class FuyaoMcpAdapter(DataAdapter):
    """同花顺数据源；六个 MCP 服务共用一个实例、一个凭证、一份节流状态。"""

    name = "fuyao"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://fuyao.aicubes.cn",
        timeout_s: float = 30.0,
        proxy: str | None = None,
        throttle_seconds: float = 1.0,
        client: McpHttpClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # 显式指定的代理优先；否则回退到系统代理，使本适配器与 A 股数据源走相同的
        # 出口（httpx 只读环境变量，而 requests 还会读操作系统代理）。
        self.proxy = proxy or system_proxy()
        self.throttle_seconds = throttle_seconds
        # 每个服务一个客户端：会话 id 是按连接持有的，共用一个会让后者的
        # initialize 覆盖前者的会话。
        self._clients: dict[str, McpHttpClient] = {}
        self._client_lock = threading.Lock()
        # 测试注入：给定时代替按服务新建的客户端（单服务场景下更直接）。
        self._injected = client
        self._last_call = 0.0
        self._throttle_lock = threading.Lock()

    # -- 传输 ---------------------------------------------------------------
    def _throttle(self) -> None:
        """两次上游调用之间保持最小间隔，避免突发触发同花顺的动态限流。"""
        if self.throttle_seconds <= 0:
            return
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.throttle_seconds:
                time.sleep(self.throttle_seconds - elapsed)
            self._last_call = time.monotonic()

    def _client(self, service: str) -> McpHttpClient:
        if self._injected is not None:
            return self._injected
        with self._client_lock:
            existing = self._clients.get(service)
            if existing is not None:
                return existing
            path = FUYAO_SERVICE_PATHS.get(service)
            if path is None:
                raise AdapterError(f"未知的同花顺服务：{service}")
            created = McpHttpClient(
                url=self.base_url + path,
                api_key=self.api_key,
                timeout_s=self.timeout_s,
                proxy=self.proxy,
            )
            self._clients[service] = created
            return created

    @staticmethod
    def _unwrap(payload: Any) -> Any:
        """校验统一响应信封并返回 ``data``。

        同花顺的业务失败经 ``code`` 表达而 HTTP 状态码恒为 200，因此只检查状态码
        会把一个错误当成数据帧渲染出来——那正是"看起来正常的错数据"。
        """
        if not isinstance(payload, dict):
            raise AdapterError(f"同花顺返回了非预期载荷：{str(payload)[:200]}")
        if "code" not in payload:
            # 有些端点（或未来的包装层）可能直接给出数据体；没有 code 就当作数据。
            return payload.get("data", payload)
        code = payload.get("code")
        if code in (0, None, "0"):
            return payload.get("data")
        try:
            numeric = int(code)
        except (TypeError, ValueError):
            numeric = None
        message = str(payload.get("message") or "").strip()
        request_id = payload.get("request_id")
        detail = f"code={code}" + (f"：{message}" if message else "")
        if request_id:
            detail += f"（request_id={request_id}）"
        if numeric == _READINESS_CODE:
            # 歧义码：由 ``_call`` 决定是否重试；用专门类型让它能被区分处理。
            raise FuyaoDataNotReady(f"同花顺数据未就绪（{detail}）")
        if numeric in _NO_DATA_CODES:
            raise AdapterError(f"同花顺无此数据（{detail}）")
        raise AdapterError(
            f"同花顺调用失败（{detail}）", retryable=numeric in _RETRYABLE_CODES
        )

    def _call(self, service: str, dataset: str, arguments: dict[str, Any]) -> Any:
        """调用一个数据集并返回其载荷；失败统一转换为 ``AdapterError``。

        对 3002「数据未就绪」做一次有界重试：快照类资源首次访问常常是冷启动
        （实测 ``get_fund_market_snapshot`` 首次 3002、随即成功）。重试在**本方法
        内**完成，而不是交给编排器的重试，因为编排器的重试作用于整个 ``fetch_*``
        调用——按报告期逐期取财务指标时，那会把"某一期未披露"升级成"整条序列失败"。
        """
        self._throttle()
        attempts = _READINESS_RETRY_ATTEMPTS
        for attempt in range(1, attempts + 1):
            try:
                payload = self._client(service).call_tool(dataset, arguments)
            except McpToolError as exc:
                # 工具级错误：重试同样的参数不会有不同结果。
                raise AdapterError(f"{dataset}: {exc}") from exc
            try:
                return self._unwrap(payload)
            except FuyaoDataNotReady:
                if attempt >= attempts:
                    raise
                time.sleep(_READINESS_RETRY_DELAY_S)

    def _dataset_call(
        self, dataset: str, arguments: dict[str, Any], *, service: str | None = None
    ) -> Any:
        """按数据集名调用；服务由数据集名前缀推导，缺省时用 A 股服务。"""
        return self._call(service or _service_for_dataset(dataset), dataset, arguments)

    # -- 通用数据集 ---------------------------------------------------------
    def list_datasets(self, service: str) -> list[dict[str, Any]]:
        """某服务的数据集目录（含 ``inputSchema``），供工具层校验与发现。"""
        if service not in FUYAO_SERVICES:
            raise AdapterError(f"未知的同花顺服务：{service}")
        return self._client(service).list_tools()

    def fetch_dataset(
        self, service: str, dataset: str, params: dict[str, Any]
    ) -> FetchResult:
        """通用数据集取数：长尾端点经此进入同一条缓存/引用/溯源链路。

        载荷被压成平面表（见 ``_records_frame``）：派发器面对的是形态各异的端点，
        统一成表格才能复用既有的渲染与裁剪预算。财务指标那种两层结构在压平前先
        折叠成"指标名为列"，否则会得到一串读不出含义的点号列名。
        """
        payload = self._dataset_call(dataset, dict(params), service=service)
        return FetchResult(df=_payload_frame(payload), interface=dataset)

    # -- 语义方法 -----------------------------------------------------------
    def fetch_quote(self, symbol: str) -> FetchResult:
        """最新行情快照。

        同花顺的快照不带中文名（要显示需另查标的检索），因此这里只返回有把握的
        字段；用一次额外请求去补一个显示名不划算。
        """
        dataset = FUYAO_ENDPOINTS["quote"]
        payload = self._call(
            _METHOD_SERVICE["quote"], dataset, {"thscodes": fuyao_thscode(symbol)}
        )
        frame = _payload_frame(payload)
        if not len(frame):
            raise AdapterError(f"{dataset}: 未返回 {symbol} 的行情")
        # 快照记录本身不带日期，只有信封上的 ``timestamp``（上游最新有效时间）。
        # 不把它接出来的话，一行报价就没有任何时点信息：读者无从判断这是今天的
        # 收盘还是几天前的，缓存 TTL 也只能退化成"抓取时刻"。
        snapshot_at = payload.get("timestamp") if isinstance(payload, dict) else None
        if snapshot_at is not None and "date" not in frame.columns:
            frame = frame.assign(
                date=_to_ms_datetime(pd.Series([snapshot_at])).iloc[0]
            )
        return FetchResult(df=frame.reset_index(drop=True), interface=dataset)

    def fetch_kline(
        self, symbol: str, period: str, adjust: str | None, years: int
    ) -> FetchResult:
        """日线序列（前/后复权）。

        同花顺的 K 线只提供日线：周/月线请求抛 ``NotImplementedError``，编排器会
        读作"该源不支持"并转到下一个数据源。这不是缺陷而是回退——把它硬凑成日线
        聚合出来的周线，会让数据来源与调用方的请求不再对应。
        """
        if period not in FUYAO_KLINE_PERIODS:
            raise NotImplementedError(f"同花顺 K 线不支持周期 {period}")
        span = max(int(years), 1)
        if span > FUYAO_MAX_WINDOW_YEARS:
            # 上游对超过十年的窗口返回 code=1003。显式报错胜过把一个被上游悄悄
            # 截短的序列当成"近 N 年"交出去。
            raise AdapterError(
                f"同花顺 K 线窗口上限为 {FUYAO_MAX_WINDOW_YEARS} 年，请求了 {span} 年"
            )
        dataset = FUYAO_ENDPOINTS["kline"]
        end = date.today()
        start = end - timedelta(days=365 * span + 30)
        payload = self._call(
            _METHOD_SERVICE["kline"],
            dataset,
            {
                "thscode": fuyao_thscode(symbol),
                "interval": "1d",
                "start": _next_day_ms(start),
                "end": _next_day_ms(end + timedelta(days=1)),
                "adjust": FUYAO_ADJUST_MAP.get(str(adjust or "").lower(), FUYAO_DEFAULT_ADJUST),
            },
        )
        frame = _payload_frame(payload)
        if not len(frame):
            raise AdapterError(f"{dataset}: 未返回 {symbol} 的 K 线")
        if "date" in frame.columns:
            # 内部契约是最新在前，使摘要与均线窗口读到的总是最新一期。
            frame = frame.sort_values("date", ascending=False)
        return FetchResult(df=frame.reset_index(drop=True), interface=dataset)

    def fetch_financials(self, symbol: str, statement: str, years: int) -> FetchResult:
        """利润表 / 资产负债表 / 现金流量表，折成"指标行 × 报告期列"的宽表。

        与 akshare 的财报摘要同形，因此列裁剪时"年度列优先"的规则、以及依赖该形态
        的计算与输出工具都无需为同花顺特判。
        """
        key = _STATEMENT_KEYS.get(str(statement).strip())
        if key is None:
            raise NotImplementedError(f"同花顺不支持报表类型 {statement}")
        dataset = FUYAO_ENDPOINTS[key]
        payload = self._call(
            _METHOD_SERVICE["financials"],
            dataset,
            {
                "thscode": fuyao_thscode(symbol),
                "period": "annual",
                "limit": max(int(years), 1) * (4 if int(years) <= 3 else 1),
            },
        )
        items = (payload or {}).get("item") if isinstance(payload, dict) else None
        if not items:
            raise AdapterError(f"{dataset}: 未返回 {symbol} 的报表数据")
        return FetchResult(
            df=_statement_frame(items), interface=dataset
        )

    def fetch_indicators(
        self, symbol: str, years: int, fields: list[str] | None
    ) -> FetchResult:
        """财务指标序列（成长/盈利/偿债/营运/现金流）。

        该端点一次只返回**一个**报告期，因此要凑出 ``years`` 年的序列就得发多次
        请求；期数由 ``mapping.fuyao_report_periods`` 决定并有上限，使最坏情况下的
        请求数可推理。某一期没有数据（次新股、尚未披露）只是少一行，不丢整条序列。
        """
        from finharness.data.mapping import fuyao_report_periods

        dataset = FUYAO_ENDPOINTS["indicators"]
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for period in fuyao_report_periods(years):
            try:
                payload = self._call(
                    _METHOD_SERVICE["indicators"],
                    dataset,
                    {"thscode": fuyao_thscode(symbol), "report": period},
                )
            except AdapterError as exc:
                # 单期失败不丢整条序列：未披露的期间本就该缺席。
                if exc.retryable:
                    raise
                errors.append(f"{period}: {exc.message}")
                continue
            row = _fold_financial_indicators(payload)
            if not row:
                continue
            row["date"] = _period_to_date(period)
            rows.append(row)
        if not rows:
            raise AdapterError(
                f"{dataset}: 未返回 {symbol} 的财务指标"
                + (f"（{'; '.join(errors[:3])}）" if errors else "")
            )
        frame = pd.DataFrame(rows)
        frame = frame.sort_values("date", ascending=False).reset_index(drop=True)
        if fields:
            keep, _unmatched = select_indicator_columns(frame.columns, fields)
            frame = frame[keep]
        return FetchResult(df=frame, interface=dataset)

    def fetch_index_constituents(self, index: str) -> FetchResult:
        """同花顺指数成分股。

        只受理能映射到 thscode 的指数；无法映射时抛 ``NotImplementedError``，让
        akshare 的中证接口接管。硬套股票号段去猜后缀会把 ``000300`` 猜成 ``.SZ``，
        而错误的代码会让上游返回空结果——那会被缓存成"无数据"。
        """
        from finharness.data.mapping import FUYAO_INDEX_THSCODES

        resolved = FUYAO_INDEX_THSCODES.get(str(index).strip())
        if resolved is None:
            raise NotImplementedError(f"同花顺不识别指数 {index}")
        dataset = FUYAO_ENDPOINTS["index_constituents"]
        payload = self._call(
            _METHOD_SERVICE["index_constituents"], dataset, {"thscode": resolved}
        )
        frame = _payload_frame(payload)
        if not len(frame):
            raise AdapterError(f"{dataset}: 未返回指数 {index} 的成分股")
        keep = [c for c in ("symbol", "name", "thscode") if c in frame.columns]
        return FetchResult(df=frame[keep].reset_index(drop=True), interface=dataset)


def _service_for_dataset(dataset: str) -> str:
    """由数据集名前缀推导所属服务。

    同花顺的工具名以服务域为前缀（``get_fund_*``、``get_futures_*``…），因此这条
    推导不需要额外的映射表，也就不会随上游新增工具而失效。
    """
    if dataset.startswith("get_a_share_index_"):
        return "a-share-index"
    if dataset.startswith("get_meta_"):
        return "meta"
    if dataset.startswith("get_fund_"):
        return "fund"
    if dataset.startswith("get_futures_"):
        return "futures"
    if dataset.startswith("get_options_"):
        return "options"
    return "a-share"


def _period_to_date(period: str) -> pd.Timestamp:
    """``yyyy-N`` 报告期 -> 该期末的日期。

    N=1..4 分别对应一季报（0331）/中报（0630）/三季报（0930）/年报（1231）。
    """
    try:
        year, quarter = str(period).split("-")
        month = int(quarter) * 3
        day = 31 if month in (3, 12) else 30
        return pd.Timestamp(year=int(year), month=month, day=day)
    except (ValueError, TypeError):
        return pd.Timestamp(str(period))


def _statement_frame(items: list[Any]) -> pd.DataFrame:
    """把逐报告期的报表记录转置成"指标行 × 报告期列"的宽表。

    同花顺按报告期返回一行（``period_end_ms`` + 各字段），而本项目沿用的财报形态是
    指标为行、报告期为 ``YYYYMMDD`` 列——转置后才与 akshare 的摘要表一致，裁剪时
    的年度优先排序也才有依据。
    """
    rows: dict[str, dict[str, Any]] = {}
    period_columns: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        period_column = _period_column(item)
        if period_column is None:
            continue
        period_columns.append(period_column)
        for field, value in item.items():
            if field in FUYAO_MS_DATE_COLUMNS or field in {"period", "fiscal_year", "fiscal_period", "thscode", "ticker", "currency"}:
                continue
            label = FUYAO_FINANCIAL_LABELS.get(field, field)
            rows.setdefault(label, {})[period_column] = value
    # 报告期列按新到旧排列：调用方与裁剪逻辑都假定"第一列是最新一期"。
    ordered = sorted(set(period_columns), reverse=True)
    frame = pd.DataFrame(
        [{"指标": label, **{col: values.get(col) for col in ordered}} for label, values in rows.items()]
    )
    if not len(frame):
        return frame
    return frame[["指标", *ordered]]


def _period_column(item: dict[str, Any]) -> str | None:
    """由一条报表记录推导其 ``YYYYMMDD`` 报告期列名。"""
    raw = item.get("period_end_ms")
    if raw is None:
        raw = item.get("report_date_ms")
    if raw is None:
        return None
    moment = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
    if pd.isna(moment):
        return None
    local = moment.tz_convert(_SHANGHAI)
    return local.strftime("%Y%m%d")


def _payload_frame(payload: Any) -> pd.DataFrame:
    """把数据集载荷压成平面表。

    优先取记录数组（``item`` 是最常见的容器名，龙虎榜用 ``stock_items``），其次把
    标量字典作为单行。同名键不合并、缺失键留空——表格的列必须稳定，否则同一数据集
    在不同调用下会渲染出不同的形状。
    """
    if isinstance(payload, list):
        if payload and all(isinstance(entry, dict) for entry in payload):
            return _normalize_frame(_records_frame(payload))
        return _normalize_frame(pd.DataFrame(payload))
    if not isinstance(payload, dict):
        return pd.DataFrame([{"value": payload}])

    for key in ("item", "stock_items", "rows", "items"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            # 存在的容器即使为空也要如实返回空表：落到下面的标量分支会造出一行
            # ``item="[]"`` 的假记录，而"查到了但没有数据"与"查到一条数据"必须
            # 可分辨——前者不该被写进缓存。
            return _normalize_frame(_records_frame(candidate))

    flat = _flatten_record(payload)
    return _normalize_frame(pd.DataFrame([flat]))


def _looks_like_a_ms_timestamp(values: pd.Series) -> bool:
    """整列是否基本都由毫秒 Unix 时间戳构成。

    量级阈值是判定的核心：1e12 毫秒约等于 2001-09，而 1e11 对应 1973 年。真实的
    行情/财务时间戳都在 1e12 以上，因此用它把"时间戳"与"恰好同名的普通量"分开。
    numpy 的 datetime64 也会通过 ``to_numeric``，但那些已经是日期，量级判断自然
    不成立（内部存储是纳秒或天数），因此不会重复转换。

    要求"多数"非空值满足，而不是全部：整列里混入一个 0 或异常值不该让转换失效。
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if not len(numeric):
        return False
    return bool((numeric > 1e12).mean() >= 0.5)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """应用同花顺的列名映射与毫秒时间戳转换。

    改名与日期转换都在这里发生，因此适配器的每个出口都落到同一套内部列名上。

    时间戳转换按两条规则识别目标列：

    * ``FUYAO_MS_DATE_COLUMNS`` 登记的列**既改名又转换**（``date_ms -> date``），
      因此它们不出现在通用改名表里——两处都写会让"先改名再转换"之类的顺序错误
      看起来无害。
    * 其余列由**名字线索 + 量级**共同判定：名字以 ``_ms`` 结尾、或含 ``date``/
      ``time``，且整列基本是毫秒量级。只按名字不够——``nav_date`` 没有任何后缀，
      却正是裸毫秒（实测 ``1.78966e+12`` 直接显示给模型）；只按量级也不够——会把
      与时间无关的大数值误当日期。两条同时成立才转换。
    """
    if not len(frame):
        return frame
    date_columns = [source for source in FUYAO_MS_DATE_COLUMNS if source in frame.columns]
    for source in date_columns:
        target = FUYAO_MS_DATE_COLUMNS[source]
        frame[target] = _to_ms_datetime(frame[source])
        if target != source:
            frame = frame.drop(columns=[source])
    extra = [
        column
        for column in frame.columns
        if column in frame
        and str(column) not in FUYAO_MS_DATE_COLUMNS
        and _is_timestamp_like_name(str(column))
    ]
    for column in extra:
        values = frame[column]
        if not _looks_like_a_ms_timestamp(values):
            continue
        frame[column] = _to_ms_datetime(values)
    rename = {
        source: target
        for source, target in FUYAO_COLUMN_MAP.items()
        if source in frame.columns and target not in frame.columns
    }
    if rename:
        frame = frame.rename(columns=rename)
    return frame


def _is_timestamp_like_name(column: str) -> bool:
    """列名是否像时间字段（``_ms`` 后缀，或含 ``date``/``time``）。

    这是**线索**而非结论：调用方还要用量级复核，因为 ``turnover_ratio`` 这类名字
    也可能含 ``time`` 的子串（``turnover`` 里没有，但更长的名字可能有），而
    ``duration_ms`` 这类以 ``_ms`` 结尾的量纲确实不是时间戳。
    """
    lowered = column.lower()
    return lowered.endswith("_ms") or "date" in lowered or "time" in lowered


__all__ = ["FuyaoMcpAdapter"]
