"""akshare 适配器：候选接口链回退、列名归一化、限流。

akshare 是一个同步库，其上游主机可达性参差不齐（在本环境中 EM 的 ``push2``
主机不可达，而新浪主机可用）。因此每个语义化抓取都会按
``mapping.AKSHARE_ENDPOINTS`` 的顺序尝试候选接口，并记录实际提供数据的那个
接口。
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.adapters.pacing import pace
from finharness.data.adapters.retry import retry_call
from finharness.data.lookback import lookback_stamp, lookback_start
from finharness.data.mapping import (
    AKSHARE_ENDPOINTS,
    AKSHARE_INTERFACE_COLUMNS,
    MACRO_INDICATORS,
    PEER_COMPANY,
    PEER_IDENTITY_COLUMNS,
    PEER_ROW_TYPE_COLUMN,
    PEER_STAT,
    PEER_STAT_LABELS,
    VALUATION_INDICATOR_UNITS,
    is_index_symbol,
    normalize_index,
    prefixed_symbol,
    select_indicator_columns,
)

_PERIOD_MAP = {"day": "daily", "week": "weekly", "month": "monthly"}
_PERIOD_LABEL = {1: "近一年", 2: "近一年", 3: "近三年", 5: "近五年"}

# --- 行情候选接口的时间预算 -----------------------------------------------
# 快照类数据源是同步、分页的，而且（对于 EM/Sina）未设置请求超时，因此一个
# 不可达的主机就可能耗尽整个 30s 的工具预算，并取消排在它后面的候选接口。
# 每个行情候选接口都由一个墙钟时限约束，而持续失败的接口会被跳过并进入
# 冷却期，而不是在每次调用时重复重试。
_QUOTE_CANDIDATE_DEADLINE_S = 8.0
_QUOTE_REQUEST_TIMEOUT_S = 8.0  # 传给接受 timeout 参数的接口
# K 线序列比快照大，正常返回本就慢一些，因此给更宽的时限；但它同样没有设置
# 请求超时，一个挂死的接口会吃光整个工具体预算（横截面回测一次要取数百只），
# 因此候选链上的每次尝试都必须有时限，超时即弃并落到下一个候选。
_KLINE_CANDIDATE_DEADLINE_S = 15.0
# 财务指标接口同样是一个没有请求超时的阻塞调用，因此也需要墙钟时限；否则一个挂死的
# 上游会吃光工具预算。15s 与 K 线同档：指标接口本身返回就慢一些，但必须给上限。
_INDICATORS_CANDIDATE_DEADLINE_S = 15.0
_UNHEALTHY_AFTER_FAILURES = 2
_UNHEALTHY_COOLDOWN_S = 300.0


class _DeadlineExceeded(Exception):
    """候选接口未在其墙钟预算内返回。"""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"未在 {seconds:.1f}s 内返回")
        self.seconds = seconds


def _call_with_deadline(build: Callable[[], Any], *, deadline_s: float) -> Any:
    """以墙钟时限约束运行一个阻塞式可调用对象。

    没有设置超时的 ``requests`` 调用无法从 Python 侧中断，因此该调用运行在一个
    守护线程上，一旦超时就被遗弃。接口失败记忆机制让这种遗弃很少发生：超时的
    候选接口会被跳过并进入冷却期，而不是在每次查询时重新启动。使用守护线程是
    有意为之——非守护线程会阻塞解释器的关闭。
    """
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = build()
        except BaseException as exc:  # noqa: BLE001 - 在调用方的线程上重新抛出
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(deadline_s)
    if worker.is_alive():
        raise _DeadlineExceeded(deadline_s)
    if "error" in box:
        raise box["error"]
    return box["value"]


def _import_akshare():
    """惰性导入 akshare，以便在未安装它时包仍可被导入。"""
    import akshare as ak  # noqa: PLC0415 - 刻意延迟导入

    return ak


def _normalize(df: pd.DataFrame, interface: str) -> pd.DataFrame:
    """按接口的列名映射重命名列，并尽量把 ``date`` 列转换为时间类型。"""
    rename = AKSHARE_INTERFACE_COLUMNS.get(interface)
    if rename:
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _slice_years(df: pd.DataFrame, years: int, *, date_col: str = "date") -> pd.DataFrame:
    """仅保留最近 ``years`` 年内的行；容忍缺失日期列的情况。"""
    if date_col not in df.columns or years <= 0:
        return df
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    cutoff = pd.Timestamp(lookback_start(years))
    trimmed = frame[frame[date_col] >= cutoff]
    return trimmed if len(trimmed) else frame


def _label_peer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """为行业均值/中位数行打上标记，避免它们被误读为公司。

    EM 的对比表把聚合值当作普通行返回，其 代码/简称 中是字面标签；如果不加
    标记，消费方在对表求均值（或取第 0 行）时会悄悄地把聚合值当成某家发行公司。
    """
    if PEER_ROW_TYPE_COLUMN in df.columns:
        return df
    for column in ("代码", "简称"):
        if column not in df.columns:
            continue
        labels = df[column].astype(str)
        if not labels.isin(PEER_STAT_LABELS).any():
            continue
        frame = df.copy()
        frame.insert(
            0,
            PEER_ROW_TYPE_COLUMN,
            labels.map(lambda value: PEER_STAT if value in PEER_STAT_LABELS else PEER_COMPANY),
        )
        return frame
    return df


_PERIOD_RE = re.compile(r"(\d{4})\s*年\s*(?:第)?\s*(\d{1,2})\s*(?:[-–]\s*\d{1,2})?\s*(?:月|季度)份?")


def _parse_macro_period(value: Any) -> pd.Timestamp:
    """解析 akshare 宏观数据表所使用的期间标签。

    实际见过的格式有：``2026年08月份``（月度）、``2026年第1-2季度``
    （累计季度）、``201501``（部分月度表），以及普通的 ISO 日期（日频序列）。
    累计的 ``第1-2季度`` 行锚定到它的第一个季度，避免该点被重复计算。
    """
    text = str(value).strip()
    match = _PERIOD_RE.search(text)
    if match:
        year, part = int(match.group(1)), int(match.group(2))
        if "季度" in text or "季" in text:
            month = (part - 1) * 3 + 1
        else:
            month = part
        return pd.Timestamp(year=year, month=month, day=1)
    if re.fullmatch(r"\d{6}", text):
        return pd.Timestamp(year=int(text[:4]), month=int(text[4:6]), day=1)
    return pd.to_datetime(text, errors="coerce")


class AkShareAdapter(DataAdapter):
    name = "akshare"

    def __init__(self, *, throttle_seconds: float = 1.0) -> None:
        self.throttle_seconds = throttle_seconds
        self._last_call = 0.0
        # 接口健康状态在整个进程内共享（一个适配器服务所有会话），因此死掉的
        # 接口不会被每个新查询重新探测。
        self._failures: dict[str, int] = {}
        self._unhealthy_until: dict[str, float] = {}
        self._health_lock = threading.Lock()

    def _throttle(self) -> None:
        self._last_call = pace(self._last_call, self.throttle_seconds)

    # -- 接口健康状态 ---------------------------------------------------------
    def _interface_available(self, interface: str) -> bool:
        """接口处于失败后的冷却期内时返回 False。"""
        with self._health_lock:
            until = self._unhealthy_until.get(interface)
            if until is None:
                return True
            if time.monotonic() >= until:
                self._unhealthy_until.pop(interface, None)
                self._failures.pop(interface, None)
                return True
            return False

    def _note_success(self, interface: str) -> None:
        with self._health_lock:
            self._failures.pop(interface, None)
            self._unhealthy_until.pop(interface, None)

    def _note_failure(self, interface: str) -> None:
        with self._health_lock:
            count = self._failures.get(interface, 0) + 1
            self._failures[interface] = count
            if count >= _UNHEALTHY_AFTER_FAILURES:
                self._unhealthy_until[interface] = time.monotonic() + _UNHEALTHY_COOLDOWN_S

    def _call(
        self,
        interface: str,
        build: Callable[[Any], pd.DataFrame],
        *,
        deadline_s: float | None = None,
    ) -> pd.DataFrame:
        """运行单个候选接口，把任何失败转换为 AdapterError。

        ``deadline_s`` 约束网络调用（而非限流），这样挂起的源会退到下一个候选
        接口，而不会耗尽调用方的预算。
        """
        ak = _import_akshare()
        self._throttle()
        try:
            if deadline_s is None:
                df = build(ak)
            else:
                df = _call_with_deadline(lambda: build(ak), deadline_s=deadline_s)
        except _DeadlineExceeded as exc:
            raise AdapterError(f"{interface}: 超时未返回（>{exc.seconds:.0f}s）", retryable=True) from exc
        except Exception as exc:  # noqa: BLE001 - 所有数据源错误都汇聚于此
            raise AdapterError(
                f"{interface}: {type(exc).__name__}: {exc}",
                retryable=type(exc).__name__ in {"ConnectionError", "Timeout", "TimeoutError"},
            ) from exc
        if not isinstance(df, pd.DataFrame):
            raise AdapterError(f"{interface}: 返回非表格数据")
        return df

    # -- 语义化抓取 -----------------------------------------------------------
    def fetch_quote(self, symbol: str) -> FetchResult:
        """最新价格，优先尝试数据最丰富的快照源。

        处于失败后冷却期的候选接口会被跳过；如果没有任何健康接口，则重试完整
        的候选链，这样恢复后的源能再次被选用。每次尝试都受时限约束，因此挂起
        的源无法耗尽调用方的全部预算。
        """
        errors: list[str] = []
        candidates = AKSHARE_ENDPOINTS["quote"]
        healthy = [interface for interface in candidates if self._interface_available(interface)]
        for interface in (healthy or list(candidates)):
            try:
                if interface == "stock_zh_a_spot_em":
                    df = _normalize(
                        self._call(
                            interface,
                            lambda ak: ak.stock_zh_a_spot_em(),
                            deadline_s=_QUOTE_CANDIDATE_DEADLINE_S,
                        ),
                        "quote",
                    )
                    if "symbol" not in df.columns:
                        raise AdapterError(f"{interface}: 缺少代码列")
                    hit = df[df["symbol"].astype(str).str.zfill(6) == symbol]
                elif interface == "stock_zh_a_spot":
                    raw = self._call(
                        interface,
                        lambda ak: ak.stock_zh_a_spot(),
                        deadline_s=_QUOTE_CANDIDATE_DEADLINE_S,
                    )
                    df = raw.rename(
                        columns={"代码": "symbol", "名称": "name", "最新价": "close",
                                 "涨跌幅": "pct_change", "成交量": "volume"}
                    )
                    if "symbol" not in df.columns:
                        raise AdapterError(f"{interface}: 缺少代码列")
                    hit = df[df["symbol"].astype(str).str.zfill(6) == symbol]
                else:
                    # 腾讯日线序列：最后一行即最新收盘价。
                    end = date.today().strftime("%Y%m%d")
                    start = (date.today() - timedelta(days=14)).strftime("%Y%m%d")
                    df = self._call(
                        interface,
                        lambda ak: ak.stock_zh_a_hist_tx(
                            symbol=prefixed_symbol(symbol, lower=True),
                            start_date=start,
                            end_date=end,
                            timeout=_QUOTE_REQUEST_TIMEOUT_S,
                        ),
                        deadline_s=_QUOTE_CANDIDATE_DEADLINE_S,
                    )
                    if not len(df):
                        raise AdapterError(f"{interface}: 返回空表")
                    latest = df.tail(1).copy()
                    latest["symbol"] = symbol
                    hit = latest
                if not len(hit):
                    raise AdapterError(f"{interface}: 未找到代码 {symbol}")
                self._note_success(interface)
                return FetchResult(df=hit.reset_index(drop=True), interface=interface)
            except AdapterError as exc:
                self._note_failure(interface)
                errors.append(exc.message)
        raise AdapterError("; ".join(errors) or "quote 无可用接口")

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int) -> pd.DataFrame:
        """按候选链抓取 K 线，统一列名并按最新在前排序。

        指数代码（``INDEX_ALIASES`` 收录的中证/上证指数）走 ``index_kline`` 候选链：
        股票端点会按号段拼市场（``000300`` 被当成深市股票查成空表），且新浪股票
        端点对 ``sz000985`` 实测返回的是**深市股票**的数据——同一串数字在两个体系
        里是不同的证券，不分流就会把错的数据缓存下来。指数无复权概念，
        ``adjust`` 被忽略。
        """
        if is_index_symbol(symbol):
            return self._fetch_index_kline(symbol, period, years)
        em_period = _PERIOD_MAP.get(period, "daily")
        errors: list[str] = []
        for interface in AKSHARE_ENDPOINTS["kline"]:
            try:
                if interface == "stock_zh_a_hist":
                    start = lookback_stamp(years, extra_days=30)
                    end = date.today().strftime("%Y%m%d")
                    df = self._call(
                        interface,
                        lambda ak: ak.stock_zh_a_hist(
                            symbol=symbol, period=em_period, start_date=start,
                            end_date=end, adjust=adjust or "",
                        ),
                        deadline_s=_KLINE_CANDIDATE_DEADLINE_S,
                    )
                else:  # 新浪 / 腾讯变体需要交易所前缀
                    prefixed = prefixed_symbol(symbol, lower=True)
                    if interface == "stock_zh_a_daily":
                        df = self._call(
                            interface,
                            lambda ak: ak.stock_zh_a_daily(symbol=prefixed, adjust=adjust or ""),
                            deadline_s=_KLINE_CANDIDATE_DEADLINE_S,
                        )
                    else:
                        start = lookback_stamp(years, extra_days=30)
                        end = date.today().strftime("%Y%m%d")
                        df = self._call(
                            interface,
                            lambda ak: ak.stock_zh_a_hist_tx(
                                symbol=prefixed, start_date=start, end_date=end
                            ),
                            deadline_s=_KLINE_CANDIDATE_DEADLINE_S,
                        )
                df = _normalize(df, "kline")
                if not len(df):
                    raise AdapterError(f"{interface}: 返回空表")
                # 各数据源的行序不同（EM 升序，Sina 升序）；内部契约为最新在前，
                # 这样摘要和均线窗口读到的总是最新的周期。
                df = _slice_years(df, years)
                if "date" in df.columns:
                    df = df.sort_values("date", ascending=False)
                return FetchResult(df=df.reset_index(drop=True), interface=interface)
            except AdapterError as exc:
                errors.append(exc.message)
        raise AdapterError("; ".join(errors) or "kline 无可用接口")

    def _fetch_index_kline(self, symbol: str, period: str, years: int) -> FetchResult:
        """指数 K 线候选链（``index_kline``）。

        - 东财 ``index_zh_a_hist``：裸代码 + 周期/起止日期；列名中文，与股票 kline
          的映射表同形，复用 ``_normalize``。
        - 腾讯 ``stock_zh_index_daily_tx``：``sh`` 前缀 + 起止日期。
        - 新浪 ``stock_zh_index_daily``：``sh`` 前缀，无日期参数。

        当前收录的指数全部在上交所发布，统一拼 ``sh``。

        指数没有复权概念，各接口都没有 ``adjust`` 参数——调用方传入的复权意图对
        指数不适用，忽略而非报错，让"取沪深300近期走势"这类请求不必先知道这条
        冷知识。

        **陈旧防护。** 新浪对个别指数只提供一段早已停止更新的序列：实测
        ``sh000985`` 返回的是 2011–2016 年的数据（中证全指），而 ``sh000300``
        是完整的。若不加判定，这份陈旧序列会被 ``_slice_years`` 的兜底（区间内
        无行时返回原表）原样交出，被渲染成"近一年走势"——比"取不到"危险得多。
        因此这里要求序列的**最新一行落在请求窗口内**，否则按"该源无可用数据"
        继续下一候选；全部候选都陈旧则整体失败（宁可报错也不给错数据）。
        """
        em_period = _PERIOD_MAP.get(period, "daily")
        start = lookback_stamp(years, extra_days=30)
        end = date.today().strftime("%Y%m%d")
        cutoff = pd.Timestamp(lookback_start(years))
        errors: list[str] = []
        for interface in AKSHARE_ENDPOINTS["index_kline"]:
            try:
                if interface == "index_zh_a_hist":
                    df = self._call(
                        interface,
                        lambda ak: ak.index_zh_a_hist(
                            symbol=symbol, period=em_period,
                            start_date=start, end_date=end,
                        ),
                        deadline_s=_KLINE_CANDIDATE_DEADLINE_S,
                    )
                elif interface == "stock_zh_index_daily_tx":
                    df = self._call(
                        interface,
                        lambda ak: ak.stock_zh_index_daily_tx(
                            symbol=f"sh{symbol}", start_date=start, end_date=end
                        ),
                        deadline_s=_KLINE_CANDIDATE_DEADLINE_S,
                    )
                else:  # 新浪：无日期参数，返回全量历史
                    df = self._call(
                        interface,
                        lambda ak: ak.stock_zh_index_daily(symbol=f"sh{symbol}"),
                        deadline_s=_KLINE_CANDIDATE_DEADLINE_S,
                    )
                df = _normalize(df, "kline")
                if not len(df):
                    raise AdapterError(f"{interface}: 返回空表")
                if "date" in df.columns:
                    latest = pd.to_datetime(df["date"], errors="coerce").max()
                    if pd.notna(latest) and latest < cutoff:
                        raise AdapterError(
                            f"{interface}: 数据陈旧（最新 {latest.date()}，早于请求窗口）"
                        )
                df = _slice_years(df, years)
                if "date" in df.columns:
                    df = df.sort_values("date", ascending=False)
                return FetchResult(df=df.reset_index(drop=True), interface=interface)
            except AdapterError as exc:
                errors.append(exc.message)
        raise AdapterError("; ".join(errors) or "指数 K 线无可用接口")

    def fetch_indicators(self, symbol: str, years: int, fields: list[str] | None) -> FetchResult:
        """抓取财务分析指标，统一日期列名并将日期降序排列。"""
        interface = AKSHARE_ENDPOINTS["indicators"][0]
        start_year = str(date.today().year - max(years, 1))
        df = self._call(
            interface,
            lambda ak: ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year),
            deadline_s=_INDICATORS_CANDIDATE_DEADLINE_S,
        )
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", ascending=False)
        if fields:
            keep, _unmatched = select_indicator_columns(df.columns, fields)
            df = df[keep]
        return FetchResult(df=df.reset_index(drop=True), interface=interface)

    def fetch_financials(self, symbol: str, statement: str, years: int) -> FetchResult:
        """抓取财务摘要，仅保留最近 ``years`` 年的期间列。"""
        interface = AKSHARE_ENDPOINTS["financials"][0]
        df = self._call(interface, lambda ak: ak.stock_financial_abstract(symbol=symbol))
        # 期间列是 YYYYMMDD 格式；只保留最近 N 年的期间列。
        period_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
        keep_periods = sorted(period_cols, reverse=True)[: max(years, 1) * 4]
        base = [c for c in df.columns if c not in period_cols]
        return FetchResult(
            df=df[base + sorted(keep_periods, reverse=True)].reset_index(drop=True),
            interface=interface,
        )

    def fetch_valuation(self, symbol: str, lookback_years: int, indicator: str) -> FetchResult:
        """针对单个指标返回一条估值序列。

        百度的接口根据 ``indicator`` 切换指标，但返回的是一个只有 ``date``/
        ``value`` 的裸表，既不含指标名也不含单位，因此这里把两者都附加上去；
        一个市值数字若仅以未标注的 ``value`` 呈现，就会出现此前被误读为市盈率
        倍数的情况。
        """
        interface = AKSHARE_ENDPOINTS["valuation"][0]
        period = _PERIOD_LABEL.get(lookback_years, "近一年")
        df = self._call(
            interface,
            lambda ak: ak.stock_zh_valuation_baidu(
                symbol=symbol, indicator=indicator, period=period
            ),
        )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", ascending=False)
        unit = VALUATION_INDICATOR_UNITS.get(indicator, "")
        df = df.rename(columns={"value": f"{indicator}({unit})" if unit else indicator})
        return FetchResult(df=df.reset_index(drop=True), interface=interface)

    def fetch_peers(self, industry: str, fields: list[str] | None) -> FetchResult:
        """``industry`` 参数携带用于同业比较的目标代码。"""
        interface = AKSHARE_ENDPOINTS["peers"][0]
        target = prefixed_symbol(industry)
        df = self._call(interface, lambda ak: ak.stock_zh_valuation_comparison_em(symbol=target))
        df = _label_peer_rows(df)
        if fields:
            matched = [c for c in df.columns if any(f in str(c) for f in fields)]
            # 身份列和行标记属于契约内容，不是过滤素材：一张无法说明某行属于谁
            # 的过滤后表格是不可用的。
            keep = [
                c for c in (PEER_ROW_TYPE_COLUMN, *PEER_IDENTITY_COLUMNS) if c in df.columns
            ]
            keep += [c for c in matched if c not in keep]
            df = df[keep]
        # 公司在前、聚合值在后：这样对整表求均值时会明显出错，而不是悄悄地把
        # 行业统计值也算进去。
        if PEER_ROW_TYPE_COLUMN in df.columns:
            rank = {PEER_COMPANY: 0, PEER_STAT: 1}
            df = df.sort_values(
                PEER_ROW_TYPE_COLUMN, key=lambda col: col.map(rank), kind="stable"
            )
        return FetchResult(df=df.reset_index(drop=True), interface=interface)

    def fetch_news(self, symbol: str | None, topic: str | None, top_n: int) -> FetchResult:
        interface = AKSHARE_ENDPOINTS["news"][0]
        query = symbol or topic or "A股"
        df = self._call(interface, lambda ak: ak.stock_news_em(symbol=query))
        return FetchResult(df=df.head(top_n).reset_index(drop=True), interface=interface)

    def fetch_announcements(self, symbol: str, since: str, top_n: int) -> FetchResult:
        interface = AKSHARE_ENDPOINTS["announcements"][0]
        begin = since.replace("-", "")
        end = date.today().strftime("%Y%m%d")
        df = self._call(
            interface,
            lambda ak: ak.stock_individual_notice_report(
                security=symbol, symbol="全部", begin_date=begin, end_date=end
            ),
        )
        return FetchResult(df=df.head(top_n).reset_index(drop=True), interface=interface)

    # -- 宏观 / 行业 ----------------------------------------------------------
    def _macro_source_frame(self, source: str, years: int) -> pd.DataFrame:
        """抓取单个宏观接口；调用方按数据源对指标进行分组。"""
        start = lookback_stamp(years, extra_days=370)
        if source == "pmi":
            return self._call("macro_china_pmi", lambda ak: ak.macro_china_pmi())
        if source == "cpi":
            return self._call("macro_china_cpi", lambda ak: ak.macro_china_cpi())
        if source == "ppi":
            return self._call("macro_china_ppi", lambda ak: ak.macro_china_ppi())
        if source == "money_supply":
            return self._call("macro_china_money_supply", lambda ak: ak.macro_china_money_supply())
        if source == "shrzgm":
            return self._call("macro_china_shrzgm", lambda ak: ak.macro_china_shrzgm())
        if source == "lpr":
            return self._call("macro_china_lpr", lambda ak: ak.macro_china_lpr())
        if source == "shibor":
            return self._call("macro_china_shibor_all", lambda ak: ak.macro_china_shibor_all())
        if source == "bond":
            return self._call(
                "bond_zh_us_rate",
                lambda ak: ak.bond_zh_us_rate(start_date=start),
            )
        if source == "currency":
            end = date.today().strftime("%Y%m%d")
            return self._call(
                "currency_boc_sina",
                lambda ak: ak.currency_boc_sina(symbol="美元", start_date=start, end_date=end),
            )
        if source == "gdp":
            return self._call("macro_china_gdp", lambda ak: ak.macro_china_gdp())
        raise AdapterError(f"未知宏观数据源：{source}")

    def fetch_macro(self, indicators: list[str], years: int) -> FetchResult:
        """长格式的宏观数据表：``date | indicator | value``（外加 label/unit）。

        指标按接口分组，使得一次抓取即可服务该接口承载的所有序列（PMI 有两条，
        货币供应量有三条）；结果采用纵向堆叠而非横向展开，因为这些序列频率不同，
        横向连接会产生大量 NaN。
        """
        resolved: list[tuple[str, Any]] = []
        for slug in indicators:
            spec = MACRO_INDICATORS.get(slug)
            if spec is None:
                raise AdapterError(f"未知宏观指标：{slug}")
            resolved.append((slug, spec))

        frames: dict[str, pd.DataFrame] = {}
        rows: list[pd.DataFrame] = []
        for slug, spec in resolved:
            if spec.source not in frames:
                frames[spec.source] = self._macro_source_frame(spec.source, years)
            source_df = frames[spec.source]
            date_col = spec.period_col
            value_col = next((c for c in spec.value_col if c in source_df.columns), None)
            if date_col not in source_df.columns or value_col is None:
                continue
            block = pd.DataFrame(
                {
                    "date": source_df[date_col].map(_parse_macro_period),
                    "indicator": slug,
                    "label": spec.label,
                    "value": pd.to_numeric(source_df[value_col], errors="coerce"),
                    "unit": spec.unit,
                }
            ).dropna(subset=["date"])
            if slug == "usdcny":
                # 中行按每 100 美元报价；归一化为每美元兑人民币，使该序列呈现为
                # 大家熟悉的 ~7 水平而不是 ~700。
                block["value"] = block["value"] / 100
            rows.append(block)

        if not rows:
            raise AdapterError("宏观接口未返回所选指标的数据")
        combined = pd.concat(rows, ignore_index=True)
        cutoff = pd.Timestamp(lookback_start(years, extra_days=370))
        combined = combined[combined["date"] >= cutoff]
        combined = combined.sort_values(["indicator", "date"], ascending=[True, False])
        return FetchResult(df=combined.reset_index(drop=True), interface="macro_china")

    def _sw_table(self, table: str, *, attempts: int = 3) -> pd.DataFrame | None:
        """抓取申万行业表，并针对瞬时限流进行重试。

        申万主机在限流时会间歇性地返回空响应体（akshare 随后抛出
        ``'NoneType' has no attribute 'find_all'``）。这类失败是瞬时的，所以
        一次短暂重试挽救该表的概率远高于其耗费的时间；而硬性失败仍会暴露出来。
        """
        return retry_call(
            lambda: self._call(table, lambda ak, t=table: getattr(ak, t)()),
            attempts=attempts,
            delay_for=lambda attempt: 0.6 * attempt,
            retry_on=AdapterError,
            exhausted=None,
        )

    def _sw_code(self, industry: str) -> str:
        """将行业名称/代码解析为申万指数代码（例如 801010）。

        优先尝试一级行业名称，然后尝试二级行业（"白酒" -> "白酒Ⅱ"），因为调用方
        自然会说出它们关心的子行业。两侧都会去掉罗马数字后缀，使 "白酒" 能匹配
        到 "白酒Ⅱ"。二级查询是尽力而为的：当其表格无法抓取时，一级匹配仍然可用，
        错误信息也会列出当时可用的选项。
        """
        key = str(industry).strip()
        if not key:
            raise AdapterError("行业名不能为空")
        # 显式给出的代码直接使用（一级和二级共用该指数路由）。
        if key.isdigit() and len(key) == 6:
            return key

        def normalise(value: Any) -> str:
            text = str(value).strip().replace(" ", "")
            for suffix in ("Ⅰ", "Ⅱ", "Ⅲ", "I", "II", "III"):
                if text.endswith(suffix):
                    text = text[: -len(suffix)]
            return text

        target = normalise(key)
        available: list[str] = []
        for table in ("sw_index_first_info", "sw_index_second_info"):
            raw = self._sw_table(table)
            if raw is None or not len(raw) or "行业代码" not in raw.columns:
                continue
            names = raw["行业名称"].astype(str).map(normalise)
            available.extend(raw["行业名称"].astype(str).tolist())
            exact = raw[names == target]
            if len(exact):
                return str(exact.iloc[0]["行业代码"]).split(".")[0]
            prefix = raw[names.str.startswith(target)]
            if len(prefix):
                return str(prefix.iloc[0]["行业代码"]).split(".")[0]
        raise AdapterError(
            f"未找到申万行业：{industry}；可选（一级/二级）：{'、'.join(available[:60])}"
        )

    def fetch_industry_perf(self, industry: str | None, years: int) -> FetchResult:
        """无 industry -> 返回申万一级行业总览；指定 industry -> 返回其指数历史。"""
        if not industry:
            df = self._sw_table("sw_index_first_info")
            if df is None or not len(df):
                raise AdapterError("申万一级行业总览暂时不可用（接口限流），请稍后重试或指定具体行业")
            return FetchResult(df=df.reset_index(drop=True), interface="sw_index_first_info")
        code = self._sw_code(industry)
        df = self._call(
            "index_hist_sw",
            lambda ak: ak.index_hist_sw(symbol=code, period="day"),
        )
        # 指数自身的代码不是可交易的证券代码；给它单独命名，以免在渲染后的表格
        # 中被误当成股票代码。
        df = df.rename(
            columns={
                "代码": "index_code", "日期": "date", "收盘": "close", "开盘": "open",
                "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
            }
        )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = _slice_years(df, years)
        if "date" in df.columns:
            df = df.sort_values("date", ascending=False)
        return FetchResult(df=df.reset_index(drop=True), interface="index_hist_sw")

    def fetch_industry_constituents(self, industry: str) -> FetchResult:
        """解析行业代码后抓取其成分股，统一为 symbol/name 两列。"""
        code = self._sw_code(industry)
        df = self._call(
            "index_component_sw",
            lambda ak: ak.index_component_sw(symbol=code),
        )
        df = df.rename(columns={"证券代码": "symbol", "证券名称": "name"})
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        return FetchResult(df=df.reset_index(drop=True), interface="index_component_sw")

    def fetch_index_constituents(self, index: str) -> FetchResult:
        """解析指数代码后抓取其成分股，仅保留 symbol/name 两列。"""
        code = normalize_index(index)
        df = self._call(
            "index_stock_cons_csindex",
            lambda ak: ak.index_stock_cons_csindex(symbol=code),
        )
        df = df.rename(columns={"成分券代码": "symbol", "成分券名称": "name"})
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        keep = [c for c in ("symbol", "name") if c in df.columns]
        return FetchResult(df=df[keep].reset_index(drop=True), interface="index_stock_cons_csindex")
