"""同花顺长尾数据集派发器（docs 03.4）。

**为什么是派发器而不是几十个 typed 工具。** 同花顺经 MCP 暴露 78 个端点，其中
``get_a_share_financials_*`` 这类核心端点已由 typed 适配器方法覆盖（因此既有工具
无需改动）。余下的长尾——特色数据（涨停池/龙虎榜/热股榜/异动/集合竞价）、基金、
期货、期权、同花顺概念指数、标的检索——彼此参数与结果形态各异，为每个端点手写一个
工具意味着手抄 70 份 schema，而那份拷贝会在上游改动时静默过期。

因此这里的形状是：**参数 schema 在运行时从服务端 ``tools/list`` 读取**，工具只
声明一个 ``dataset`` 名与一个参数字典，并在调用前对照那份 schema 校验。上游新增
端点无需改任何本地代码。

**校验是拒绝而不是过滤。** 未知的 dataset 名、未知的参数名、缺失的必填参数一律
``ValueError``（转成 ``ok=False``）。静默丢弃一个不认识的参数会让模型以为自己查了
某个筛选条件，而实际上拿到的是全量 30 条——那种错误在结果里看不出来。

**外发确认。** 这些工具会向同花顺发出请求，因此声明 ``egress=True``，首次调用须经
用户确认（docs 03.7.1）。确认按类别共享，所以一次应答覆盖本对话内所有同花顺调用。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from finharness.data.mapping import (
    FUYAO_SERVICES,
    fuyao_arguments,
    fuyao_dataset_params,
    fuyao_inherited_defaults,
    fuyao_param_summary,
    fuyao_required_params,
)
from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, Tier, ToolGroup, param, tool

# 目录结果里每个数据集最多展示多少个参数名。参数多的端点（期货、基金 F10）有十几个，
# 全列会把目录本身撑爆，而模型真正需要的是"有这么个数据集、能按什么筛"。
_MAX_PARAMS_SHOWN = 12


def _find_dataset(catalog: list[dict[str, Any]], dataset: str) -> dict[str, Any] | None:
    for entry in catalog:
        if str(entry.get("name")) == dataset:
            return entry
    return None


def _schema_of(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """取数据集的 ``inputSchema``；没有目录条目时返回 None（跳过校验）。"""
    if not isinstance(entry, dict):
        return None
    schema = entry.get("inputSchema") or entry.get("input_schema")
    return schema if isinstance(schema, dict) else None


def _validate_arguments(
    dataset: str,
    arguments: dict[str, Any],
    schema: dict[str, Any] | None,
    *,
    extra_accepted: tuple[str, ...] = (),
    required_aliases: dict[str, set[str]] | None = None,
) -> None:
    """对照上游 schema 校验参数；不合规即 ``ValueError``。

    只做三件能明确判定的事：未知参数名、缺失必填、以及枚举越界。不做完整的 JSON
    Schema 校验（那需要引入依赖，而这些端点的失败模式几乎全落在前三类）。schema
    取不到时跳过——宁可让上游去报错，也不要把一次本可成功的调用挡在本地。

    ``required_aliases`` 让便捷参数也能满足必填：上游要求 ``thscode``，而调用方写的是
    ``symbol``（已在翻译阶段变成 ``thscode``，名字对得上）；但上游要求 ``thscode``
    而调用方写 ``symbols`` 时译出的名字是 ``thscodes``，若只看译后的键就会误报缺参。
    """
    declared = fuyao_dataset_params(schema)
    if not declared:
        return
    accepted = set(declared) | set(extra_accepted)
    unknown = [name for name in arguments if name not in accepted]
    if unknown:
        raise ValueError(
            f"数据集 {dataset} 不支持参数 {'、'.join(sorted(unknown))}；"
            f"可用参数：{'、'.join(declared)}"
        )
    aliases = required_aliases or {}
    missing = [
        name
        for name in fuyao_required_params(schema)
        if name not in arguments and not (aliases.get(name, set()) & set(arguments))
    ]
    if missing:
        raise ValueError(f"数据集 {dataset} 缺少必填参数：{'、'.join(missing)}")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return
    for name, value in arguments.items():
        spec = properties.get(name)
        if not isinstance(spec, dict):
            continue
        allowed = spec.get("enum")
        if isinstance(allowed, (list, tuple)) and allowed and value not in allowed:
            raise ValueError(
                f"数据集 {dataset} 的参数 {name} 取值 {value!r} 不合法；"
                f"可选：{'、'.join(str(item) for item in allowed)}"
            )


def _describe_dataset(entry: dict[str, Any]) -> str:
    """目录里一个数据集的一行摘要。"""
    name = str(entry.get("name") or "")
    description = str(entry.get("description") or "").strip()
    summary = fuyao_param_summary(_schema_of(entry))
    parts = [f"- `{name}`"]
    if description:
        parts.append(description)
    if summary:
        params = summary.split(", ")
        shown = "、".join(params[:_MAX_PARAMS_SHOWN])
        if len(params) > _MAX_PARAMS_SHOWN:
            shown += f" 等 {len(params)} 个"
        parts.append(f"参数：{shown}")
    return "；".join(parts)


def _filter_rows(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    """在**已返回**的行中按子串匹配（不区分大小写），跨所有字符串列。

    **为什么需要它。** 长尾数据集常有几百行的清单类结果——同花顺概念指数 390 个、
    行业 320 个、指数成分股 300 只——而渲染窗口只有 20 行。目标行落在窗口之外时，
    模型在结果里根本看不到它。实测教训：问「白酒概念板块成分股」时，"白酒概念"
    确实在 390 个概念里，但被截断掉了，模型于是转去逐个翻成分股，耗尽轮次仍未答出。

    **作用范围必须如实标注。** 这是本地过滤：上游分页时只看得到已取回的那一页，
    因此"没匹配到"不等于"上游没有"。调用方（render）据此给出那条提示，避免把一次
    本地过滤失手说成一条全局结论。
    """
    needle = query.lower()
    mask = pd.Series(False, index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            mask |= series.astype(str).str.lower().str.contains(needle, regex=False, na=False)
    return frame[mask].reset_index(drop=True)


class _FuyaoDatasetTool(BaseTool):
    """数据集派发器的共用实现；子类只声明它受理哪些服务。"""

    # 该工具受理的服务名。数据集名与服务的对应关系由名字前缀推导
    # （``finharness.data.adapters.fuyao_adapter._service_for_dataset``），因此这里
    # 只需列出边界。
    _services: tuple[str, ...] = ()

    async def _dispatch(
        self,
        *,
        dataset: str,
        params: dict[str, Any] | None = None,
        query: str | None = None,
    ) -> RawData:
        """校验 dataset 与参数，然后经数据门面取数。"""
        service = self._resolve_service(dataset)
        catalog = await self.data.dataset_catalog(service)
        entry = _find_dataset(catalog, dataset)
        if entry is None:
            known = "、".join(sorted(str(item.get("name")) for item in catalog))
            raise ValueError(
                f"数据集 {dataset!r} 在 {service} 服务中不存在；可用数据集：{known}。"
                "如需检索，先调用 list_fuyao_datasets"
            )
        schema = _schema_of(entry)
        # 便捷参数（symbol/symbols）映射到上游的 thscode/thscodes，因此校验时要把
        # 它们算作"已接受"，否则模型按文档传 symbol 会被当成非法参数拒绝。
        arguments = fuyao_arguments(params, service=service)
        _validate_arguments(
            dataset,
            arguments,
            schema,
            extra_accepted=("thscode", "thscodes"),
            # 上游要求 thscode 而调用方写 symbols 时，译出的键是 thscodes；把两者的
            # 关系告诉校验器，否则一次合法调用会被误报为缺少必填参数。
            required_aliases={"thscode": {"thscodes"}},
        )
        raw = await self.data.query_dataset(service, dataset, arguments)
        raw = self._annotate(raw, dataset=dataset, arguments=arguments, schema=schema)
        # ``query`` 是**本地**过滤条件，必须留在 params 之外：它一旦进入就会被上游
        # schema 校验，并被原样发给服务端（上游没有这个参数）。记在结果参数里供渲染
        # 读取，与 ``arguments`` 同一机制。
        if query:
            raw.params = {**(raw.params or {}), "query": query}
        return raw

    def _resolve_service(self, dataset: str) -> str:
        """把数据集名落到本工具受理的某个服务上；受理范围外即报错。"""
        from finharness.data.adapters.fuyao_adapter import _service_for_dataset

        service = _service_for_dataset(dataset)
        if service not in self._services:
            raise ValueError(
                f"数据集 {dataset!r} 属于 {service} 服务，不属于本工具"
                f"（{self.name} 受理 {'、'.join(self._services)}）"
            )
        return service

    @staticmethod
    def _annotate(
        raw: RawData,
        *,
        dataset: str,
        arguments: dict[str, Any],
        schema: dict[str, Any] | None,
    ) -> RawData:
        """把实际生效的请求记进结果参数，使引用与复核能据以重现。

        ``RawData.params`` 会进入引用说明与缓存键，因此让它携带真实请求（而不是
        模型的原始入参）才能保证"这份数据是怎么来的"可复现——便捷参数被翻译过，
        记录翻译后的形态才有意义。

        ``inherited`` 是上游替我们填入的 schema 默认值。它必须单独记下来：省略
        ``thscodes`` 查询快照得到的是两只**默认**个股，而不是全市场，而请求里没有
        任何痕迹能表明这一点。不披露它，模型就可能把一份隐含筛选过的数据读成全局。
        """
        declared = set(fuyao_dataset_params(schema))
        requested = {k: v for k, v in arguments.items() if k in declared}
        inherited = fuyao_inherited_defaults(schema, arguments)
        raw.params = {
            **(raw.params or {}),
            "dataset": dataset,
            "arguments": requested,
            "inherited": inherited,
        }
        return raw

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """数据集结果的渲染：表头说明"查了什么"，随后是有界表格。

        表头不是装饰：长尾数据集的列名是上游的字段名，加上这一行，读者才知道那些
        字段出自哪个数据集、按什么条件筛的——包括**上游替我们填的**那些条件。
        """
        if raw.df is None or not len(raw.df):
            return (
                f"（数据集 {raw.endpoint} 未返回数据；"
                "可能的原因为该标的/该日期无记录，或筛选条件过窄）",
                [raw],
            )
        params = raw.params or {}
        arguments = params.get("arguments") or {}
        inherited = params.get("inherited") or {}
        query = str(params.get("query") or "").strip()
        filters = "、".join(f"{key}={value}" for key, value in arguments.items())
        header = f"数据集 {raw.endpoint}"
        if filters:
            header += f"（条件：{filters}）"
        if inherited:
            # 标出"由上游默认决定"而不是"你说了算"：这两者对结论的含义完全不同。
            defaults = "、".join(f"{key}={value}" for key, value in inherited.items())
            header += f"（未指定、由上游默认决定：{defaults}）"

        frame = self._newest_first(raw.df)
        matched: pd.DataFrame | None = None
        if query:
            matched = _filter_rows(frame, query)
            if not len(matched):
                # 过滤掉全部行时**如实说明作用范围**：匹配是在已返回的行里做的，
                # 因此"没匹配到"不等于"上游没有该项"。把它说成后者会把一次本地
                # 过滤失手变成一条错的全局结论。
                return (
                    f"{header}"
                    f"\n\n在返回的 {len(frame)} 行中没有匹配「{query}」的行。"
                    "注意：过滤只在**已返回**的数据上做，因此这不代表上游没有该项；"
                    "可去掉 query 看全量，或用 detail=\"full\" 扩大展示范围。",
                    [raw],
                )
            frame = matched
        header += f"（query={query}：匹配 {len(frame)} 行）" if query else ""

        body = self.trim_dataframe(
            frame, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        return f"{header}\n\n{body}", [raw]

    @staticmethod
    def _newest_first(df: pd.DataFrame) -> pd.DataFrame:
        """按时间列倒序，使渲染裁剪保留**最新**的行。

        ``trim_dataframe`` 只取前 N 行。内部对多数数据源的约定是"最新在前"，但
        **长尾数据集不是**：交易日历按自然日正序返回、历史类端点也常为升序。此时
        取前 N 行会把最老的一段交给模型。

        实测教训：交易日历返回 241 天（2025-09-22 ~ 2026-09-18），渲染只展示最早
        20 天（到 2025-10-27），于是"2026 年 9 月有几个交易日"无法回答——数据明明
        在手上，模型却看不到。倒序后保留的正是尾部。

        只在识别出明确的时间列时才排序：其它数据集的行序（如榜单名次）本身就是
        语义的一部分，重排会破坏它。
        """
        if df is None or not len(df):
            return df
        for column in ("date", "report_period", "ex_date", "report_date"):
            if column in df.columns:
                ordered = df.sort_values(column, ascending=False).reset_index(drop=True)
                # 首尾相同的时刻（同日多条记录）排序不稳定，用原顺序兜住。
                return ordered if len(ordered) == len(df) else df
        return df


@tool(
    name="list_fuyao_datasets",
    description=(
        "列出同花顺某服务可用的数据集及其参数，用于发现可查的数据（特色数据、基金、"
        "期货、期权、概念指数、标的检索等）。数据集与参数以服务端实时目录为准。"
    ),
    capability=Capability.DATASET,
    # 发现是长尾能力的前置步骤，但多数问题用现有工具即可回答，故按需注入。
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    timeout=30,
    # 复核只核报告与会话自有数据；再向第三方拉一份目录既耗 token 也不增加核验力。
    review_eligible=False,
    # 目录查询本身也访问第三方服务。
    egress=True,
    output_schema_note="返回数据集名、用途与参数列表（markdown）。",
)
class ListFuyaoDatasetsTool(BaseTool):
    @param("service", desc="服务：a-share（A股+特色数据+估值）/a-share-index（同花顺指数）/meta（标的检索）/fund/futures/options")
    @param("query", desc="可选关键词，按数据集名或描述过滤（本地过滤，不发给上游）")
    async def _dispatch(self, *, service: str, query: str | None = None) -> RawData:
        if service not in FUYAO_SERVICES:
            raise ValueError(
                f"未知的同花顺服务 {service!r}；可用：{'、'.join(FUYAO_SERVICES)}"
            )
        catalog = await self.data.dataset_catalog(service)
        rows = [
            {"name": str(entry.get("name") or ""), "description": str(entry.get("description") or ""),
             "parameters": fuyao_param_summary(_schema_of(entry))}
            for entry in catalog
        ]
        if query:
            needle = query.strip().lower()
            rows = [
                row
                for row in rows
                if needle in row["name"].lower() or needle in row["description"].lower()
            ]
        return RawData(
            kind="text",
            text=(
                f"同花顺服务 {service}（{FUYAO_SERVICES[service]}）可用数据集"
                f"{'（过滤：' + query + '）' if query else ''}：共 {len(rows)} 个\n\n"
                + _render_catalog(catalog, rows, query)
            ),
            endpoint=f"fuyao:catalog:{service}",
            params={"service": service, "query": query},
        )


def _render_catalog(
    catalog: list[dict[str, Any]], rows: list[dict[str, str]], query: str | None
) -> str:
    """渲染数据集清单；过滤后无结果时给出可用的服务列表。"""
    if not rows:
        if query:
            return (
                f"没有匹配「{query}」的数据集。可用的服务："
                + "、".join(f"{name}（{purpose}）" for name, purpose in FUYAO_SERVICES.items())
            )
        return "该服务当前没有可用数据集。"
    by_name = {str(entry.get("name") or ""): entry for entry in catalog}
    seen: set[str] = set()
    lines: list[str] = []
    for row in rows:
        name = row["name"]
        if name in seen:
            continue
        seen.add(name)
        lines.append(_describe_dataset(by_name.get(name, row)))
    return "\n".join(lines)


@tool(
    name="query_a_share_data",
    description=(
        "查询同花顺 A 股长尾数据：特色数据（涨停/跌停/炸板池、连板天梯、龙虎榜、"
        "热股榜、个股异动、集合竞价）、交易日历、估值快照、同花顺概念指数列表与成分股、"
        "标的检索与标的列表。先按 dataset 名指定要查的数据集。"
    ),
    capability=Capability.DATASET,
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    review_eligible=False,
    egress=True,
    output_schema_note="返回该数据集的表格（列名来自上游字段）。",
)
class QueryAShareDataTool(_FuyaoDatasetTool):
    # 概念指数与标的检索都是"按 A 股标的寻址"的，因此与 a-share 归在同一工具下：
    # 让模型为"查成分股"与"查涨停池"选两个工具名，只会增加选错的概率。
    _services = ("a-share", "a-share-index", "meta")

    @param("dataset", desc="数据集名（如 get_a_share_special_data_limit_up_pool、get_a_share_calendar_trading_days、get_a_share_index_constituents_ths_stock_list）；用 list_fuyao_datasets 查看全部")
    @param("params", desc="数据集参数，如 {'symbol': '600519'} 或 {'symbols': ['600519','000858']} 或 {'date': '2026-09-18'}；必填项见 list_fuyao_datasets 的参数列表")
    @param("query", desc="可选：只在**已返回的行**里按名称/代码做子串过滤（不区分大小写），用于在几百行的清单里定位某一行（如 query='白酒'）。"
                         "这是本地过滤，不改变发往上游的请求；没匹配到不代表上游没有该项")
    async def _dispatch(self, *, dataset: str, params: dict[str, Any] | None = None, query: str | None = None) -> RawData:
        return await super()._dispatch(dataset=dataset, params=params, query=query)


@tool(
    name="query_fund_data",
    description=(
        "查询同花顺公募基金数据：**ETF 二级市场行情（快照与历史日线）**、净值与业绩、"
        "持仓与行业配置、分红、持有人结构、基金经理、F10 诊断、财务指标、回测结果与额度。"
        "ETF/LOF 的行情走这里（A 股行情工具只覆盖个股）。先按 dataset 名指定数据集。"
    ),
    capability=Capability.DATASET,
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    review_eligible=False,
    egress=True,
    output_schema_note="返回该数据集的表格（列名来自上游字段）。",
)
class QueryFundDataTool(_FuyaoDatasetTool):
    _services = ("fund",)

    @param("dataset", desc="数据集名（ETF 行情用 get_fund_market_snapshot / get_fund_market_historical；另有 get_fund_portfolio_holdings、get_fund_performance_nav、get_fund_holders_top 等）；用 list_fuyao_datasets(service='fund') 查看全部")
    @param("params", desc="数据集参数，如 {'symbol': '510300'} 或 {'symbols': [...]}；必填项见 list_fuyao_datasets")
    @param("query", desc="可选：只在**已返回的行**里按名称/代码做子串过滤（不区分大小写），用于在大清单里定位某一行。本地过滤，不改变上游请求")
    async def _dispatch(self, *, dataset: str, params: dict[str, Any] | None = None, query: str | None = None) -> RawData:
        return await super()._dispatch(dataset=dataset, params=params, query=query)


@tool(
    name="query_futures_data",
    description=(
        "查询同花顺期货数据：品种与合约详情、持仓排名（品种/公司/合约，含历史）、"
        "仓单、基差（主力连续与历史）、交易时间轴与行情。先按 dataset 名指定数据集。"
    ),
    capability=Capability.DATASET,
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    review_eligible=False,
    egress=True,
    output_schema_note="返回该数据集的表格（列名来自上游字段）。",
)
class QueryFuturesDataTool(_FuyaoDatasetTool):
    _services = ("futures",)

    @param("dataset", desc="数据集名（如 get_futures_varieties_list、get_futures_positions_variety_daily、get_futures_basis_historical）；用 list_fuyao_datasets(service='futures') 查看全部")
    @param("params", desc="数据集参数；必填项见 list_fuyao_datasets")
    @param("query", desc="可选：只在**已返回的行**里按名称/代码做子串过滤（不区分大小写），用于在大清单里定位某一行。本地过滤，不改变上游请求")
    async def _dispatch(self, *, dataset: str, params: dict[str, Any] | None = None, query: str | None = None) -> RawData:
        return await super()._dispatch(dataset=dataset, params=params, query=query)


@tool(
    name="query_options_data",
    description=(
        "查询同花顺期权数据：品种列表、合约详情、分时与日线行情。先按 dataset 名指定数据集。"
    ),
    capability=Capability.DATASET,
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    timeout=60,
    data_tool=True,
    review_eligible=False,
    egress=True,
    output_schema_note="返回该数据集的表格（列名来自上游字段）。",
)
class QueryOptionsDataTool(_FuyaoDatasetTool):
    _services = ("options",)

    @param("dataset", desc="数据集名（如 get_options_varieties_list、get_options_contracts_detail）；用 list_fuyao_datasets(service='options') 查看全部")
    @param("params", desc="数据集参数；必填项见 list_fuyao_datasets")
    @param("query", desc="可选：只在**已返回的行**里按名称/代码做子串过滤（不区分大小写），用于在大清单里定位某一行。本地过滤，不改变上游请求")
    async def _dispatch(self, *, dataset: str, params: dict[str, Any] | None = None, query: str | None = None) -> RawData:
        return await super()._dispatch(dataset=dataset, params=params, query=query)


__all__ = [
    "ListFuyaoDatasetsTool",
    "QueryAShareDataTool",
    "QueryFundDataTool",
    "QueryFuturesDataTool",
    "QueryOptionsDataTool",
]
