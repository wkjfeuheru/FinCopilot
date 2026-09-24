"""同花顺数据集派发工具：dataset 校验、参数校验与渲染。

这些工具是本集成暴露给模型的那一面，因此测试重点在"模型给错东西时会发生什么"：
未知 dataset、未知参数、缺失必填、枚举越界、传错服务——每一种都必须显式失败
（``ok=False``），而不是静默降级成一次全量查询。
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from finharness.config.settings import DataSettings, Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.mcp_client import McpHttpClient
from finharness.tools.base import BaseTool
from finharness.tools.fin.dataset import (
    ListFuyaoDatasetsTool,
    QueryAShareDataTool,
    QueryFundDataTool,
    QueryFuturesDataTool,
    QueryOptionsDataTool,
)

LIMIT_UP = "get_a_share_special_data_limit_up_pool"
CALENDAR = "get_a_share_calendar_trading_days"
CONSTITUENTS = "get_a_share_index_constituents_ths_stock_list"
SEARCH = "get_meta_tickers_search"
FUND_HOLDINGS = "get_fund_portfolio_holdings"
FUTURES_BASIS = "get_futures_basis_historical"
SNAPSHOT = "get_a_share_prices_snapshot"
DRAGON_TIGER = "get_a_share_special_data_dragon_tiger_list"


def envelope(data: object) -> dict:
    return {"code": 0, "message": "ok", "request_id": "r", "data": data}


# 每个数据集的 schema 与应答载荷。schema 是参数的唯一事实来源，因此测试服务端也
# 按同一机制提供它——而不是让本地代码假定某种形状。
DATASETS: dict[str, dict] = {
    LIMIT_UP: {
        "schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": [],
        },
        "payload": envelope({"item": [{"thscode": "600519.SH", "name": "贵州茅台", "reason": "白酒"}]}),
    },
    CALENDAR: {
        "schema": {"type": "object", "properties": {}, "required": []},
        "payload": envelope({"item": [{"date": "20260918", "date_ms": 1758124800000}]}),
    },
    CONSTITUENTS: {
        "schema": {
            "type": "object",
            "properties": {"thscode": {"type": "string"}},
            "required": ["thscode"],
        },
        "payload": envelope({"item": [{"thscode": "600519.SH", "ticker": "600519", "name": "贵州茅台"}]}),
    },
    SEARCH: {
        "schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "asset_type": {"type": "string", "enum": ["a-share", "fund-etf"]}},
            "required": ["q"],
        },
        "payload": envelope({"item": [{"thscode": "600519.SH", "name": "贵州茅台"}]}),
    },
    FUND_HOLDINGS: {
        "schema": {
            "type": "object",
            "properties": {"thscode": {"type": "string"}, "report_date": {"type": "string"}},
            "required": ["thscode"],
        },
        "payload": envelope({"item": [{"ticker": "600519", "name": "贵州茅台", "ratio": 9.8}]}),
    },
    FUTURES_BASIS: {
        "schema": {
            "type": "object",
            "properties": {"variety": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["variety"],
        },
        "payload": envelope({"item": [{"variety": "IF", "basis": 12.3}]}),
    },
    # 下面两个端点的 schema 带**实质性**默认值——这正是线上 a-share 服务的真实形态
    # （已对照真实 tools/list 确认）。
    SNAPSHOT: {
        "schema": {
            "type": "object",
            "properties": {
                "thscodes": {"type": "string", "default": "600519.SH,000001.SZ"},
                "limit": {"type": "integer", "default": 100},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        },
        "payload": envelope({"item": [{"ticker": "600519", "last_price": 1500.0}]}),
    },
    DRAGON_TIGER: {
        "schema": {
            "type": "object",
            "properties": {
                "board_type": {
                    "type": "string",
                    "enum": ["all", "org", "hot_money"],
                    "default": "all",
                },
                "date": {"type": "string"},
            },
            "required": [],
        },
        "payload": envelope({"item": [{"ticker": "600519", "name": "贵州茅台"}]}),
    },
}


class Server:
    """一个按数据集名应答的模拟 MCP 服务端，记录收到的每次调用。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read().decode())
            method = body["method"]
            if method == "initialize":
                return httpx.Response(
                    200,
                    json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                    headers={"Mcp-Session-Id": "s"},
                )
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                tools = [
                    {"name": name, "description": "测试数据集", "inputSchema": spec["schema"]}
                    for name, spec in DATASETS.items()
                ]
                return httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": tools}}
                )
            params = body["params"]
            self.calls.append((params["name"], params.get("arguments") or {}))
            spec = DATASETS.get(params["name"])
            payload = spec["payload"] if spec else {"code": 1001, "message": "未知数据集"}
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
                    },
                },
            )

        return httpx.MockTransport(handler)


def make_tools(tmp_path, tool_cls: type[BaseTool]) -> tuple[BaseTool, Server]:
    """构造一个工具实例，其数据门面接在一个模拟 MCP 服务端上。"""
    server = Server()
    settings = Settings(data=DataSettings(cache_dir=tmp_path / "cache"))
    client = McpHttpClient(
        url="https://fuyao.aicubes.cn/mcp/a-share",
        api_key="k",
        client=httpx.Client(transport=server.transport()),
    )
    from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter

    adapter = FuyaoMcpAdapter(api_key="k", throttle_seconds=0.0, client=client)
    access = DataAccess([adapter], settings=settings)
    return tool_cls(access), server


def run(tool: BaseTool, **kwargs):
    return asyncio.run(tool.run(**kwargs))


# -- 目录 ------------------------------------------------------------------


def test_the_catalog_lists_datasets_with_their_parameters(tmp_path):
    tool, _ = make_tools(tmp_path, ListFuyaoDatasetsTool)

    result = run(tool, service="a-share")

    assert result.ok
    assert LIMIT_UP in result.content
    # 参数名与必填标记都要出现：模型据此组织参数。
    assert "date" in result.content
    assert "thscode" in result.content


def test_the_catalog_renders_enums_because_wrong_enums_are_the_common_failure(tmp_path):
    tool, _ = make_tools(tmp_path, ListFuyaoDatasetsTool)

    result = run(tool, service="a-share")

    assert "a-share|fund-etf" in result.content


def test_the_catalog_can_be_filtered_by_a_keyword(tmp_path):
    tool, _ = make_tools(tmp_path, ListFuyaoDatasetsTool)

    result = run(tool, service="a-share", query="trading_days")

    assert CALENDAR in result.content
    assert LIMIT_UP not in result.content


def test_a_filter_that_matches_nothing_says_so_and_lists_the_services(tmp_path):
    tool, _ = make_tools(tmp_path, ListFuyaoDatasetsTool)

    result = run(tool, service="a-share", query="不存在的关键词")

    assert result.ok
    assert "没有匹配" in result.content
    assert "fund" in result.content  # 可用服务清单


def test_an_unknown_service_is_refused_with_the_valid_ones(tmp_path):
    tool, _ = make_tools(tmp_path, ListFuyaoDatasetsTool)

    result = run(tool, service="crypto")

    assert result.ok is False
    assert "未知的同花顺服务" in (result.error or "")


# -- dataset 校验 ---------------------------------------------------------


def test_an_unknown_dataset_is_refused_and_the_available_ones_are_listed(tmp_path):
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset="get_a_share_nonexistent")

    assert result.ok is False
    assert "不存在" in (result.error or "")
    assert LIMIT_UP in (result.error or "")


def test_a_dataset_from_another_service_is_refused_by_this_tool(tmp_path):
    """基金数据集交给 query_a_share_data 会静默失败；显式拒绝并指出正确的工具。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=FUND_HOLDINGS, params={"symbol": "510300"})

    assert result.ok is False
    assert "fund" in (result.error or "")
    # 数据集本身是合法的，只是问错了工具——所以不会发出任何上游请求。
    assert "不属于本工具" in (result.error or "")


def test_each_dispatcher_accepts_its_own_service(tmp_path):
    tool, _ = make_tools(tmp_path, QueryFundDataTool)

    result = run(tool, dataset=FUND_HOLDINGS, params={"symbol": "510300"})

    assert result.ok


def test_futures_dispatcher_requires_its_required_parameter(tmp_path):
    tool, _ = make_tools(tmp_path, QueryFuturesDataTool)

    result = run(tool, dataset=FUTURES_BASIS, params={})

    assert result.ok is False
    assert "缺少必填参数" in (result.error or "")
    assert "variety" in (result.error or "")


def test_options_dispatcher_reports_an_unknown_dataset(tmp_path):
    tool, _ = make_tools(tmp_path, QueryOptionsDataTool)

    result = run(tool, dataset="get_options_nope")

    assert result.ok is False


# -- 参数校验 -------------------------------------------------------------


def test_an_unknown_parameter_is_refused_rather_than_dropped(tmp_path):
    """静默丢弃会让模型以为自己筛过了，而实际拿到的是全量——这种错误在结果里看不出来。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP, params={"board": "org"})

    assert result.ok is False
    assert "不支持参数" in (result.error or "")
    assert "board" in (result.error or "")
    assert server.calls == []  # 未发请求


def test_an_enum_violation_is_refused(tmp_path):
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=SEARCH, params={"q": "茅台", "asset_type": "crypto"})

    assert result.ok is False
    assert "不合法" in (result.error or "")
    assert server.calls == []


def test_a_valid_enum_passes(tmp_path):
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=SEARCH, params={"q": "茅台", "asset_type": "fund-etf"})

    assert result.ok


def test_the_symbol_convenience_parameter_becomes_a_thscode(tmp_path):
    """模型知道六位代码，不知道 thscode；翻译发生在本地，上游只看到 thscode。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    run(tool, dataset=CONSTITUENTS, params={"symbol": "600519"})

    assert server.calls[0][1]["thscode"] == "600519.SH"


def test_a_symbol_list_becomes_a_comma_joined_thscodes(tmp_path):
    tool, server = make_tools(tmp_path, QueryFundDataTool)

    run(tool, dataset=FUND_HOLDINGS, params={"symbols": ["510300", "159915"]})

    assert server.calls[0][1]["thscodes"] == "510300.SH,159915.SZ"


def test_a_date_parameter_is_passed_through_because_upstream_takes_yyyy_mm_dd(tmp_path):
    """只有 start/end 是毫秒时间戳；date 类参数上游要的是 YYYY-MM-DD 字符串。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    run(tool, dataset=LIMIT_UP, params={"date": "2026-09-18", "limit": 5})

    sent = server.calls[0][1]
    assert sent["date"] == "2026-09-18"
    assert sent["limit"] == 5


def test_an_illegal_symbol_is_refused_before_any_request(tmp_path):
    """带一个错代码发出去只会得到一份空数据；本地拒绝才说得清原因。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=CONSTITUENTS, params={"symbol": "abc"})

    assert result.ok is False
    assert "6 位数字" in (result.error or "")
    assert server.calls == []


def test_an_unparseable_date_is_refused(tmp_path):
    """start/end 是毫秒时间戳参数；给一个解析不出日期的字符串必须显式失败。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP, params={"start": "前天"})

    assert result.ok is False
    assert "无法解析时间" in (result.error or "")
    assert server.calls == []


def test_a_millisecond_window_is_translated_from_dates(tmp_path):
    """文档口径是毫秒，但模型更可能给出日期；两种写法都要接受。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    run(tool, dataset=LIMIT_UP, params={"start": "2026-09-01", "end": "2026-09-18"})

    sent = server.calls[0][1]
    assert sent["start"] == 1788192000000
    assert sent["end"] == 1789660800000


def test_params_may_be_omitted_entirely(tmp_path):
    """没有必填参数的数据集可以不带 params 调用。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=CALENDAR)

    assert result.ok


# -- 渲染 ------------------------------------------------------------------


def test_the_result_header_names_the_dataset_and_the_filters(tmp_path):
    """列名是上游字段名；带上"查了什么"这一行，读者才知道它出自哪里。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP, params={"date": "2026-09-18", "limit": 5})

    assert f"数据集 fuyao:{LIMIT_UP}" in result.content
    assert "date=2026-09-18" in result.content
    assert "limit=5" in result.content
    assert "贵州茅台" in result.content


def test_an_empty_result_explains_the_possible_reasons(tmp_path):
    """空结果与"上游没有这个数据集"是两件事，必须能分辨。"""
    DATASETS[LIMIT_UP]["payload"] = envelope({"item": []})
    try:
        tool, _ = make_tools(tmp_path, QueryAShareDataTool)
        result = run(tool, dataset=LIMIT_UP, params={"limit": 5})
    finally:
        DATASETS[LIMIT_UP]["payload"] = envelope(
            {"item": [{"thscode": "600519.SH", "name": "贵州茅台", "reason": "白酒"}]}
        )

    assert result.ok
    assert "未返回数据" in result.content
    assert "无记录" in result.content


def test_rendered_results_carry_a_citation_source(tmp_path):
    """数据工具必须注册引用，否则报告里的数字无法溯源。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP, params={"limit": 5})

    assert result.sources
    assert result.sources[0].endpoint == f"fuyao:{LIMIT_UP}"


def test_the_tools_declare_egress_so_the_gate_confirms_them():
    """把用户查询发往第三方，因此首次调用须经确认（docs 03.7.1）。"""
    for tool_cls in (
        ListFuyaoDatasetsTool,
        QueryAShareDataTool,
        QueryFundDataTool,
        QueryFuturesDataTool,
        QueryOptionsDataTool,
    ):
        assert tool_cls.egress is True, tool_cls.name
        # 复核只核报告与会话自有数据，不向第三方再拉一份。
        assert tool_cls.review_eligible is False, tool_cls.name
        assert tool_cls.capability.value == "数据集", tool_cls.name


# -- 上游默认值披露 -------------------------------------------------------
# 上游 schema 为若干参数声明了实质性默认值。调用方省略它们时，拿到的不是"全部"，
# 而是上游选定的那一小撮——请求里不留任何痕迹。以下把它变成可见信息。


def test_an_inherited_upstream_default_is_disclosed_in_the_header(tmp_path):
    """快照的 thscodes 默认是两只具体个股；不披露就会被读成全市场。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=SNAPSHOT, params={"limit": 5})

    assert "未指定、由上游默认决定" in result.content
    assert "thscodes=600519.SH,000001.SZ" in result.content


def test_a_specified_filter_is_not_reported_as_an_inherited_default(tmp_path):
    """自己指定了筛选条件时，它不该被说成"由上游默认决定"——那两件事的含义相反。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=SNAPSHOT, params={"symbols": ["600519"], "limit": 5})

    assert "thscodes=600519.SH" in result.content  # 作为指定条件出现
    inherited_clause = result.content.split("未指定、由上游默认决定")[-1]
    assert "thscodes" not in inherited_clause.split("）")[0]


def test_the_inherited_default_is_recorded_in_the_result_params(tmp_path):
    """引用与复核要能看出这份数据是按什么条件取的（含上游代填的条件）。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=SNAPSHOT, params={"limit": 5})

    raw = result.sources[0]
    assert raw.params["inherited"]["thscodes"] == "600519.SH,000001.SZ"
    assert raw.params["arguments"]["limit"] == 5


def test_a_paging_default_is_also_disclosed_because_it_bounds_the_result(tmp_path):
    """分页默认同样影响"看到了多少"，因此也属披露范围。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=SNAPSHOT, params={"symbols": ["600519"]})

    assert "limit=100" in result.content


def test_an_enum_default_is_disclosed(tmp_path):
    """``board_type=all`` 之类的默认改变了结果的构成（全部/机构/游资）。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=DRAGON_TIGER)

    assert "board_type=all" in result.content


# -- 可发现性：ETF 行情必须在基金工具的说明里 ----------------------------
# 实测教训：模型曾断言"不含 ETF 的二级市场价格"——那是错的（fund 服务提供
# get_fund_market_snapshot），只是工具描述没提行情，模型便无从知道该去哪里查。
# 这里把"描述必须告知该能力"钉住，因为工具描述是模型唯一能看到的入口。


def test_the_fund_tool_description_advertises_etf_market_data():
    """A 股行情工具只覆盖个股，因此 ETF 价格的唯一入口必须在基金工具的描述里。"""
    from finharness.shared.declaration import declared

    description = QueryFundDataTool.description

    assert "ETF" in description
    assert "行情" in description
    # 数据集名要能被搜到，否则模型只能靠猜。
    params = {spec.name: spec.description for spec in declared("query_fund_data").params}
    assert "get_fund_market_snapshot" in params["dataset"]


def test_the_system_prompt_points_etf_prices_at_the_fund_tool():
    """提示词的"能做什么"清单同样要给出这条分流，否则模型到不了那个工具。"""
    from finharness.engine.prompt import system_prompt

    text = system_prompt()

    assert "ETF" in text
    assert "get_fund_market_snapshot" in text


# -- 时间序列数据集必须保留**最新**的行 --------------------------------
# 实测教训：交易日历返回 241 天（2025-09-22 ~ 2026-09-18，正序），而 trim_dataframe
# 只取前 20 行，于是渲染出来的是 2025-10 之前的旧日期——"2026年9月有几个交易日"
# 因此无法回答，尽管数据已在手上。


def test_a_chronological_dataset_keeps_the_newest_rows(tmp_path):
    """日历类数据集按自然日正序返回；裁剪必须保留尾部而非头部。"""
    rows = [
        {"date_ms": 1758499200000 + i * 86400000}  # 逐日递增
        for i in range(60)
    ]
    DATASETS[CALENDAR]["payload"] = envelope({"item": rows})
    try:
        tool, _ = make_tools(tmp_path, QueryAShareDataTool)
        result = run(tool, dataset=CALENDAR)
    finally:
        DATASETS[CALENDAR]["payload"] = envelope(
            {"item": [{"date": "20260918", "date_ms": 1758124800000}]}
        )

    assert result.ok
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", result.content)
    # 最新的一天必须在渲染结果里，且已是倒序（首行最新）。
    assert dates, result.content[:200]
    assert dates[0] == max(dates), f"首行应是最新日期，实际 {dates[0]} / 最大 {max(dates)}"


def test_a_ranking_dataset_keeps_its_original_order(tmp_path):
    """榜单的名次本身就是语义，不能因为"像时间列"就被重排。"""
    DATASETS[LIMIT_UP]["payload"] = envelope(
        {"item": [{"rank": 1, "name": "甲"}, {"rank": 2, "name": "乙"}, {"rank": 3, "name": "丙"}]}
    )
    try:
        tool, _ = make_tools(tmp_path, QueryAShareDataTool)
        result = run(tool, dataset=LIMIT_UP)
    finally:
        DATASETS[LIMIT_UP]["payload"] = envelope(
            {"item": [{"thscode": "600519.SH", "name": "贵州茅台", "reason": "白酒"}]}
        )

    assert result.content.index("甲") < result.content.index("乙") < result.content.index("丙")


# -- 行级过滤：在几百行清单里定位目标行 ---------------------------------
# 实测教训：问「白酒概念板块成分股」时，"白酒概念"确实在 390 个概念里，但渲染只看
# 前 20 行，模型看不到它，于是转去逐个翻成分股，耗尽轮次仍未答出。


def _big_catalog_payload() -> dict:
    rows = [{"thscode": f"885{i:03d}.TI", "name": f"概念{i:03d}"} for i in range(1, 391)]
    rows[286] = {"thscode": "885525.TI", "name": "白酒概念"}
    return envelope({"item": rows})


def test_query_filter_surfaces_a_row_beyond_the_render_window(tmp_path):
    """目标行在第 287 位、渲染窗口只有 20 行；过滤后必须能直接看到它。"""
    dataset = "get_a_share_index_catalog_ths_index_list"
    DATASETS[dataset] = {
        "schema": {"type": "object", "properties": {"tag": {"type": "string"}}, "required": []},
        "payload": _big_catalog_payload(),
    }
    try:
        tool, _ = make_tools(tmp_path, QueryAShareDataTool)
        result = run(tool, dataset=dataset, query="白酒")
    finally:
        DATASETS.pop(dataset, None)

    assert result.ok
    assert "白酒概念" in result.content
    assert "885525.TI" in result.content
    assert "匹配 1 行" in result.content


def test_query_does_not_reach_the_upstream_request(tmp_path):
    """``query`` 是本地显示过滤：它绝不能进入上游参数（上游没有这个参数）。"""
    tool, server = make_tools(tmp_path, QueryAShareDataTool)

    run(tool, dataset=LIMIT_UP, params={"limit": 5}, query="茅台")

    assert server.calls, "应发出一次上游调用"
    sent = server.calls[0][1]
    assert "query" not in sent
    assert sent == {"limit": 5}


def test_a_query_that_matches_nothing_says_so_without_claiming_absence(tmp_path):
    """本地过滤没匹配到 ≠ 上游没有该项；措辞必须守住这条区别。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP, query="不存在的名字")

    assert result.ok
    assert "没有匹配" in result.content
    # 不得表述成"上游没有"。
    assert "不代表上游没有" in result.content


def test_query_matches_across_symbol_columns_too(tmp_path):
    """按代码过滤也应生效，不只是名称。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP, query="600519")

    assert "贵州茅台" in result.content


def test_without_query_the_result_is_unchanged(tmp_path):
    """不带 query 时行为不变（不引入意外的过滤）。"""
    tool, _ = make_tools(tmp_path, QueryAShareDataTool)

    result = run(tool, dataset=LIMIT_UP)

    assert result.ok
    assert "贵州茅台" in result.content
    assert "query=" not in result.content
