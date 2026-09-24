"""同花顺适配器：载荷归一化、语义方法契约与故障分类。

全程使用 ``httpx.MockTransport``，因此这里不触碰网络，也不需要 API Key。重点在于
两件容易出错的事：结果是否落到本项目**既有的**表格契约上（行情列名、财报宽表、
指标表），以及上游的业务错误码是否被翻译成正确的可重试性。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter, _service_for_dataset
from finharness.data.adapters.mcp_client import McpHttpClient
from finharness.data.mapping import MARKET_COLUMNS

_SHANGHAI = timezone(timedelta(hours=8))


def ms(year: int, month: int, day: int) -> int:
    """Asia/Shanghai 午夜的毫秒时间戳——同花顺所有日期字段的口径。"""
    return int(datetime(year, month, day, tzinfo=_SHANGHAI).timestamp() * 1000)


def envelope(data: object, code: int = 0, message: str = "ok") -> dict:
    return {"code": code, "message": message, "request_id": "req-1", "data": data}


def make_adapter(handlers: dict[str, object], *, api_key: str | None = "k") -> FuyaoMcpAdapter:
    """构造一个适配器，其服务端按数据集名返回 ``handlers`` 中登记的载荷。"""
    seen: list[tuple[str, dict]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        if body["method"] == "initialize":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {}},
                headers={"Mcp-Session-Id": "s"},
            )
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        if body["method"] == "tools/list":
            tools = [{"name": name, "inputSchema": {"type": "object"}} for name in handlers]
            return httpx.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": tools}}
            )
        params = body["params"]
        name = params["name"]
        seen.append((name, params.get("arguments") or {}))
        handler = handlers.get(name)
        if handler is None:
            payload = {"code": 1001, "message": f"未知数据集 {name}"}
        elif callable(handler):
            payload = handler(params.get("arguments") or {})
        else:
            payload = handler
        result = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    adapter = FuyaoMcpAdapter(
        api_key=api_key,
        throttle_seconds=0.0,
        client=McpHttpClient(
            url="https://fuyao.aicubes.cn/mcp/a-share",
            api_key=api_key,
            client=httpx.Client(transport=httpx.MockTransport(transport)),
        ),
    )
    adapter.seen = seen  # type: ignore[attr-defined]
    return adapter


QUOTE_DATASET = "get_a_share_prices_snapshot"
KLINE_DATASET = "get_a_share_prices_historical"
INCOME_DATASET = "get_a_share_financials_income_statements"
INDICATORS_DATASET = "get_a_share_financials_indicators"
CONSTITUENTS_DATASET = "get_a_share_index_constituents_ths_stock_list"


def quote_payload() -> dict:
    return envelope(
        {
            "timestamp": ms(2026, 9, 18),
            "total": 1,
            "item": [
                {
                    "thscode": "600519.SH",
                    "ticker": "600519",
                    "last_price": 1500.0,
                    "price_change": 10.0,
                    "price_change_ratio_pct": 0.67,
                    "open_price": 1495.0,
                    "high_price": 1510.0,
                    "low_price": 1490.0,
                    "prev_price": 1490.0,
                    "volume": 123456,
                    "turnover": 1.85e9,
                }
            ],
        }
    )


# -- 行情 ------------------------------------------------------------------


def test_quote_lands_on_the_internal_market_contract():
    """下游渲染与图表读的是 date/open/high/low/close/volume/amount，不是同花顺的列名。"""
    adapter = make_adapter({QUOTE_DATASET: quote_payload()})

    result = adapter.fetch_quote("600519")

    assert result.interface == QUOTE_DATASET
    assert result.df.iloc[0]["symbol"] == "600519"
    assert result.df.iloc[0]["close"] == 1500.0
    # 内部契约里没有的字段保留原名，而不是被丢弃。
    for column in ("open", "high", "low", "close", "volume", "amount"):
        assert column in result.df.columns
    assert set(MARKET_COLUMNS) <= set(result.df.columns)


def test_quote_asks_for_the_thscode_not_the_bare_code():
    adapter = make_adapter({QUOTE_DATASET: quote_payload()})

    adapter.fetch_quote("600519")

    name, arguments = adapter.seen[0]
    assert name == QUOTE_DATASET
    assert arguments["thscodes"] == "600519.SH"


def test_quote_accepts_a_single_code_as_a_plain_string():
    """文档写的是 `thscodes`（复数），但实测接受单值字符串；数组会因逗号拼接失败。"""
    seen_args: list[dict] = []

    def capture(arguments: dict) -> dict:
        seen_args.append(arguments)
        return quote_payload()

    make_adapter({QUOTE_DATASET: capture}).fetch_quote("000858")

    assert isinstance(seen_args[0]["thscodes"], str)


def test_an_empty_quote_is_an_adapter_error():
    """空的 item 容器必须如实变成空表，而不是一行 ``item="[]"`` 的假记录。"""
    adapter = make_adapter({QUOTE_DATASET: envelope({"timestamp": ms(2026, 9, 18), "total": 0, "item": []})})

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_quote("600519")

    assert "未返回" in str(exc.value)


def test_quote_carries_the_snapshot_timestamp_as_its_date():
    """快照行本身不带日期，只有信封上的 timestamp；缺了它这一行就无时点可言。"""
    adapter = make_adapter({QUOTE_DATASET: quote_payload()})

    frame = adapter.fetch_quote("600519").df

    assert frame.iloc[0]["date"].strftime("%Y-%m-%d") == "2026-09-18"


def test_an_empty_list_payload_yields_an_empty_frame_not_a_fake_row():
    """数据集派发器也要能分辨"查到了但没有数据"与"查到一条数据"。"""
    adapter = make_adapter({"get_a_share_special_data_limit_up_pool": envelope({"item": []})})

    frame = adapter.fetch_dataset("a-share", "get_a_share_special_data_limit_up_pool", {}).df

    assert len(frame) == 0


# -- K 线 ------------------------------------------------------------------


def kline_payload() -> dict:
    return envelope(
        {
            "timestamp": ms(2026, 9, 18),
            "item": [
                {"date_ms": ms(2026, 9, 18), "open_price": 1, "high_price": 2, "low_price": 0.5, "close_price": 1.5, "volume": 10, "turnover": 20},
                {"date_ms": ms(2026, 9, 17), "open_price": 2, "high_price": 3, "low_price": 1.0, "close_price": 2.5, "volume": 11, "turnover": 21},
            ],
        }
    )


def test_kline_is_sorted_latest_first_and_dated():
    adapter = make_adapter({KLINE_DATASET: kline_payload()})

    frame = adapter.fetch_kline("600519", "day", None, 1).df

    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2026-09-18", "2026-09-17"]
    assert frame.iloc[0]["close"] == 1.5


@pytest.mark.parametrize(("internal", "expected"), [("qfq", "forward"), ("hfq", "backward"), (None, "none")])
def test_kline_maps_the_adjust_vocabulary(internal, expected):
    """内部用 qfq/hfq，同花顺用 forward/backward——两套词表在这里对接。"""
    seen: list[dict] = []

    def capture(arguments: dict) -> dict:
        seen.append(arguments)
        return kline_payload()

    make_adapter({KLINE_DATASET: capture}).fetch_kline("600519", "day", internal, 1)

    assert seen[0]["adjust"] == expected
    assert seen[0]["interval"] == "1d"


def test_weekly_kline_reports_not_supported_so_the_chain_falls_back():
    """这不是缺陷而是回退理由：编排器读作"该源不支持"，转给 akshare。"""
    adapter = make_adapter({KLINE_DATASET: kline_payload()})

    with pytest.raises(NotImplementedError):
        adapter.fetch_kline("600519", "week", None, 1)

    assert adapter.seen == []  # 未发出任何请求


def test_a_window_beyond_the_upstream_limit_is_refused_locally():
    """上游对超过十年的窗口返回 code=1003；本地先拒绝，以免把被截短的序列当近 N 年。"""
    adapter = make_adapter({KLINE_DATASET: kline_payload()})

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_kline("600519", "day", None, 20)

    assert "上限" in str(exc.value)
    assert adapter.seen == []


def test_kline_window_starts_the_requested_number_of_years_back():
    seen: list[dict] = []

    def capture(arguments: dict) -> dict:
        seen.append(arguments)
        return kline_payload()

    make_adapter({KLINE_DATASET: capture}).fetch_kline("600519", "day", None, 2)

    span_days = (seen[0]["end"] - seen[0]["start"]) / 1000 / 86400
    assert 730 <= span_days <= 800


# -- 财报 ------------------------------------------------------------------


def income_payload() -> dict:
    common = {"thscode": "600519.SH", "ticker": "600519", "period": "annual", "currency": "CNY"}
    return envelope(
        {
            "timestamp": ms(2024, 4, 30),
            "item": [
                {**common, "fiscal_year": 2023, "fiscal_period": "FY", "report_date_ms": ms(2024, 4, 30), "period_end_ms": ms(2023, 12, 31),
                 "operating_income": 1.47e11, "net_profit": 7.4e10, "parent_holder_net_profit": 7.4e10, "basic_eps": 59.49},
                {**common, "fiscal_year": 2022, "fiscal_period": "FY", "report_date_ms": ms(2023, 4, 30), "period_end_ms": ms(2022, 12, 31),
                 "operating_income": 1.27e11, "net_profit": 6.3e10, "parent_holder_net_profit": 6.3e10, "basic_eps": 50.0},
            ],
        }
    )


def test_financials_pivot_into_the_wide_statement_shape():
    """指标为行、报告期为 YYYYMMDD 列——与 akshare 的摘要表同形，下游才无需特判。"""
    adapter = make_adapter({INCOME_DATASET: income_payload()})

    frame = adapter.fetch_financials("600519", "利润", 3).df

    assert frame.columns[0] == "指标"
    assert "20231231" in frame.columns
    assert "20221231" in frame.columns
    labels = list(frame["指标"])
    assert "营业收入" in labels
    assert "归母净利润" in labels


def test_financials_order_period_columns_newest_first():
    """裁剪逻辑假定第一列是最新一期，因此顺序不能交给上游决定。"""
    adapter = make_adapter({INCOME_DATASET: income_payload()})

    frame = adapter.fetch_financials("600519", "利润", 3).df

    assert list(frame.columns)[1:] == ["20231231", "20221231"]


def test_financials_do_not_leave_raw_english_field_names_as_rows():
    adapter = make_adapter({INCOME_DATASET: income_payload()})

    labels = set(adapter.fetch_financials("600519", "利润", 3).df["指标"])

    assert "operating_income" not in labels


def test_financials_ask_for_annual_periods():
    seen: list[dict] = []

    def capture(arguments: dict) -> dict:
        seen.append(arguments)
        return income_payload()

    make_adapter({INCOME_DATASET: capture}).fetch_financials("600519", "利润", 5)

    assert seen[0]["period"] == "annual"
    assert seen[0]["thscode"] == "600519.SH"


def test_an_unsupported_statement_is_reported_as_not_supported():
    adapter = make_adapter({INCOME_DATASET: income_payload()})

    with pytest.raises(NotImplementedError):
        adapter.fetch_financials("600519", "股东权益变动", 3)


# -- 财务指标 --------------------------------------------------------------


def indicators_payload(period: str, roe: float | None, margin: float | None) -> dict:
    return envelope(
        {
            "thscode": "600519.SH",
            "report": period,
            "abilities": [
                {"ability": "growth", "indicators": [{"index_id": "net_profit_yoy_growth_ratio", "value": "12.5"}]},
                {"ability": "profitability", "indicators": [
                    {"index_id": "index_weighted_avg_roe", "value": None if roe is None else str(roe)},
                    {"index_id": "sale_gross_margin", "value": None if margin is None else str(margin)},
                ]},
            ],
        }
    )


def test_indicators_fold_abilities_into_labelled_columns():
    """两层结构直接展平会得到读不出含义的点号列名；折成指标名为列才可用。"""
    adapter = make_adapter({INDICATORS_DATASET: lambda a: indicators_payload(a["report"], 30.5, 91.2)})

    frame = adapter.fetch_indicators("600519", 1, None).df

    assert "加权净资产收益率" in frame.columns
    assert "销售毛利率" in frame.columns
    assert "date" in frame.columns
    assert frame.iloc[0]["加权净资产收益率"] == 30.5


def test_indicators_cover_every_period_newest_first():
    adapter = make_adapter({INDICATORS_DATASET: lambda a: indicators_payload(a["report"], 30.5, 91.2)})

    frame = adapter.fetch_indicators("600519", 1, None).df

    assert len(frame) == 4  # 一年 = 4 个季度
    dates = list(frame["date"])
    assert dates == sorted(dates, reverse=True)


def test_indicators_keep_going_when_one_period_has_no_data():
    """未披露的期间本就该缺席，不该丢掉整条序列。"""
    def handler(arguments: dict) -> dict:
        if arguments["report"] == "2025-4":
            return {"code": 3002, "message": "数据未就绪"}
        return indicators_payload(arguments["report"], 30.5, 91.2)

    adapter = make_adapter({INDICATORS_DATASET: handler})

    frame = adapter.fetch_indicators("600519", 1, None).df

    assert len(frame) == 3


def test_indicators_report_a_failure_only_when_no_period_succeeded():
    adapter = make_adapter({INDICATORS_DATASET: {"code": 3002, "message": "数据未就绪"}})

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_indicators("600519", 1, None)

    assert "未返回" in str(exc.value)


def test_indicators_preserve_unpublished_values_as_empty_not_zero():
    """上游以 null 表示"该期未披露"；填 0 会把它变成一条看起来真实的记录。"""
    adapter = make_adapter({INDICATORS_DATASET: lambda a: indicators_payload(a["report"], None, 91.2)})

    frame = adapter.fetch_indicators("600519", 1, None).df

    assert frame["加权净资产收益率"].isna().all()


def test_indicator_field_filter_uses_the_chinese_aliases():
    """`fields=["ROE"]` 走中文子串匹配，因此指标名必须已翻译。"""
    adapter = make_adapter({INDICATORS_DATASET: lambda a: indicators_payload(a["report"], 30.5, 91.2)})

    frame = adapter.fetch_indicators("600519", 1, ["ROE"]).df

    assert "加权净资产收益率" in frame.columns
    assert "销售毛利率" not in frame.columns


# -- 指数成分股 ------------------------------------------------------------


def constituents_payload() -> dict:
    return envelope(
        {
            "timestamp": ms(2026, 9, 18),
            "item": [
                {"thscode": "600519.SH", "ticker": "600519", "name": "贵州茅台"},
                {"thscode": "000858.SZ", "ticker": "000858", "name": "五粮液"},
            ],
        }
    )


def test_index_constituents_use_the_index_thscode_not_a_guessed_suffix():
    """000300 按股票号段会猜成 .SZ，而沪深300实际是 .SH；错误的代码会返回空结果。"""
    seen: list[dict] = []

    def capture(arguments: dict) -> dict:
        seen.append(arguments)
        return constituents_payload()

    frame = make_adapter({CONSTITUENTS_DATASET: capture}).fetch_index_constituents("000300").df

    assert seen[0]["thscode"] == "000300.SH"
    assert list(frame["symbol"]) == ["600519", "000858"]
    assert list(frame["name"]) == ["贵州茅台", "五粮液"]


@pytest.mark.parametrize("index", ["886042.TI", "399300", "unknown-index"])
def test_an_unmapped_index_reports_not_supported_so_akshare_takes_over(index):
    adapter = make_adapter({CONSTITUENTS_DATASET: constituents_payload()})

    with pytest.raises(NotImplementedError):
        adapter.fetch_index_constituents(index)

    assert adapter.seen == []


# -- 业务错误码 ------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (1001, False),  # 缺参数
        (1002, False),  # 参数格式错
        (2001, False),  # 未认证
        (2003, False),  # 权限不足
        (4001, True),   # 限流
        (5001, True),   # 服务端错误
        (5002, True),   # 上游超时
        (5003, True),   # 上游不可用
    ],
)
def test_business_codes_map_to_the_right_retryability(code, retryable):
    """HTTP 恒为 200，业务失败只在 code 里，因此必须显式翻译这些码。"""
    adapter = make_adapter({QUOTE_DATASET: {"code": code, "message": "boom"}})

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_quote("600519")

    assert exc.value.retryable is retryable
    assert f"code={code}" in str(exc.value)


@pytest.mark.parametrize("code", [3001, 3004])
def test_no_data_codes_are_not_retryable_and_hand_off_to_the_next_source(code):
    """3001「标的不存在」与 3004「类型不支持」是确定的缺席：不重试，换源。

    3002「数据未就绪」**不在**此列——它是歧义的（可能只是快照预热），由适配器自己
    做有界重试，见下面 3002 的专门用例。
    """
    adapter = make_adapter({QUOTE_DATASET: {"code": code, "message": "无数据"}})

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_quote("600519")

    assert exc.value.retryable is False
    assert "无此数据" in str(exc.value)


def test_a_missing_api_key_fails_without_sending_anything():
    adapter = make_adapter({QUOTE_DATASET: quote_payload()}, api_key=None)

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_quote("600519")

    assert "未配置" in str(exc.value)
    assert exc.value.retryable is False
    assert adapter.seen == []


def test_a_payload_without_the_envelope_is_treated_as_the_data_itself():
    """没有 code 就直接给数据体的包装层不该被当成失败。"""
    adapter = make_adapter({QUOTE_DATASET: {"item": quote_payload()["data"]["item"]}})

    assert adapter.fetch_quote("600519").df.iloc[0]["close"] == 1500.0


# -- 通用数据集 ------------------------------------------------------------


def test_fetch_dataset_flattens_nested_records():
    """嵌套 dict 展开为点号列，让形态各异的端点都能落进同一张表。"""
    payload = envelope(
        {
            "count": 1,
            "stock_items": [
                {"ticker": "600519", "name": "贵州茅台", "concept_list": ["白酒", "消费"],
                 "org": {"net_value": 1.2e8, "buy_num": 3}}
            ],
        }
    )
    adapter = make_adapter({"get_a_share_special_data_dragon_tiger_list": payload})

    frame = adapter.fetch_dataset("a-share", "get_a_share_special_data_dragon_tiger_list", {}).df

    assert frame.iloc[0]["name"] == "贵州茅台"
    assert frame.iloc[0]["org.net_value"] == 1.2e8
    # 列表序列化为可读字符串；展开会让列数取决于数据。
    assert frame.iloc[0]["concept_list"] == "白酒、消费"


def test_fetch_dataset_handles_a_scalar_payload_as_a_single_row():
    payload = envelope({"quota": 100, "used": 3})
    adapter = make_adapter({"get_fund_quota_summary": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_quota_summary", {}).df

    assert frame.iloc[0]["quota"] == 100


def test_fetch_dataset_converts_millisecond_dates():
    payload = envelope({"item": [{"ex_date_ms": ms(2025, 6, 20), "dividend_per_share": 0.5}]})
    adapter = make_adapter({"get_a_share_corporate_actions_adjustment_factors": payload})

    frame = adapter.fetch_dataset("a-share", "get_a_share_corporate_actions_adjustment_factors", {}).df

    assert frame.iloc[0]["ex_date"].strftime("%Y-%m-%d") == "2025-06-20"


def test_fetch_dataset_passes_arguments_through_untouched():
    """参数由调用方按服务端 schema 组织，适配器不再二次加工。"""
    seen: list[dict] = []

    def capture(arguments: dict) -> dict:
        seen.append(arguments)
        return envelope({"item": []})

    adapter = make_adapter({"get_a_share_special_data_limit_up_pool": capture})
    adapter.fetch_dataset(
        "a-share", "get_a_share_special_data_limit_up_pool", {"date": "2026-09-18"}
    )

    assert seen[0] == {"date": "2026-09-18"}


def test_list_datasets_returns_the_runtime_catalog():
    """目录来自服务端 tools/list，因此上游新增端点无需改本地代码。"""
    adapter = make_adapter({QUOTE_DATASET: quote_payload(), KLINE_DATASET: kline_payload()})

    tools = adapter.list_datasets("a-share")

    assert {tool["name"] for tool in tools} == {QUOTE_DATASET, KLINE_DATASET}


def test_list_datasets_rejects_an_unknown_service():
    adapter = make_adapter({})

    with pytest.raises(AdapterError):
        adapter.list_datasets("crypto")


@pytest.mark.parametrize(
    ("dataset", "service"),
    [
        ("get_a_share_prices_snapshot", "a-share"),
        ("get_a_share_index_catalog_ths_index_list", "a-share-index"),
        ("get_meta_tickers_search", "meta"),
        ("get_fund_portfolio_holdings", "fund"),
        ("get_futures_prices_daily", "futures"),
        ("get_options_varieties_list", "options"),
    ],
)
def test_the_service_is_derived_from_the_dataset_name_prefix(dataset, service):
    """前缀推导免去一张会随上游新增工具而失效的映射表。"""
    assert _service_for_dataset(dataset) == service


# --- 长尾时间列的通用毫秒转换 -------------------------------------------
# 基金持仓一次就带来 start/end/publish/modify 四个 ``*_ms`` 字段；逐个登记一张全量
# 清单必然漏，因此只按 ``_ms`` 后缀 + 量级判定转换。以下钉住该规则的行为。


def test_arbitrary_ms_columns_are_converted_to_dates():
    """``publish_date_ms`` 这类未登记列也要变成日期，而不是裸毫秒数。"""
    payload = envelope(
        {"item": [{"ticker": "600519", "publish_date_ms": ms(2026, 7, 21), "hold_ratio": 5.7}]}
    )
    adapter = make_adapter({"get_fund_portfolio_holdings": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_portfolio_holdings", {}).df

    assert frame["publish_date_ms"].dtype.kind == "M"
    assert frame.iloc[0]["publish_date_ms"].strftime("%Y-%m-%d") == "2026-07-21"


def test_an_ms_suffix_column_of_the_wrong_magnitude_is_left_alone():
    """恰好以 ``_ms`` 结尾、但量级不是毫秒时间戳（如毫秒时长）不得被误转。"""
    payload = envelope({"item": [{"ticker": "600519", "duration_ms": 1500}]})
    adapter = make_adapter({"get_fund_portfolio_holdings": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_portfolio_holdings", {}).df

    assert frame.iloc[0]["duration_ms"] == 1500
    assert frame["duration_ms"].dtype.kind != "M"


def test_a_mix_of_garbage_and_ms_values_is_not_converted():
    """多数值不像时间戳时不转：宁可留原始值，也不把噪音转成 2001 年的日期。"""
    payload = envelope(
        {"item": [{"ticker": "a", "weird_ms": 3}, {"ticker": "b", "weird_ms": 7}]}
    )
    adapter = make_adapter({"get_fund_portfolio_holdings": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_portfolio_holdings", {}).df

    assert list(frame["weird_ms"]) == [3, 7]


# --- 3002「数据未就绪」的有界重试 ---------------------------------------
# 实测：get_fund_market_snapshot 连发三次，首次 code=3002，后两次 code=0。把它当
# "确定没有数据"会让一次上游冷启动抖动被误判为该源无此数据，直接回退。


def test_a_readiness_code_is_retried_once_and_can_succeed():
    """首次 3002、随后成功时，适配器应自己重试拿到数据，而不是上报失败。"""
    calls = {"n": 0}

    def handler(arguments: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"code": 3002, "message": "Fund market snapshot is not available yet"}
        return envelope({"item": [{"thscode": "510300.SH", "last_price": 4.582}]})

    adapter = make_adapter({"get_fund_market_snapshot": handler})
    result = adapter.fetch_dataset("fund", "get_fund_market_snapshot", {})

    assert calls["n"] == 2
    assert len(result.df) == 1


def test_a_readiness_code_that_persists_is_reported_not_ready():
    """重试预算用尽后仍失败，则如实上报为"未就绪"——不可重试，交由回退链换源。"""
    adapter = make_adapter(
        {"get_fund_market_snapshot": {"code": 3002, "message": "not available yet"}}
    )

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_dataset("fund", "get_fund_market_snapshot", {})

    assert exc.value.retryable is False
    assert "未就绪" in str(exc.value)


def test_a_not_ready_period_is_skippable_within_a_series():
    """逐期取指标时，某一期未披露只该跳过该期，不该丢掉整条序列。

    这正是 3002 必须有独立类型的理由：``fetch_indicators`` 按期间循环，需要把
    "这一期还没披露"与"这次调用失败"分开处理。
    """
    def handler(arguments: dict) -> dict:
        if arguments.get("report") == "2025-4":
            return {"code": 3002, "message": "数据未就绪"}
        return indicators_payload(arguments["report"], 30.5, 91.2)

    adapter = make_adapter({INDICATORS_DATASET: handler})
    frame = adapter.fetch_indicators("600519", 1, None).df

    # 四期里有一期未就绪，其余三期照常返回。
    assert len(frame) == 3


# --- 时间列识别：名字线索 + 量级 ----------------------------------------

def test_a_date_column_without_any_suffix_is_still_converted():
    """``nav_date`` 没有任何后缀，却正是裸毫秒（实测 1.78966e+12）。"""
    payload = envelope({"item": [{"unit_nav": 4.5793, "nav_date": ms(2026, 9, 18)}]})
    adapter = make_adapter({"get_fund_performance_nav": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_performance_nav", {}).df

    assert frame["nav_date"].dtype.kind == "M"
    assert frame.iloc[0]["nav_date"].strftime("%Y-%m-%d") == "2026-09-18"


def test_a_large_number_in_a_time_named_column_is_not_misread_as_a_date():
    """名字像时间但量级不对（如毫秒时长 1500）不得被当成 1970 年的日期。"""
    payload = envelope({"item": [{"duration_ms": 1500, "settle_time": 42}]})
    adapter = make_adapter({"get_fund_performance_nav": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_performance_nav", {}).df

    assert frame.iloc[0]["duration_ms"] == 1500
    assert frame.iloc[0]["settle_time"] == 42
    assert frame["duration_ms"].dtype.kind != "M"
    assert frame["settle_time"].dtype.kind != "M"


def test_an_ordinary_large_value_without_a_time_name_is_left_alone():
    """量级像时间戳但与时间无关的字段（如巨额市值）不能被误转。

    注意 ``turnover`` 已在 ``FUYAO_COLUMN_MAP`` 里映射为 ``amount``，因此用一个
    不在映射表中的字段来验证这条规则。
    """
    payload = envelope({"item": [{"market_value": 3.2e12, "capital_flow": 1.85e12}]})
    adapter = make_adapter({"get_fund_performance_nav": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_performance_nav", {}).df

    assert frame.iloc[0]["market_value"] == 3.2e12
    assert frame.iloc[0]["capital_flow"] == 1.85e12
    assert frame["market_value"].dtype.kind != "M"
    assert frame["capital_flow"].dtype.kind != "M"


# --- 期货与基金：新增域的归一化回归 -------------------------------------
# 实测确认：期货日 K（rb/IF/CU/SC）、基金行情与净值都经同一条通用通道可用。


def test_futures_daily_kline_lands_on_the_market_contract():
    """期货日 K 用 open_price/close_price/timestamp，必须归一到内部行情列名。"""
    payload = envelope(
        {"item": [
            {"timestamp": ms(2026, 9, 1), "open_price": 3146, "high_price": 3149,
             "low_price": 3107, "close_price": 3123, "volume": 616523, "turnover": 19252417000},
        ]}
    )
    adapter = make_adapter({"get_futures_prices_daily": payload})

    frame = adapter.fetch_dataset("futures", "get_futures_prices_daily", {}).df

    for column in ("open", "high", "low", "close", "volume", "amount"):
        assert column in frame.columns, column
    assert frame.iloc[0]["close"] == 3123
    # timestamp 不是 date_ms，靠"名字含 time + 毫秒量级"被识别并转成日期。
    assert frame["timestamp"].dtype.kind == "M"
    assert frame.iloc[0]["timestamp"].strftime("%Y-%m-%d") == "2026-09-01"


def test_fund_nav_date_is_converted_for_the_nav_endpoint():
    """净值端点的 nav_date 是裸毫秒且无后缀；不转就是 1.79e12 这种数字。"""
    payload = envelope({"item": [{"nav_date": ms(2026, 9, 18), "unit_nav": 4.5793, "adj_nav": 2.1518}]})
    adapter = make_adapter({"get_fund_performance_nav": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_performance_nav", {}).df

    assert frame.iloc[0]["nav_date"].strftime("%Y-%m-%d") == "2026-09-18"
    assert frame.iloc[0]["unit_nav"] == 4.5793


def test_fund_holdings_ms_columns_are_all_converted():
    """基金持仓一次带来 start/end/publish/modify 四个时间列，全部要转。"""
    payload = envelope(
        {"item": [{
            "thscode": "300750.SZ", "stock_name": "宁德时代", "hold_ratio": 4.2,
            "start_date_ms": ms(2026, 4, 1), "end_date_ms": ms(2026, 6, 30),
            "publish_date_ms": ms(2026, 7, 21), "modify_time_ms": ms(2026, 7, 21),
        }]}
    )
    adapter = make_adapter({"get_fund_portfolio_holdings": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_portfolio_holdings", {}).df

    for column in ("start_date_ms", "end_date_ms", "publish_date_ms", "modify_time_ms"):
        assert frame[column].dtype.kind == "M", column
    assert frame.iloc[0]["end_date_ms"].strftime("%Y-%m-%d") == "2026-06-30"


def test_fund_returns_percentages_are_left_as_raw_numbers():
    """收益率是百分数原值（8.88 表示 8.88%），不得被当成时间戳或做单位换算。"""
    payload = envelope({"item": [{"return_month": -4.35, "return_year": 2.22, "return_now": 113.68}]})
    adapter = make_adapter({"get_fund_performance_returns": payload})

    frame = adapter.fetch_dataset("fund", "get_fund_performance_returns", {}).df

    assert frame.iloc[0]["return_month"] == -4.35
    assert frame.iloc[0]["return_now"] == 113.68
