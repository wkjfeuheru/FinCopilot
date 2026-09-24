from datetime import date

import pytest

from finharness.data.mapping import (
    DEFAULT_VALUATION_INDICATOR,
    EXCHANGE_PREFIXES,
    UnknownExchangePrefix,
    exchange_prefix,
    indicator_field_matches,
    normalize_valuation_indicator,
    prefixed_symbol,
    select_indicator_columns,
)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("600519", "SH"),  # 沪市主板
        ("688981", "SH"),  # 科创板
        ("900901", "SH"),  # 沪市B股
        ("000001", "SZ"),  # 深市主板
        ("002594", "SZ"),  # 中小板并入主板
        ("300750", "SZ"),  # 创业板
        ("200011", "SZ"),  # 深市B股
        ("430047", "BJ"),  # 北交所
        ("831010", "BJ"),
        ("870204", "BJ"),
        ("920002", "BJ"),
    ],
)
def test_exchange_prefix_covers_every_board(symbol, expected):
    assert exchange_prefix(symbol) == expected


def test_prefixed_symbol_upper_and_lower():
    assert prefixed_symbol("600519") == "SH600519"
    assert prefixed_symbol("600519", lower=True) == "sh600519"
    assert prefixed_symbol("300750", lower=True) == "sz300750"


@pytest.mark.parametrize("symbol", ["999999", "123456", "abc123"])
def test_unknown_segment_fails_loudly_instead_of_guessing(symbol):
    """猜测出来的前缀会让某些上游 interface 返回空 frame，
    随后又被缓存成"无数据"。"""
    with pytest.raises(UnknownExchangePrefix):
        exchange_prefix(symbol)


def test_prefix_table_has_no_single_character_keys():
    assert all(len(key) >= 2 for key in EXCHANGE_PREFIXES)


def test_blank_valuation_indicator_defaults_to_pe_ttm():
    assert normalize_valuation_indicator(None) == DEFAULT_VALUATION_INDICATOR
    assert normalize_valuation_indicator("  ") == DEFAULT_VALUATION_INDICATOR


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("市盈率", "市盈率(TTM)"),
        ("pe", "市盈率(TTM)"),
        ("PE_TTM", "市盈率(TTM)"),
        ("市净率", "市净率"),
        ("总市值", "总市值"),
        ("市值", "总市值"),
        ("市盈率(静)", "市盈率(静)"),
    ],
)
def test_valuation_indicator_aliases_normalize(alias, expected):
    assert normalize_valuation_indicator(alias) == expected


def test_unknown_valuation_indicator_fails_loudly():
    """转发一个未知指标会返回另一个指标的数据序列。"""
    with pytest.raises(ValueError, match="不支持的估值指标"):
        normalize_valuation_indicator("净资产收益率")


@pytest.mark.parametrize(
    ("column", "field", "expected"),
    [
        ("净资产收益率(%)", "ROE", True),   # 中文标签的英文简写
        ("总资产报酬率(%)", "ROA", True),
        ("销售毛利率(%)", "毛利率", True),
        ("销售毛利率(%)", "净利润", False),
        ("资产负债率(%)", "负债", True),     # 普通子串匹配
    ],
)
def test_indicator_field_matching_resolves_aliases(column, field, expected):
    assert indicator_field_matches(column, field) is expected


def test_select_indicator_columns_keeps_date_and_reports_unmatched():
    columns = ["date", "销售净利率(%)", "净资产收益率(%)", "资产负债率(%)"]

    kept, unmatched = select_indicator_columns(columns, ["ROE", "权益乘数"])

    # ROE 通过别名解析；权益乘数 在此处没有对应列，因此会被报告出来。
    assert kept == ["date", "净资产收益率(%)"]
    assert unmatched == ["权益乘数"]


def test_select_indicator_columns_falls_back_to_all_when_nothing_matches():
    """没有匹配项时必须返回可用的 frame，而不是只有 date 的残缺表。"""
    columns = ["date", "销售净利率(%)", "资产负债率(%)"]

    kept, unmatched = select_indicator_columns(columns, ["毛利率"])

    assert kept == columns
    assert unmatched == ["毛利率"]


def test_every_macro_indicator_declares_its_publisher_and_cadence():
    """时效披露依赖这些静态事实：缺了它们，工具只能报出一个裸数据期。

    发布机构与节奏是关于「该指标如何发布」的事实，与本次取到的数值无关，因此集中
    声明在映射表里，避免分散到各工具或提示词中重复维护。
    """
    from finharness.data.mapping import MACRO_INDICATORS

    for slug, spec in MACRO_INDICATORS.items():
        assert spec.publisher, f"{slug} 缺少发布机构"
        assert spec.cadence, f"{slug} 缺少发布节奏"
        assert spec.frequency in {"daily", "weekly", "monthly", "quarterly", "yearly"}


# --- 同花顺映射（docs 03.5） -----------------------------------------

def test_thscode_appends_the_exchange_suffix():
    from finharness.data.mapping import thscode

    assert thscode("600519") == "600519.SH"
    assert thscode("000858") == "000858.SZ"
    assert thscode("300750") == "300750.SZ"
    assert thscode("688981") == "688981.SH"
    assert thscode("830799") == "830799.BJ"


def test_thscode_rejects_an_unknown_board():
    """无法识别号段时报错而非猜测：错误的后缀会让上游返回空结果，
    而空结果会被缓存成"无数据"。"""
    from finharness.data.mapping import UnknownExchangePrefix, thscode

    with pytest.raises(UnknownExchangePrefix):
        thscode("777777")


def test_every_fuyao_service_has_a_path_and_a_description():
    from finharness.data.mapping import FUYAO_SERVICE_PATHS, FUYAO_SERVICES

    assert set(FUYAO_SERVICE_PATHS) == set(FUYAO_SERVICES)
    for service, path in FUYAO_SERVICE_PATHS.items():
        assert path.startswith("/mcp/"), service
        assert FUYAO_SERVICES[service], service


def test_fuyao_endpoints_only_name_typed_methods():
    """只有存在 typed 适配器方法的数据集才登记在此；长尾端点经通用派发器触达，
    其名字来自服务端 ``tools/list``，不应在此重复维护。"""
    from finharness.data.mapping import FUYAO_ENDPOINTS

    assert set(FUYAO_ENDPOINTS) == {
        "quote",
        "kline",
        "financials:利润",
        "financials:资产",
        "financials:现金流",
        "indicators",
        "index_constituents",
    }
    for dataset in FUYAO_ENDPOINTS.values():
        assert dataset.startswith("get_"), dataset


@pytest.mark.parametrize(
    ("period", "years", "expected_first"),
    [
        (date(2026, 9, 20), 1, "2026-2"),
        (date(2026, 1, 15), 1, "2025-4"),
        (date(2026, 12, 31), 1, "2026-3"),
    ],
)
def test_quarterly_periods_start_from_the_last_disclosed_quarter(period, years, expected_first):
    """尚未披露的季度会被上游以 code=3002 拒绝，因此不能乐观地从当季开始。"""
    from finharness.data.mapping import fuyao_report_periods

    assert fuyao_report_periods(years, today=period)[0] == expected_first


def test_quarterly_periods_cover_the_requested_span():
    from finharness.data.mapping import fuyao_report_periods

    assert len(fuyao_report_periods(3, today=date(2026, 9, 20))) == 12


def test_long_lookbacks_fall_back_to_annual_reports():
    """5 年按季度取就是 20 次请求；长期回看改用年报，使请求数保持有界。"""
    from finharness.data.mapping import fuyao_report_periods

    periods = fuyao_report_periods(5, today=date(2026, 9, 20))

    assert periods == ["2025-4", "2024-4", "2023-4", "2022-4", "2021-4"]
    assert all(p.endswith("-4") for p in periods)


def test_annual_periods_respect_the_april_disclosure_deadline():
    """年报法定截止是次年 4 月 30 日，因此 5 月之前最近一期年报是前年的。"""
    from finharness.data.mapping import fuyao_report_periods

    assert fuyao_report_periods(4, today=date(2026, 5, 1))[0] == "2025-4"
    assert fuyao_report_periods(4, today=date(2026, 2, 1))[0] == "2024-4"


def test_period_count_is_capped_regardless_of_the_requested_span():
    """请求数必须是有界常量，而不是会话长度的函数。"""
    from finharness.data.mapping import (
        FUYAO_INDICATOR_MAX_PERIODS,
        fuyao_report_periods,
    )

    assert len(fuyao_report_periods(50, today=date(2026, 9, 20))) == FUYAO_INDICATOR_MAX_PERIODS


def test_fuyao_indicator_labels_are_reachable_by_the_english_aliases():
    """``fields=["ROE"]`` 走的是中文子串匹配，因此指标名若不翻译就会静默一无所获。"""
    from finharness.data.mapping import FUYAO_INDICATOR_LABELS

    for field in ("ROE", "ROA", "NPM", "GPM"):
        assert any(
            indicator_field_matches(label, field) for label in FUYAO_INDICATOR_LABELS.values()
        ), f"{field} 无法命中任何同花顺指标标签"


def test_fuyao_column_map_targets_the_internal_market_contract():
    """行情列必须落在 MARKET_COLUMNS 内；否则 K 线渲染的摘要会读不到 close。"""
    from finharness.data.mapping import FUYAO_COLUMN_MAP, MARKET_COLUMNS

    for source, target in FUYAO_COLUMN_MAP.items():
        assert target in MARKET_COLUMNS or target in {"symbol", "pct_change", "pre_close"}, (
            source,
            target,
        )


def test_ms_date_conversion_and_renaming_live_in_one_table():
    """毫秒列同时负责改名，因此不得再出现在通用列映射里——两处都写会让
    "先改名再转换"之类的顺序错误看起来无害。"""
    from finharness.data.mapping import FUYAO_COLUMN_MAP, FUYAO_MS_DATE_COLUMNS

    assert not set(FUYAO_MS_DATE_COLUMNS) & set(FUYAO_COLUMN_MAP)
    assert FUYAO_MS_DATE_COLUMNS["date_ms"] == "date"


def test_adjust_map_covers_the_internal_vocabulary():
    from finharness.data.mapping import FUYAO_ADJUST_MAP, FUYAO_DEFAULT_ADJUST

    assert FUYAO_ADJUST_MAP == {"qfq": "forward", "hfq": "backward"}
    assert FUYAO_DEFAULT_ADJUST == "none"


# --- 同花顺标的代码解析 -------------------------------------------------

@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("600519", "600519.SH"),  # 沪市主板
        ("300750", "300750.SZ"),  # 创业板
        ("830799", "830799.BJ"),  # 北交所
        ("688981", "688981.SH"),  # 科创板
    ],
)
def test_fuyao_thscode_resolves_a_share_codes(code, expected):
    """``fuyao_thscode`` 只解析 **A 股号段**。

    这不是保守做法而是实测事实：同花顺的 A 股端点拒绝非 A 股 thscode——把
    ``510300.SH``（ETF）发给 ``prices/snapshot`` 会返回 ``code=1002 Unknown
    A-share thscode``。给 ETF 补后缀并不能让它经 A 股通道取到数据。
    """
    from finharness.data.mapping import fuyao_thscode

    assert fuyao_thscode(code) == expected


@pytest.mark.parametrize("code", ["510300", "159915", "588000", "161725"])
def test_fuyao_thscode_refuses_non_a_share_codes(code):
    """ETF/LOF 号段必须原样拒绝：猜一个后缀只会白费一次注定失败的请求。

    真要用它们，有两条正确路径：经 A 股端点（它会明确报"不是 A 股代码"），或经
    fund 服务的 ``get_fund_market_snapshot``（实测可用，见下面的按服务翻译测试）。
    """
    from finharness.data.mapping import UnknownExchangePrefix, fuyao_thscode

    with pytest.raises(UnknownExchangePrefix):
        fuyao_thscode(code)


def test_the_non_a_share_error_is_a_fallback_reason_not_a_fatal_one():
    """异常类型是契约的一部分：``UnknownExchangePrefix`` 会让编排器**继续回退**。

    若抛普通 ``ValueError``，``DataAccess`` 会把它当成"调用方参数非法"直接中断整条
    请求，使一个 akshare 本可作答的代码在第一个适配器上就失败。
    """
    from finharness.data.mapping import UnknownExchangePrefix, fuyao_thscode

    assert issubclass(UnknownExchangePrefix, ValueError)  # 仍是 ValueError 的子类

    with pytest.raises(UnknownExchangePrefix):
        fuyao_thscode("510300")


def test_fuyao_thscode_passes_through_a_complete_thscode():
    """调用方自己查过标的检索时，不该被再加工一次。"""
    from finharness.data.mapping import fuyao_thscode

    assert fuyao_thscode("886042.TI") == "886042.TI"
    assert fuyao_thscode("000001.OF") == "000001.OF"


@pytest.mark.parametrize("code", ["abc", "12345", "1234567", "   ", "999999"])
def test_fuyao_thscode_refuses_what_it_cannot_resolve(code):
    """猜一个后缀比报错更糟：上游会返回另一个标的的数据，而结果看起来正常。"""
    from finharness.data.mapping import fuyao_thscode

    with pytest.raises(ValueError):
        fuyao_thscode(code)


def test_stock_prefixes_win_over_fund_prefixes():
    """000001 既是平安银行也是华夏成长（场外）；按股票口径解析是本模块的有意取舍。

    场外基金必须由调用方给出完整 thscode（000001.OF），因为从代码无法区分。
    """
    from finharness.data.mapping import fuyao_thscode

    assert fuyao_thscode("000001") == "000001.SZ"


def test_the_symbol_convenience_params_map_to_the_upstream_names():
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"symbol": "600519"}) == {"thscode": "600519.SH"}
    assert fuyao_arguments({"symbols": ["600519", "000858"]}) == {
        "thscodes": "600519.SH,000858.SZ"
    }
    # 已经是 thscode 的原样通过。
    assert fuyao_arguments({"thscode": "886042.TI"}) == {"thscode": "886042.TI"}


def test_empty_symbol_values_are_dropped_not_forwarded():
    """空值不该被翻译成一个空 thscode 发出去。"""
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"symbol": "", "limit": 5}) == {"limit": 5}
    assert fuyao_arguments({"symbols": []}) == {}


def test_only_ms_timestamp_params_are_converted_to_epoch():
    """date 类参数上游要 YYYY-MM-DD 字符串，只有 start/end 是毫秒时间戳。"""
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"start": "2026-09-01", "end": "20260918"}) == {
        "start": 1788192000000,
        "end": 1789660800000,
    }
    assert fuyao_arguments({"date": "2026-09-18"}) == {"date": "2026-09-18"}
    # 已经是毫秒的原样通过。
    assert fuyao_arguments({"start": 1788192000000}) == {"start": 1788192000000}


def test_an_illegal_date_is_refused_rather_than_dropped():
    from finharness.data.mapping import fuyao_arguments

    with pytest.raises(ValueError):
        fuyao_arguments({"start": "前天"})


def test_param_summary_marks_required_and_lists_enums():
    """枚举越界是上游最常见的参数错误，因此目录里要带上可选值。"""
    from finharness.data.mapping import fuyao_param_summary

    summary = fuyao_param_summary(
        {
            "type": "object",
            "properties": {
                "thscode": {"type": "string"},
                "period": {"enum": ["annual", "quarterly"]},
            },
            "required": ["thscode"],
        }
    )

    assert summary == "thscode, period?=annual|quarterly"



# --- 上游默认值的继承与披露 ---------------------------------------------
# 线上 a-share 服务为若干参数声明了实质性默认值（已对照真实 tools/list 确认）。
# 调用方省略它们时拿到的不是"全部"，而是上游选定的那一小撮，而请求里不留痕迹。

SNAPSHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "thscodes": {"type": "string", "default": "600519.SH,000001.SZ"},
        "limit": {"type": "integer", "default": 100},
        "offset": {"type": "integer", "default": 0},
    },
    "required": [],
}


def test_inherited_defaults_are_separated_from_what_the_caller_asked_for():
    """省略 thscodes 得到的是两只默认个股，不是全市场；这一区别必须可判定。"""
    from finharness.data.mapping import fuyao_inherited_defaults

    inherited = fuyao_inherited_defaults(SNAPSHOT_SCHEMA, {"limit": 5})

    assert inherited["thscodes"] == "600519.SH,000001.SZ"
    assert "limit" not in inherited  # 调用方自己指定了


def test_a_specified_parameter_is_never_reported_as_inherited():
    """两件事的含义相反：自己指定的 vs 上游替我们决定的。"""
    from finharness.data.mapping import fuyao_inherited_defaults

    inherited = fuyao_inherited_defaults(SNAPSHOT_SCHEMA, {"thscodes": "000858.SZ"})

    assert "thscodes" not in inherited
    assert inherited["limit"] == 100


def test_effective_arguments_merge_caller_values_over_defaults():
    from finharness.data.mapping import fuyao_effective_arguments

    effective = fuyao_effective_arguments(SNAPSHOT_SCHEMA, {"limit": 5})

    assert effective == {
        "thscodes": "600519.SH,000001.SZ",
        "limit": 5,
        "offset": 0,
    }


def test_parameters_without_defaults_are_not_invented():
    """没有默认值的参数保持缺席——凭空造一个值比省略它危险得多。"""
    from finharness.data.mapping import fuyao_inherited_defaults

    schema = {"properties": {"q": {"type": "string"}, "limit": {"type": "integer", "default": 10}}}

    assert fuyao_inherited_defaults(schema, {}) == {"limit": 10}


def test_inherited_defaults_are_empty_when_the_schema_is_missing():
    """目录取不到时不能编造默认值：那会把"未知"写成"上游如此"。

    调用方自己给的值不算继承，因此结果为空（而不是把它回显成"上游默认"）。
    """
    from finharness.data.mapping import fuyao_inherited_defaults

    assert fuyao_inherited_defaults(None, {"limit": 5}) == {}
    assert fuyao_inherited_defaults({}, {"limit": 5}) == {}


# --- 按服务翻译标的（同一代码在不同服务含义不同）-------------------------

def test_a_share_service_only_accepts_a_share_symbols():
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"symbol": "600519"}, service="a-share") == {"thscode": "600519.SH"}


def test_a_share_service_refuses_an_etf_symbol():
    """510300 在 A 股端点是无效代码；本地拒绝比发出去拿到 code=1002 更早、更清楚。"""
    from finharness.data.mapping import UnknownExchangePrefix, fuyao_arguments

    with pytest.raises(UnknownExchangePrefix):
        fuyao_arguments({"symbol": "510300"}, service="a-share")


def test_the_fund_service_resolves_etf_and_lof_symbols():
    """实测：``get_fund_market_snapshot("510300.SH")`` 返回最新价与成交量。"""
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"symbol": "510300"}, service="fund") == {"thscode": "510300.SH"}
    assert fuyao_arguments({"symbol": "159915"}, service="fund") == {"thscode": "159915.SZ"}
    assert fuyao_arguments({"symbol": "161725"}, service="fund") == {"thscode": "161725.SZ"}


def test_the_fund_service_still_accepts_plain_a_share_codes():
    """股票号段优先：``000001`` 是平安银行（也是某只场外基金，代码无法区分）。"""
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"symbol": "000001"}, service="fund") == {"thscode": "000001.SZ"}


def test_a_fund_symbol_list_is_translated_per_item():
    from finharness.data.mapping import fuyao_arguments

    assert fuyao_arguments({"symbols": ["510300", "159915"]}, service="fund") == {
        "thscodes": "510300.SH,159915.SZ"
    }


def test_a_complete_thscode_passes_through_under_every_service():
    """调用方自己查过标的检索时，不该被再加工——服务差异不影响已解析的代码。"""
    from finharness.data.mapping import fuyao_arguments

    for service in (None, "a-share", "fund", "futures"):
        assert fuyao_arguments({"thscode": "886042.TI"}, service=service) == {
            "thscode": "886042.TI"
        }
