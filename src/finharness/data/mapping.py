"""数据源接口名、列映射与交易所前缀的唯一事实来源。

新增一个数据接口时，只应扩展此处的映射表，绝不能把
来源特有的知识散落到各个适配器或工具中。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Final

# 同花顺的时间戳口径：Asia/Shanghai 午夜。用固定偏移而不是 zoneinfo，因为中国
# 自 1991 年起不再实行夏令时，而 ``Asia/Shanghai`` 在无 tzdata 的宿主上可能缺失。
_SHANGHAI_TZ: Final = timezone(timedelta(hours=8))

# --- 接口名 -------------------------------------------------------
# 映射 (adapter, 语义方法) -> 有序的候选接口名。第一个成功的候选接口
# 会被记录到 RawData.endpoint，因此引用与缓存键始终指向
# 实际提供数据的那个接口。
AKSHARE_ENDPOINTS: Final[dict[str, tuple[str, ...]]] = {
    # quote 优先使用 EM 快照（字段最丰富：name、pct_change、turnover），
    # 其次是腾讯日线序列（其最后一行即最新收盘价）。新浪
    # 快照是最后手段：它以无时间戳的请求分页抓取整个市场，
    # 且其官方文档警告会对频繁调用者封禁 IP，因此仅在
    # 更丰富的两个来源都不可用时才会用到。
    "quote": ("stock_zh_a_spot_em", "stock_zh_a_hist_tx", "stock_zh_a_spot"),
    "kline": ("stock_zh_a_hist", "stock_zh_a_daily", "stock_zh_a_hist_tx"),
    "indicators": ("stock_financial_analysis_indicator",),
    "financials": ("stock_financial_abstract",),
    "valuation": ("stock_zh_valuation_baidu",),
    "peers": ("stock_zh_valuation_comparison_em",),
    "news": ("stock_news_em",),
    "announcements": ("stock_individual_notice_report",),
    # 宏观/行业不是回退链：akshare 通过一个专用接口暴露每个经济
    # 序列，而行业/指数路由则从下方的申万/中证映射表
    # 解析得到。
    "macro": ("macro_china",),
    "industry": ("sw_index",),
}

TUSHARE_ENDPOINTS: Final[dict[str, tuple[str, ...]]] = {
    "quote": ("daily_basic",),
    "kline": ("daily",),
    "indicators": ("fina_indicator",),
    "financials": ("income",),
}

# --- 估值指标 -------------------------------------------------
# ``stock_zh_valuation_baidu`` 通过一个接口提供多个指标，并返回
# 一个仅含 ``date``/``value`` 的数据框，不带指标名。因此请求必须
# 携带指标，且其单位必须在渲染时附加 ——
# 否则市值数字会被读成市盈率倍数。
VALUATION_INDICATORS: Final[tuple[str, ...]] = (
    "总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率",
)
# 每个规范指标可接受的拼写（查找时会转为小写）。
VALUATION_INDICATOR_ALIASES: Final[dict[str, str]] = {
    "总市值": "总市值", "市值": "总市值", "total_mv": "总市值", "market_cap": "总市值",
    "市盈率(ttm)": "市盈率(TTM)", "市盈率": "市盈率(TTM)", "pe": "市盈率(TTM)",
    "pe_ttm": "市盈率(TTM)", "市盈率ttm": "市盈率(TTM)",
    "市盈率(静)": "市盈率(静)", "静态市盈率": "市盈率(静)",
    "pe_static": "市盈率(静)",
    "市净率": "市净率", "pb": "市净率",
    "市现率": "市现率", "pcf": "市现率",
}
# 百度以亿元为单位报告总市值；比率类指标则以倍数（倍）计。
VALUATION_INDICATOR_UNITS: Final[dict[str, str]] = {
    "总市值": "亿元", "市盈率(TTM)": "倍", "市盈率(静)": "倍",
    "市净率": "倍", "市现率": "倍",
}
DEFAULT_VALUATION_INDICATOR: Final[str] = "市盈率(TTM)"


def normalize_valuation_indicator(indicator: str | None) -> str:
    """将别名/空值映射为规范指标，并拒绝未知指标。

    显式报错：把无法识别的指标盲目转发给数据源，会在一个没有标签的
    ``value`` 列下返回错误指标的序列。
    """
    key = str(indicator or "").strip()
    if not key:
        return DEFAULT_VALUATION_INDICATOR
    if key in VALUATION_INDICATORS:
        return key
    mapped = VALUATION_INDICATOR_ALIASES.get(key) or VALUATION_INDICATOR_ALIASES.get(key.lower())
    if mapped is not None:
        return mapped
    raise ValueError(
        f"不支持的估值指标：{indicator}；可选 {'/'.join(VALUATION_INDICATORS)}"
    )


# --- 指标字段过滤 ----------------------------------------------
# ``get_indicators(fields=[...])`` 按关键字过滤列。调用方（以及模型）
# 用英文简写命名指标，而数据源使用中文标签，因此单纯的子串匹配
# 会静默地一无所获：请求 ``ROE`` 会丢掉恰好承载它的那一列
# （``净资产收益率(%)``），且不会给调用方任何信号。
INDICATOR_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "roe": ("净资产收益率",),
    "roa": ("总资产报酬率", "总资产净利率"),
    "eps": ("每股收益",),
    "bps": ("每股净资产",),
    "pe": ("市盈率",),
    "pb": ("市净率",),
    "pcf": ("市现率",),
    "npm": ("销售净利率", "净利率"),
    "gpm": ("销售毛利率", "毛利率"),
}


def indicator_field_matches(column: str, field: str) -> bool:
    """判断 ``field``（本身或其别名）是否指向 ``column``。"""
    label = str(column).lower()
    key = str(field).strip().lower()
    if not key:
        return False
    if key in label:
        return True
    return any(alias.lower() in label for alias in INDICATOR_FIELD_ALIASES.get(key, ()))


def select_indicator_columns(
    columns: tuple[str, ...] | list[str], fields: list[str]
) -> tuple[list[str], list[str]]:
    """返回 ``(保留的列, 未匹配到任何列的请求字段)``。

    ``date`` 始终保留。当没有任何字段匹配时，``kept`` 为全部列，
    以便调用方仍能得到可用的数据框，并且每个字段都会作为未匹配
    上报，而不会被静默吞掉。
    """
    matched: set[str] = set()
    unmatched: list[str] = []
    for field in fields:
        hits = [c for c in columns if str(c) != "date" and indicator_field_matches(c, field)]
        if hits:
            matched.update(hits)
        else:
            unmatched.append(field)
    if not matched:
        return list(columns), unmatched
    kept = [c for c in columns if str(c) == "date" or c in matched]
    return kept, unmatched

# --- 规范化行情列 -------------------------------------------
# quote/kline 数据框的内部契约（docs 03.5.2）。
MARKET_COLUMNS: Final[tuple[str, ...]] = (
    "date", "open", "high", "low", "close", "volume", "amount",
)

# akshare 中文列名 -> 内部 snake_case 名称。
AKSHARE_COLUMN_MAP: Final[dict[str, str]] = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
    "代码": "symbol",
    "名称": "name",
    "最新价": "close",
    "股票代码": "symbol",
}

# 每个语义接口的列别名，叠加在 AKSHARE_COLUMN_MAP 之上应用。
# 仅列出的接口会被规范化；其他接口保留其来源语义。
AKSHARE_INTERFACE_COLUMNS: Final[dict[str, dict[str, str]]] = {
    "quote": {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "涨跌幅": "pct_change",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    },
    "kline": {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    },
}

# --- 同行对比 -------------------------------------------------------
# 必须能在 `fields` 过滤后保留的身份列：若丢掉 代码/简称，
# 调用方就无法分辨哪一行是目标公司、哪些行是汇总值，
# 从而把同行表误读为每一行都是一家公司。
PEER_IDENTITY_COLUMNS: Final[tuple[str, ...]] = ("排名", "代码", "简称")
# EM 对比表把两个汇总行混入公司列表中，
# 在 代码/简称 列中标示出来。它们会打上 PEER_ROW_TYPE_COLUMN 标签，
# 使整表的平均值明显出错，而不是被静默污染。
PEER_STAT_LABELS: Final[tuple[str, ...]] = ("行业中值", "行业平均")
PEER_ROW_TYPE_COLUMN: Final[str] = "行类型"
PEER_COMPANY: Final[str] = "公司"
PEER_STAT: Final[str] = "行业统计"

# --- 交易所前缀 ----------------------------------------------------
# 基于号段（而非首位数字），以便创业板、科创板、B 股和北交所
# 都能被正确分类。最长前缀优先。
EXCHANGE_PREFIXES: Final[dict[str, str]] = {
    "600": "SH", "601": "SH", "603": "SH", "605": "SH", "688": "SH", "689": "SH",
    "900": "SH",
    "000": "SZ", "001": "SZ", "002": "SZ", "003": "SZ", "300": "SZ", "301": "SZ",
    "200": "SZ",
    "430": "BJ", "830": "BJ", "831": "BJ", "832": "BJ", "833": "BJ", "834": "BJ",
    "835": "BJ", "836": "BJ", "837": "BJ", "838": "BJ", "839": "BJ",
    "870": "BJ", "871": "BJ", "872": "BJ", "873": "BJ", "874": "BJ",
    "875": "BJ", "876": "BJ", "877": "BJ", "878": "BJ", "879": "BJ",
    "920": "BJ",
}


class UnknownExchangePrefix(ValueError):
    """当 6 位代码不匹配任何已知交易所号段时抛出。"""


def exchange_prefix(symbol: str) -> str:
    """为一支 6 位 A 股代码返回 ``SH``/``SZ``/``BJ``。

    显式报错而非猜测：错误的前缀会使某些上游接口返回空数据框，
    而该空结果又会被缓存为“无数据”。
    """
    for length in (3, 2):
        prefix = EXCHANGE_PREFIXES.get(symbol[:length])
        if prefix is not None:
            return prefix
    raise UnknownExchangePrefix(f"无法识别交易所前缀：{symbol}")


def prefixed_symbol(symbol: str, *, lower: bool = False) -> str:
    """返回带交易所前缀的代码，如 ``SH600519`` 或 ``sh600519``。"""
    result = f"{exchange_prefix(symbol)}{symbol}"
    return result.lower() if lower else result


# --- 宏观经济指标 -------------------------------------------
# 规范英文 slug -> 提供该指标的 akshare 接口，以及需要提取的
# 周期/数值列。一个接口常常承载多个序列
# （制造业 + 非制造业 PMI；M0/M1/M2），因此适配器按 ``source``
# 对请求的指标分组，并让每个接口只抓取一次。
@dataclass(frozen=True, slots=True)
class MacroSpec:
    label: str
    source: str
    period_col: str
    value_col: tuple[str, ...]
    unit: str
    frequency: str
    # 发布机构与发布节奏。它们不参与取数，只用于把「这是截至哪一期的数据」讲清楚：
    # 读者据此判断当前该看到哪一期、下一期何时发布（见 data/freshness.py）。
    publisher: str = ""
    cadence: str = ""


MACRO_INDICATORS: Final[dict[str, MacroSpec]] = {
    "pmi_manufacturing": MacroSpec(
        "制造业PMI", "pmi", "月份", ("制造业-指数",), "指数", "monthly",
        "中国物流与采购联合会", "当月最后一日发布",
    ),
    "pmi_non_manufacturing": MacroSpec(
        "非制造业PMI", "pmi", "月份", ("非制造业-指数",), "指数", "monthly",
        "中国物流与采购联合会", "当月最后一日发布",
    ),
    "cpi_yoy": MacroSpec(
        "CPI同比", "cpi", "月份", ("全国-同比增长",), "%", "monthly",
        "国家统计局", "次月上旬发布（约9-10日）",
    ),
    "ppi_yoy": MacroSpec(
        "PPI同比", "ppi", "月份", ("当月同比增长",), "%", "monthly",
        "国家统计局", "次月上旬发布（与CPI同日）",
    ),
    "m2_yoy": MacroSpec(
        "M2同比", "money_supply", "月份", ("货币和准货币(M2)-同比增长",), "%", "monthly",
        "中国人民银行", "次月中旬发布（约10-15日）",
    ),
    "m1_yoy": MacroSpec(
        "M1同比", "money_supply", "月份", ("货币(M1)-同比增长",), "%", "monthly",
        "中国人民银行", "次月中旬发布（约10-15日）",
    ),
    "social_financing": MacroSpec(
        "社融增量", "shrzgm", "月份", ("社会融资规模增量",), "亿元", "monthly",
        "中国人民银行", "次月中旬发布（约10-15日）",
    ),
    "lpr_1y": MacroSpec(
        "1年期LPR", "lpr", "TRADE_DATE", ("LPR1Y",), "%", "monthly",
        "全国银行间同业拆借中心", "每月20日发布",
    ),
    "lpr_5y": MacroSpec(
        "5年期LPR", "lpr", "TRADE_DATE", ("LPR5Y",), "%", "monthly",
        "全国银行间同业拆借中心", "每月20日发布",
    ),
    "shibor_on": MacroSpec(
        "隔夜SHIBOR", "shibor", "日期", ("O/N-定价",), "%", "daily",
        "全国银行间同业拆借中心", "每交易日发布",
    ),
    "bond_10y": MacroSpec(
        "10年期国债收益率", "bond", "日期", ("中国国债收益率10年",), "%", "daily",
        "中债金融估值中心", "每交易日发布",
    ),
    "usdcny": MacroSpec(
        "美元兑人民币", "currency", "日期", ("央行中间价", "中行折算价"), "元/美元", "daily",
        "中国外汇交易中心", "每交易日发布",
    ),
    "gdp_yoy": MacroSpec(
        "GDP同比", "gdp", "季度", ("国内生产总值-同比增长",), "%", "quarterly",
        "国家统计局", "季后约15-18日发布",
    ),
}
MACRO_INDICATOR_LABELS: Final[dict[str, str]] = {
    slug: spec.label for slug, spec in MACRO_INDICATORS.items()
}


def normalize_macro_indicator(name: str) -> str:
    """将输入的 slug 或中文标签映射为规范的宏观指标 slug。"""
    key = str(name or "").strip()
    if not key:
        raise ValueError("宏观指标名不能为空")
    if key in MACRO_INDICATORS:
        return key
    lowered = key.lower()
    for slug in MACRO_INDICATORS:
        if slug == lowered:
            return slug
    for slug, spec in MACRO_INDICATORS.items():
        if spec.label == key:
            return slug
    raise ValueError(
        f"不支持的宏观指标：{name}；可选 {', '.join(MACRO_INDICATORS)}"
    )


# --- 指数 / 行业选股池 ----------------------------------------------
# 场景选股池对应的中证指数代码。名称与代码均可接受，
# 便于模型既可说“沪深300”也可说“000300”。
INDEX_ALIASES: Final[dict[str, str]] = {
    "沪深300": "000300", "hs300": "000300", "csi300": "000300", "000300": "000300",
    "中证500": "000905", "zz500": "000905", "csi500": "000905", "000905": "000905",
    "中证1000": "000852", "zz1000": "000852", "csi1000": "000852", "000852": "000852",
    "上证50": "000016", "sz50": "000016", "000016": "000016",
    "科创50": "000688", "kc50": "000688", "000688": "000688",
    "上证指数": "000001", "000001": "000001",
}


def normalize_index(index: str) -> str:
    """将指数名称/代码解析为其中证代码，并拒绝未知者。"""
    key = str(index or "").strip()
    if not key:
        raise ValueError("指数名不能为空")
    resolved = INDEX_ALIASES.get(key) or INDEX_ALIASES.get(key.lower())
    if resolved is None:
        raise ValueError(f"不支持的指数：{index}；可选 {', '.join(sorted(set(INDEX_ALIASES)))}")
    return resolved


# =====================================================================
# 同花顺（iFinD / Fuyao MCP，docs 03.5）
# =====================================================================
# 同花顺以六个 MCP 服务暴露其能力，每个服务是一个 Streamable HTTP 端点。其标的
# 标识是 **thscode**——六位代码加交易所后缀（``600519.SH``），与本项目内部的
# 裸六位代码不同，因此每个请求都要在这里做一次转换，而不是让各适配器各自拼串。
#
# 下方三张表是同花顺知识的唯一存放处：服务清单、语义方法到数据集的映射、以及
# 列名/指标名的中文标签。按仓库约定（本模块 docstring），新增接口只扩表，绝不把
# 来源特有的知识散落到适配器或工具中。

# 服务名 -> 服务用途。工具层据此把 ``service`` 枚举暴露给模型；适配器据此拼 URL。
FUYAO_SERVICES: Final[dict[str, str]] = {
    "a-share": "A股行情、财报、估值、特色数据、交易日历、集合竞价",
    "a-share-index": "同花顺指数列表与成分股、指数行情",
    "meta": "标的检索与标的列表（thscode 解析的前置步骤）",
    "fund": "公募基金：净值、持仓、业绩、经理、F10",
    "futures": "期货：合约、持仓、仓单、基差、行情",
    "options": "期权：合约与行情",
}

# 服务名 -> 该服务的 MCP 端点路径。六个服务共用一个 API Key（请求头 X-api-key）。
FUYAO_SERVICE_PATHS: Final[dict[str, str]] = {
    "a-share": "/mcp/a-share",
    "a-share-index": "/mcp/a-share-index",
    "meta": "/mcp/meta",
    "fund": "/mcp/fund",
    "futures": "/mcp/futures",
    "options": "/mcp/options",
}

# --- 语义方法 -> 数据集 ------------------------------------------------
# 与 AKSHARE_ENDPOINTS / TUSHARE_ENDPOINTS 同构：适配器的 fetch_* 方法据此选取
# 数据集名，而该名字会进入 RawData.endpoint（``fuyao:<dataset>``），使引用与缓存
# 键始终指向真正提供数据的那个接口。
#
# 只登记**有 typed 适配器方法**的数据集。特色数据、基金/期货/期权与估值快照没有
# 对应的方法——它们经通用数据集派发器（``fetch_dataset``）触达，其数据集名来自
# 服务端运行时返回的 ``tools/list``，因此无需（也不应）在此列出。
FUYAO_ENDPOINTS: Final[dict[str, str]] = {
    "quote": "get_a_share_prices_snapshot",
    "kline": "get_a_share_prices_historical",
    "financials:利润": "get_a_share_financials_income_statements",
    "financials:资产": "get_a_share_financials_balance_sheets",
    "financials:现金流": "get_a_share_financials_cash_flow_statements",
    "indicators": "get_a_share_financials_indicators",
    "index_constituents": "get_a_share_index_constituents_ths_stock_list",
}

# --- 复权方式 ----------------------------------------------------------
# 内部契约是 ``None``（不复权）/``qfq``（前复权）/``hfq``（后复权），同花顺用
# ``none``/``forward``/``backward``。两套词表在这里对接，避免适配器里出现
# 只有它自己认识的字符串。
FUYAO_ADJUST_MAP: Final[dict[str, str]] = {
    "qfq": "forward",
    "hfq": "backward",
}
FUYAO_DEFAULT_ADJUST: Final[str] = "none"

# 同花顺的 K 线当前只提供日线。周/月线请求因此会落到 akshare——这不是缺陷，而是
# 一条正常的回退理由（适配器抛 NotImplementedError，编排器读作"该源不支持"）。
FUYAO_KLINE_PERIODS: Final[frozenset[str]] = frozenset({"day", "daily", "1d"})

# 历史窗口上限：``prices/historical`` 的 ``end - start`` 不得超过十年，超出返回
# code=1003。适配器据此截断，而不是把请求发出去等一个业务错误回来。
FUYAO_MAX_WINDOW_YEARS: Final[int] = 10

# --- 列名归一化 --------------------------------------------------------
# 同花顺用 snake_case 且已带单位语义（``last_price``/``turnover``），但内部行情契约
# 是 ``date,open,high,low,close,volume,amount``（MARKET_COLUMNS），因此需要一层映射。
# 尚未出现在约定列里的字段（``price_change`` 等）保留原名——它们是有效信息，丢弃
# 只会让渲染出来的表凭空少几个字段。
FUYAO_COLUMN_MAP: Final[dict[str, str]] = {
    "ticker": "symbol",
    "last_price": "close",
    "open_price": "open",
    "high_price": "high",
    "low_price": "low",
    "prev_price": "pre_close",
    "close_price": "close",
    "turnover": "amount",
    "price_change_ratio_pct": "pct_change",
}

# 毫秒时间戳列 -> 该列转换后的目标列名。同花顺的每个日期字段都是 Asia/Shanghai
# 午夜的毫秒 Unix 时间戳；直接交给 pandas 会被读成 1970 年的整数，因此必须显式
# 转换并在渲染前落地为日期。此表同时负责改名，故这些列不出现在 FUYAO_COLUMN_MAP
# 中——两处都写会让"先改名再转换"之类的顺序错误看起来无害。
FUYAO_MS_DATE_COLUMNS: Final[dict[str, str]] = {
    "date_ms": "date",
    "period_end_ms": "report_period",
    "report_date_ms": "report_date",
    "ex_date_ms": "ex_date",
}

# --- 财务报表字段标签 --------------------------------------------------
# 三张表的字段是一套稳定的英文字段名（``operating_income`` 等）。工具与计算层
# （calc_metrics / calc_valuation / make_chart）都按中文语义读取列名，因此这里把
# 标准字段译成中文标签，使渲染出来的表与其它数据源的表读起来一致。
FUYAO_FINANCIAL_LABELS: Final[dict[str, str]] = {
    # 利润表
    "operating_income": "营业收入",
    "operating_costs": "营业成本",
    "operating_expenses": "营业费用",
    "sales_fee": "销售费用",
    "manage_fee": "管理费用",
    "research_and_development_expenses": "研发费用",
    "operating_profit": "营业利润",
    "interest_expenses": "利息费用",
    "profit_total": "利润总额",
    "income_tax_expense": "所得税费用",
    "net_profit": "净利润",
    "parent_holder_net_profit": "归母净利润",
    "basic_eps": "基本每股收益",
    # 资产负债表
    "assets_total": "资产总计",
    "total_current_assets": "流动资产合计",
    "non_current_nets_total": "非流动资产合计",
    "cash": "货币资金",
    "accounts_receivable": "应收账款",
    "total_debt": "负债合计",
    "holder_equity_total": "所有者权益合计",
    # 现金流量表
    "act_cash_flow_net": "经营活动现金流净额",
    "invest_cash_flow_net": "投资活动现金流净额",
    "financing_cash_flow_net": "筹资活动现金流净额",
    "pay_fixed_assets_etc_cash": "购建固定资产等支付的现金",
    "pay_dividends_profits_interest_cash": "分配股利利润或偿付利息支付的现金",
    "cash_equivalents_net_addition": "现金及现金等价物净增加额",
}

# --- 财务指标标签 ------------------------------------------------------
# 同花顺的财务指标按 成长/盈利/偿债/营运/现金流 五类分组返回，每组内的指标以
# ``index_id`` 标识。这些是英文字段名，而 ``INDICATOR_FIELD_ALIASES`` 的关键字
# （roe/roa/…）是**中文子串**匹配，因此不翻译就意味着 ``fields=["ROE"]`` 会静默
# 一无所获——正是该别名表当初要避免的那种失败。
FUYAO_INDICATOR_LABELS: Final[dict[str, str]] = {
    # 成长
    "total_assets_growth_ratio": "总资产同比增长率",
    "net_profit_yoy_growth_ratio": "净利润同比增长率",
    "operating_income_yoy_growth_ratio": "营业收入同比增长率",
    "operating_profit_yoy_growth_ratio": "营业利润同比增长率",
    # 盈利
    "sale_gross_margin": "销售毛利率",
    "sale_net_interest_ratio": "销售净利率",
    "total_assets_net_ratio": "总资产报酬率",
    "index_deduct_weighted_avg_roe": "扣非加权净资产收益率",
    "index_weighted_avg_roe": "加权净资产收益率",
    # 偿债
    "current_ratio": "流动比率",
    "quick_ratio": "速动比率",
    "assets_debt_ratio": "资产负债率",
    "cash_ratio": "现金比率",
    "earned_interest_multiple": "已获利息倍数",
    # 营运
    "long_term_debt_equity_ratio": "长期负债权益比",
    "total_assets_turnover_ratio": "总资产周转率",
    "inventory_turnover_ratio": "存货周转率",
    "current_assets_turnover_ratio": "流动资产周转率",
    "receive_account_turnover_ratio": "应收账款周转率",
    # 现金流
    "cash_operating_index": "现金营运指数",
    "operating_cash_flow_net_divide_income": "经营现金流净额比营业收入",
    "net_profit_cash_content": "净利润现金含量",
    "operating_cash_net_yoy_growth_ratio": "经营现金流净额同比增长率",
    "cash_meet_invest_ratio": "现金满足投资比率",
}


def thscode(symbol: str) -> str:
    """六位 A 股代码 -> 同花顺 thscode（``600519`` -> ``600519.SH``）。

    交易所后缀由 ``exchange_prefix`` 按号段推导，与腾讯/新浪接口用的是同一张表，
    因此创业板的 ``300xxx``、科创板的 ``688xxx``、北交所的 ``8xxxxx`` 都能正确分类。
    """
    return f"{symbol}.{exchange_prefix(symbol)}"


# --- 财务指标的报告期取数策略 ------------------------------------------
# 同花顺的指标端点一次只返回**一个**报告期，因此要凑出 ``years`` 年的序列就得发
# 多次请求。这里的两个上限把它变成一个有界的常量，而不是会话长度的函数：
# 短期回看取季度（细节更重要），长期回看取年度（否则 5 年 = 20 次调用）。
FUYAO_INDICATOR_QUARTERLY_YEARS_MAX: Final[int] = 3
FUYAO_INDICATOR_MAX_PERIODS: Final[int] = 12


def fuyao_report_periods(years: int, *, today: date | None = None) -> list[str]:
    """为同花顺的指标端点生成报告期列表，最新在前。

    同花顺以 ``yyyy-N`` 表示报告期（N=1..4 分别对应一季报/中报/三季报/年报）。
    短期回看按季度取，长期回看按年度（``yyyy-4`` 即年报）取，两者都以
    ``FUYAO_INDICATOR_MAX_PERIODS`` 封顶，使最坏情况下的请求数可推理。
    """
    reference = today or date.today()
    span = max(int(years), 1)
    if span <= FUYAO_INDICATOR_QUARTERLY_YEARS_MAX:
        periods: list[str] = []
        year = reference.year
        # 当前年度尚未披露的季度会被上游以 code=3002（数据未就绪）拒绝，因此从
        # 最近一个**应当已披露**的季度起回看，而不是乐观地从当季开始。
        quarter = (reference.month - 1) // 3  # 1..4；0 表示一季报尚未披露
        for _ in range(span * 4):
            if quarter <= 0:
                year -= 1
                quarter = 4
            periods.append(f"{year}-{quarter}")
            quarter -= 1
        return periods[:FUYAO_INDICATOR_MAX_PERIODS]
    # 年报的法定披露截止是次年 4 月 30 日，因此在 5 月之前，最近一期的年报是
    # 前年的那一份。从它开始回看，避免向"尚未披露"的期间索取数据。
    latest_annual = reference.year - 1 if reference.month >= 5 else reference.year - 2
    return [f"{latest_annual - offset}-4" for offset in range(span)][
        :FUYAO_INDICATOR_MAX_PERIODS
    ]


# --- 指数代码 ----------------------------------------------------------
# 同花顺的指数以 thscode 寻址（``000300.SH``、``886042.TI``），而本项目内部用中证
# 裸代码（``000300``）。不能用股票号段表去猜后缀：``000300`` 按号段会得到 ``.SZ``，
# 而沪深300实际是 ``000300.SH``——一个错误的代码会让上游返回空结果，空结果又会被
# 缓存成"无数据"。
#
# 键与 ``INDEX_ALIASES`` 的取值集合一致：``DataAccess.index_constituents`` 先经
# ``normalize_index``，因此适配器只会看到这些规范代码。表里没有的指数（如同花顺的
# 概念指数）经数据集派发器按其 thscode 直接查询，不走这条路径。
FUYAO_INDEX_THSCODES: Final[dict[str, str]] = {
    "000300": "000300.SH",
    "000905": "000905.SH",
    "000852": "000852.SH",
    "000016": "000016.SH",
    "000688": "000688.SH",
    "000001": "000001.SH",
}


# --- 便捷标的参数 --------------------------------------------------------
# 这里只受理 **A 股号段**，且这是实测得出的事实而非保守做法：同花顺的 A 股端点
# 明确拒绝非 A 股 thscode——把 ``510300.SH``（ETF）发给 ``prices/snapshot`` 会返回
# ``code=1002 Unknown A-share thscode``。因此给 ETF/LOF/场外基金补后缀并不能让它
# 们经 A 股通道取到数据，只会白费一次请求。
#
# **非 A 股标的有各自的服务**：ETF/LOF 的二级市场行情走 fund 服务的
# ``get_fund_market_snapshot``（实测 ``510300.SH`` 返回最新价/成交量），场外基金走
# fund 的净值端点。那是数据集派发器（``query_fund_data``）的职责，不是 typed 语义
# 方法的——把两种标的塞进同一个方法，会让"这只股票"与"这只基金"共用一条取数路径，
# 而它们的字段、单位与语义并不同。


def fuyao_thscode(code: str) -> str:
    """六位 A 股代码 -> thscode（``600519`` -> ``600519.SH``）。

    只解析 A 股号段；其余代码（ETF/LOF/场外基金/可转债）抛 ``UnknownExchangePrefix``。

    **异常类型是有意的选择。** ``UnknownExchangePrefix`` 继承自 ``ValueError``，但
    ``DataAccess`` 对它有专门的分支，会把它读作"这个源提供不了这个标的"并**继续回退**
    到下一个数据源。若抛普通 ``ValueError``，它会被当成"调用方参数非法"而直接中断整条
    请求——那会让一个 akshare 本可作答的代码在第一个适配器上就失败。

    已经是 ``代码.后缀`` 形态的值原样通过：调用方若自己查过标的检索，不该被再加工。
    """
    text = str(code).strip()
    if "." in text:
        return text.upper()
    if not (len(text) == 6 and text.isdigit()):
        raise UnknownExchangePrefix(f"标的代码必须是 6 位数字：{code}")
    return f"{text}.{exchange_prefix(text)}"


# --- 数据集参数契约 -----------------------------------------------------
# 派发器把服务端 ``tools/list`` 里的 inputSchema 作为上游参数的唯一事实来源，因此
# 本表只承载**本地**需要知道的那点翻译：如何把模型容易给出的说法变成上游要的形态。
# 它不是 schema 的副本——上游新增参数无需改这里。

# 便捷标的参数 -> (目标上游参数名, 是否多值)。
# 模型知道六位代码，不知道 thscode；让它可以只传代码，而不必先做一次标的检索。
FUYAO_SYMBOL_PARAMS: Final[dict[str, tuple[str, bool]]] = {
    "symbol": ("thscode", False),
    "symbols": ("thscodes", True),
}

# 时间为毫秒 Unix 时间戳的参数。让模型可以直接写 ``YYYY-MM-DD``；写毫秒则原样
# 通过（两种都接受，因为文档口径是毫秒，而模型更可能给出日期）。
FUYAO_TIMESTAMP_PARAMS: Final[frozenset[str]] = frozenset({"start", "end"})

# 有意**不**为 limit 类参数注入本地默认值：上游 schema 自带默认（如全市场快照
# ``limit=100``、股票池 ``size=50``）。注入我们自己的数字会静默改写上游语义，
# 而结果的规模由渲染预算（``trim_dataframe``）兜住，不需要在取数层再砍一刀。


# 上市基金（ETF/LOF）的号段 -> 交易所。**只在 fund 服务下使用**：同花顺的 A 股端点
# 拒绝这些代码（``510300.SH`` 发给 ``prices/snapshot`` 返回 ``code=1002``），而 fund
# 服务的二级市场行情端点接收它们（实测 ``get_fund_market_snapshot("510300.SH")``
# 返回最新价与成交量）。因此同一串代码的含义取决于它发往哪个服务，翻译必须按服务区分。
FUYAO_TRADED_FUND_PREFIXES: Final[dict[str, str]] = {
    # 沪市：封闭式、LOF、ETF、REITs
    "500": "SH", "501": "SH", "502": "SH", "505": "SH", "506": "SH", "508": "SH",
    "510": "SH", "511": "SH", "512": "SH", "513": "SH", "515": "SH", "516": "SH",
    "517": "SH", "518": "SH", "519": "SH", "520": "SH", "521": "SH", "522": "SH",
    "523": "SH", "560": "SH", "561": "SH", "562": "SH", "563": "SH", "588": "SH",
    "589": "SH",
    # 深市：分级、ETF、LOF、REITs
    "150": "SZ", "151": "SZ", "152": "SZ", "153": "SZ", "154": "SZ", "155": "SZ",
    "156": "SZ", "157": "SZ", "158": "SZ", "159": "SZ", "160": "SZ", "161": "SZ",
    "162": "SZ", "163": "SZ", "164": "SZ", "165": "SZ", "166": "SZ", "167": "SZ",
    "168": "SZ", "169": "SZ", "180": "SZ",
}

# 受理上市基金标的的服务：只有 fund。其余服务按 A 股口径翻译，遇到基金代码时
# 报错而不是猜后缀——猜错会让上游返回另一个标的的数据，而结果看起来完全正常。
FUYAO_FUND_SERVICES: Final[frozenset[str]] = frozenset({"fund"})


def fuyao_arguments(
    params: dict[str, Any] | None, *, service: str | None = None
) -> dict[str, Any]:
    """把用户友好的参数翻译成上游要的形态。

    三件事：``symbol``/``symbols`` 转成 ``thscode``/``thscodes``（多值以逗号拼接，
    与上游的口径一致）、``YYYY-MM-DD`` 转毫秒时间戳，以及按**服务**选择标的号段表。

    ``service`` 决定用哪张号段表：A 股服务只认 A 股号段，fund 服务额外认 ETF/LOF
    号段。同一个六位代码在两个服务里的含义可能不同，因此这个参数不是可选的优化，
    而是正确性的一部分——用错号段表要么白费一次注定失败的请求，要么猜出一个让上游
    返回**另一个标的**的后缀。

    已经是 ``thscode`` 形态的值原样通过：调用方若自己查过标的检索，不该被再加工
    一次。翻译失败（非法代码、无法解析的日期）以 ``ValueError`` 上报，工具转成
    ``ok=False``——静默跳过会让请求带着一个错代码发出去，然后得到一份空数据。
    """
    accepts_funds = service in FUYAO_FUND_SERVICES
    translated: dict[str, Any] = {}
    for key, value in dict(params or {}).items():
        if key in FUYAO_SYMBOL_PARAMS:
            target, multi = FUYAO_SYMBOL_PARAMS[key]
            if value in (None, "", []):
                continue
            convert = _to_fund_thscode if accepts_funds else _to_thscode
            if multi:
                translated[target] = _to_thscode_list(value, convert=convert)
            else:
                translated[target] = convert(value)
            continue
        if key in FUYAO_TIMESTAMP_PARAMS:
            translated[key] = _to_epoch_ms(value)
            continue
        translated[key] = value
    return translated


def _to_thscode(value: Any) -> str:
    """单个标的值 -> A 股 thscode；已经是 ``代码.后缀`` 形态的原样返回。"""
    text = str(value).strip()
    if not text:
        raise ValueError("标的代码不能为空")
    return fuyao_thscode(text)


def _exchange_for_fund(code: str) -> str:
    """上市基金代码 -> 交易所后缀；号段不认识时抛 ``UnknownExchangePrefix``。"""
    prefix = FUYAO_TRADED_FUND_PREFIXES.get(code[:3])
    if prefix is None:
        raise UnknownExchangePrefix(
            f"无法从代码 {code} 确定交易所；若它是场外基金，"
            "请先用 get_meta_tickers_search 解析出完整 thscode 再传入"
        )
    return prefix


def _to_fund_thscode(value: Any) -> str:
    """单个标的值 -> fund 服务的 thscode：A 股与**已上市基金**号段都受理。

    解析顺序是股票号段优先（``000001`` 既是平安银行也是某只场外基金，而从代码无法
    区分；按股票口径解析是明确取舍），失败后落到上市基金号段。
    """
    text = str(value).strip()
    if not text:
        raise ValueError("标的代码不能为空")
    if "." in text:
        return text.upper()
    if not (len(text) == 6 and text.isdigit()):
        raise ValueError(f"标的代码必须是 6 位数字：{value}")
    try:
        return f"{text}.{exchange_prefix(text)}"
    except UnknownExchangePrefix:
        return f"{text}.{_exchange_for_fund(text)}"


def _to_thscode_list(value: Any, *, convert: Callable[[Any], str] = _to_thscode) -> str:
    """多个标的 -> 逗号分隔的 thscode 串（上游 ``thscodes`` 的口径）。"""
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    codes = [convert(item) for item in items if str(item).strip()]
    if not codes:
        raise ValueError("标的代码列表不能为空")
    return ",".join(codes)


def _to_epoch_ms(value: Any) -> int:
    """``YYYY-MM-DD`` 或 ``YYYYMMDD`` -> Asia/Shanghai 午夜的毫秒时间戳。

    已经是数字（毫秒）的值原样通过：文档口径就是毫秒，两种写法都该被接受。

    用标准库解析而不是 pandas：本模块是纯映射表，不引入重量级依赖（它此前只有
    ``dataclasses``/``datetime``）。
    """
    if isinstance(value, bool):
        raise ValueError(f"时间参数不能是布尔值：{value}")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("时间参数不能为空")
    if text.isdigit() and len(text) >= 12:  # 已经是毫秒时间戳
        return int(text)
    parsed: datetime | None = None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError(f"无法解析时间：{value}（请用 YYYY-MM-DD 或毫秒时间戳）")
    shanghai = parsed.replace(tzinfo=_SHANGHAI_TZ)
    return int(shanghai.timestamp() * 1000)


def fuyao_dataset_params(schema: Mapping[str, Any] | None) -> list[str]:
    """从上游 ``inputSchema`` 里取出参数名；无 schema 时返回空列表。

    参数的唯一事实来源是服务端返回的 schema（派发器在运行时读取），因此这里只做
    一次提取，不维护任何本地副本。
    """
    if not isinstance(schema, Mapping):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return []
    return [str(name) for name in properties]


def fuyao_required_params(schema: Mapping[str, Any] | None) -> list[str]:
    """从上游 ``inputSchema`` 里取出必填参数名。"""
    if not isinstance(schema, Mapping):
        return []
    required = schema.get("required")
    if not isinstance(required, (list, tuple)):
        return []
    return [str(name) for name in required]


def fuyao_param_summary(schema: Mapping[str, Any] | None) -> str:
    """为一个数据集的参数生成一行可读摘要，供目录工具渲染。

    枚举值会一并列出（``period=annual|quarterly``），因为选错枚举是上游最常见的
    参数错误，而模型看不到 schema 时只能猜。
    """
    if not isinstance(schema, Mapping):
        return ""
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return ""
    required = set(fuyao_required_params(schema))
    parts: list[str] = []
    for name, spec in properties.items():
        if not isinstance(spec, Mapping):
            parts.append(str(name))
            continue
        label = str(name) + ("" if name in required else "?")
        enum = spec.get("enum")
        if isinstance(enum, (list, tuple)) and enum:
            label += "=" + "|".join(str(item) for item in enum)
        parts.append(label)
    return ", ".join(parts)


def fuyao_effective_arguments(
    schema: Mapping[str, Any] | None, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """调用方给的参数，加上上游会**代为填入**的 schema 默认值。

    **为什么必须把它算出来。** 上游为若干参数声明了默认值，其中一些是实质性的筛选
    条件而非分页旋钮——``get_a_share_prices_snapshot`` 的 ``thscodes`` 默认是
    ``600519.SH,000001.SZ``，``..._anomaly_analysis_stock`` 默认是两只具体个股。
    调用方省略该参数时，它拿到的**不是全市场**，而是这两只股票，而请求里没有任何
    痕迹留下：一份看起来正常的行情表，配一个错误的隐含筛选。

    因此这里把默认值显式补出来，供结果头部披露。值与"调用方主动指定"在语义上不同，
    所以调用方（工具层）需要能区分二者——本函数返回的键若不在 ``arguments`` 里，
    即为继承自上游的默认。
    """
    if not isinstance(schema, Mapping):
        return dict(arguments)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return dict(arguments)
    effective = dict(arguments)
    for name, spec in properties.items():
        if name in effective or not isinstance(spec, Mapping):
            continue
        if "default" in spec:
            effective[str(name)] = spec["default"]
    return effective


def fuyao_inherited_defaults(
    schema: Mapping[str, Any] | None, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """``fuyao_effective_arguments`` 中调用方**没有**指定的那一部分。

    单独提供一个函数，是因为"哪些条件是上游替我们决定的"才是需要披露的信息；
    把已指定的参数也一并列出只会稀释它。
    """
    effective = fuyao_effective_arguments(schema, arguments)
    return {name: value for name, value in effective.items() if name not in arguments}

