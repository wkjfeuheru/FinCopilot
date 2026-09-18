"""数据源接口名、列映射与交易所前缀的唯一事实来源。

新增一个数据接口时，只应扩展此处的映射表，绝不能把
来源特有的知识散落到各个适配器或工具中。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

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
