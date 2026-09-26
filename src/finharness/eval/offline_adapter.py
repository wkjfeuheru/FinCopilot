"""用于 ``--offline`` 自检的确定性离线数据源（docs 03.13）。

提供合成的 A 股数据帧，使工具无需网络或凭证即可端到端执行。数值是固定的、
并非真实值：重点在于工具调用能够成功并产出引用，这正是评测工具自检所验证的。
"""

from __future__ import annotations

import pandas as pd

from finharness.data.adapters.base import DataAdapter, FetchResult


class OfflineAdapter(DataAdapter):
    """一个适配器替身，对每次语义化取数都以一个小数据帧作答。"""

    name = "offline"

    # 排行夹具的数据日期：与其它离线夹具保持同一“当下”基准。
    RANKING_DATE = "2026-09-14"

    def _quote_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "代码": "600519",
                    "名称": "贵州茅台",
                    "最新价": 1500.0,
                    "涨跌幅": 1.23,
                    "成交量": 123456,
                    "成交额": 1.85e9,
                    "日期": "2026-09-14",
                }
            ]
        )

    def fetch_quote(self, symbol: str) -> FetchResult:
        return FetchResult(df=self._quote_frame(), interface="offline:quote")

    def fetch_kline(
        self, symbol: str, period: str, adjust: str | None, years: int
    ) -> FetchResult:
        dates = pd.date_range(end="2026-09-14", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "open": [1500 + i for i in range(30)],
                "high": [1510 + i for i in range(30)],
                "low": [1490 + i for i in range(30)],
                "close": [1505 + i for i in range(30)],
                "volume": [100000 + i * 10 for i in range(30)],
            }
        )
        return FetchResult(df=df, interface=f"offline:kline:{period}")

    def fetch_indicators(
        self, symbol: str, years: int, fields: list[str] | None
    ) -> FetchResult:
        df = pd.DataFrame(
            {
                "报告期": ["2023-12-31", "2024-12-31", "2025-12-31"],
                "ROE": [0.30, 0.31, 0.32],
                "毛利率": [0.91, 0.92, 0.92],
            }
        )
        return FetchResult(df=df, interface="offline:indicators")

    def fetch_financials(self, symbol: str, statement: str, years: int) -> FetchResult:
        df = pd.DataFrame(
            {
                "报告期": ["2023-12-31", "2024-12-31", "2025-12-31"],
                "营业总收入": [1.47e11, 1.60e11, 1.72e11],
                "归母净利润": [7.4e10, 8.0e10, 8.6e10],
            }
        )
        return FetchResult(df=df, interface=f"offline:financials:{statement}")

    def fetch_valuation(
        self, symbol: str, lookback_years: int, indicator: str
    ) -> FetchResult:
        df = pd.DataFrame(
            {
                "date": ["2026-09-14"],
                "pe": [19.6],
                "pb": [7.1],
            }
        )
        return FetchResult(df=df, interface="offline:valuation")

    def fetch_peers(self, industry: str, fields: list[str] | None) -> FetchResult:
        df = pd.DataFrame(
            {
                "代码": ["600519", "000858"],
                "名称": ["贵州茅台", "五粮液"],
                "pe": [19.6, 15.2],
                "pb": [7.1, 3.9],
            }
        )
        return FetchResult(df=df, interface="offline:peers")

    def fetch_news(
        self, symbol: str | None, topic: str | None, top_n: int
    ) -> FetchResult:
        df = pd.DataFrame(
            [
                {
                    "标题": "离线自检新闻",
                    "时间": "2026-09-14",
                    "来源": "offline",
                    "内容": "用于自检的占位新闻。",
                }
            ]
        )
        return FetchResult(df=df, interface="offline:news")

    def fetch_announcements(self, symbol: str, since: str, top_n: int) -> FetchResult:
        df = pd.DataFrame(
            [{"公告标题": "离线自检公告", "公告日期": "2026-09-14", "类型": "定期报告"}]
        )
        return FetchResult(df=df, interface="offline:announcements")

    def fetch_macro(self, indicators: list[str], years: int) -> FetchResult:
        df = pd.DataFrame(
            {
                "指标": indicators or ["PMI"],
                "期间": ["2026-08"],
                "数值": [50.2],
                "机构": ["国家统计局"],
            }
        )
        return FetchResult(df=df, interface="offline:macro")

    def fetch_industry_perf(self, industry: str | None, years: int) -> FetchResult:
        df = pd.DataFrame(
            [{"行业": industry or "申万一级", "涨跌幅": 1.1, "PE": 18.0}]
        )
        return FetchResult(df=df, interface="offline:industry_perf")

    def fetch_industry_ranking(self, period: str = "day", as_of: str | None = None) -> FetchResult:
        """离线排行：已按涨跌幅降序，前五名固定，供评测做确定性断言。"""
        rows = [
            ("801950", "煤炭", 0.63),
            ("801130", "纺织服饰", 0.35),
            ("801780", "银行", 0.31),
            ("801110", "家用电器", 0.01),
            ("801160", "公用事业", -0.11),
            ("801960", "石油石化", -0.15),
            ("801980", "美容护理", -0.28),
            ("801170", "交通运输", -0.43),
            ("801140", "轻工制造", -0.82),
            ("801230", "综合", -0.98),
            ("801040", "钢铁", -1.01),
            ("801210", "社会服务", -1.04),
        ]
        df = pd.DataFrame(
            [
                {
                    "code": code,
                    "industry": name,
                    "date": self.RANKING_DATE,
                    "close": round(3000 + pct * 100, 2),
                    "pct_change": pct,
                }
                for code, name, pct in rows
            ]
        )
        return FetchResult(df=df, interface=f"offline:industry_ranking:{period}")

    def fetch_industry_constituents(self, industry: str) -> FetchResult:
        df = pd.DataFrame([{"代码": "600519", "名称": "贵州茅台"}])
        return FetchResult(df=df, interface="offline:industry_constituents")

    def fetch_index_constituents(self, index: str) -> FetchResult:
        df = pd.DataFrame([{"代码": "600519", "名称": "贵州茅台"}])
        return FetchResult(df=df, interface="offline:index_constituents")

    def fetch_web_search(
        self, query: str, top_n: int, topic: str | None, time_range: str | None
    ) -> FetchResult:
        df = pd.DataFrame(
            [{"title": "离线自检网页", "url": "https://example.invalid", "content": "占位内容"}]
        )
        return FetchResult(df=df, interface="offline:web")

    def fetch_research_reports(
        self,
        report_type: str,
        industry: str | None,
        institution: str | None,
        keyword: str | None,
        start_date: str | None,
        end_date: str | None,
        top_n: int,
        with_text: bool,
    ) -> FetchResult:
        df = pd.DataFrame([{"标题": "离线自检研报", "机构": institution or "offline"}])
        return FetchResult(df=df, interface="offline:reports")


__all__ = ["OfflineAdapter"]
