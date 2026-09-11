"""AkShare synchronous adapter; DataAccess executes it off the event loop."""

import akshare as ak

from finharness.data.adapters.base import DataAdapter


class AkShareAdapter(DataAdapter):
    name = "akshare"

    def fetch_quote(self, symbol: str):
        return ak.stock_zh_a_spot_em().query("代码 == @symbol")

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int):
        period_map = {"day": "daily", "week": "weekly", "month": "monthly"}
        return ak.stock_zh_a_hist(symbol=symbol, period=period_map.get(period, "daily"), adjust=adjust or "")

    def fetch_indicators(self, symbol: str, years: int, fields: list[str] | None):
        data = ak.stock_financial_analysis_indicator(symbol=symbol)
        if fields:
            columns = [column for column in data.columns if column in fields or column == "报告期"]
            return data[columns]
        return data.head(years)
