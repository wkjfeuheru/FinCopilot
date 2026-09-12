"""tushare adapter: optional fallback source, imported and enabled lazily.

tushare is not a hard dependency and needs a paid token, so this adapter is
import-safe without either. Construction succeeds; each fetch raises a clear
``AdapterError`` when the library or token is unavailable, which the
orchestrator treats as an ordinary fallback reason.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult


def _import_tushare():
    try:
        import tushare as ts  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:
        raise AdapterError("tushare 未安装，无法作为降级数据源") from exc
    return ts


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "trade_date": "date", "vol": "volume",
            "ts_code": "symbol", "close": "close", "open": "open",
            "high": "high", "low": "low", "amount": "amount",
        }
    )


class TushareAdapter(DataAdapter):
    name = "tushare"

    def __init__(self, *, token_env: str = "TUSHARE_TOKEN") -> None:
        self.token_env = token_env

    def _pro(self):
        ts = _import_tushare()
        token = os.getenv(self.token_env)
        if not token:
            raise AdapterError(f"tushare 缺少 token（环境变量 {self.token_env} 未设置）")
        try:
            return ts.pro_api(token)
        except Exception as exc:  # noqa: BLE001 - token/init failures are source errors
            raise AdapterError(f"tushare 初始化失败：{exc}") from exc

    @staticmethod
    def _ts_code(symbol: str) -> str:
        from finharness.data.mapping import exchange_prefix

        return f"{symbol}.{exchange_prefix(symbol)}"

    def fetch_quote(self, symbol: str) -> FetchResult:
        pro = self._pro()
        try:
            df = pro.daily_basic(ts_code=self._ts_code(symbol), limit=1)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"daily_basic: {exc}") from exc
        if df is None or not len(df):
            raise AdapterError("daily_basic: 空结果")
        return FetchResult(df=_normalize(df).reset_index(drop=True), interface="daily_basic")

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int) -> FetchResult:
        pro = self._pro()
        start = (date.today() - timedelta(days=365 * max(years, 1) + 30)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        try:
            df = pro.daily(ts_code=self._ts_code(symbol), start_date=start, end_date=end)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"daily: {exc}") from exc
        if df is None or not len(df):
            raise AdapterError("daily: 空结果")
        return FetchResult(df=_normalize(df).sort_values("date").reset_index(drop=True), interface="daily")

    def fetch_indicators(self, symbol: str, years: int, fields: list[str] | None) -> FetchResult:
        pro = self._pro()
        start = (date.today() - timedelta(days=365 * max(years, 1) + 30)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        try:
            df = pro.fina_indicator(ts_code=self._ts_code(symbol), start_date=start, end_date=end)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"fina_indicator: {exc}") from exc
        if df is None or not len(df):
            raise AdapterError("fina_indicator: 空结果")
        return FetchResult(df=df.reset_index(drop=True), interface="fina_indicator")

    def fetch_financials(self, symbol: str, statement: str, years: int) -> FetchResult:
        pro = self._pro()
        start = (date.today() - timedelta(days=365 * max(years, 1) + 30)).strftime("%Y%m%d")
        end = date.today().strftime("%Y%m%d")
        try:
            df = pro.income(ts_code=self._ts_code(symbol), start_date=start, end_date=end)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"income: {exc}") from exc
        if df is None or not len(df):
            raise AdapterError("income: 空结果")
        return FetchResult(df=df.reset_index(drop=True), interface="income")
