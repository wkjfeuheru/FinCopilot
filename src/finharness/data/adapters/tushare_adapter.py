"""tushare 适配器：可选的降级数据源，惰性导入并按需启用。

tushare 不是硬依赖，且需要付费 token，因此本适配器在缺少两者时都能安全导入。
构造可以成功；当库或 token 不可用时，每次抓取都会抛出明确的 ``AdapterError``，
编排器会把它当作普通的回退原因。
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult


def _import_tushare():
    """惰性导入 tushare；未安装时转换为 AdapterError。"""
    try:
        import tushare as ts  # noqa: PLC0415 - 刻意延迟导入
    except ImportError as exc:
        raise AdapterError("tushare 未安装，无法作为降级数据源") from exc
    return ts


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """把 tushare 的列名统一为本项目内部契约的列名。"""
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
        """读取 token 并初始化 tushare pro 客户端，缺失或失败时抛出 AdapterError。"""
        ts = _import_tushare()
        token = os.getenv(self.token_env)
        if not token:
            raise AdapterError(f"tushare 缺少 token（环境变量 {self.token_env} 未设置）")
        try:
            return ts.pro_api(token)
        except Exception as exc:  # noqa: BLE001 - token/初始化失败属于数据源错误
            raise AdapterError(f"tushare 初始化失败：{exc}") from exc

    @staticmethod
    def _ts_code(symbol: str) -> str:
        """把六位证券代码转换为 tushare 的 ``代码.交易所后缀`` 形式。"""
        from finharness.data.mapping import exchange_prefix

        return f"{symbol}.{exchange_prefix(symbol)}"

    def fetch_quote(self, symbol: str) -> FetchResult:
        """经 daily_basic 接口获取最新行情快照。"""
        pro = self._pro()
        try:
            df = pro.daily_basic(ts_code=self._ts_code(symbol), limit=1)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"daily_basic: {exc}") from exc
        if df is None or not len(df):
            raise AdapterError("daily_basic: 空结果")
        return FetchResult(df=_normalize(df).reset_index(drop=True), interface="daily_basic")

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int) -> FetchResult:
        """经 daily 接口获取日线，统一列名并按日期升序排列。"""
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
        """经 fina_indicator 接口获取财务指标。"""
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
        """经 income 接口获取利润表数据。"""
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
