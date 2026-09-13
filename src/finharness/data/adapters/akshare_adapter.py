"""akshare adapter: candidate-chain fallback, column normalization, throttling.

akshare is a synchronous library whose upstream hosts vary in reachability
(the EM ``push2`` hosts were unreachable in this environment while the Sina
hosts worked). Each semantic fetch therefore tries its candidate interfaces in
``mapping.AKSHARE_ENDPOINTS`` order and records which one actually served the
data.
"""

from __future__ import annotations

import threading
import time
from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.mapping import (
    AKSHARE_ENDPOINTS,
    AKSHARE_INTERFACE_COLUMNS,
    PEER_COMPANY,
    PEER_IDENTITY_COLUMNS,
    PEER_ROW_TYPE_COLUMN,
    PEER_STAT,
    PEER_STAT_LABELS,
    VALUATION_INDICATOR_UNITS,
    prefixed_symbol,
    select_indicator_columns,
)

_PERIOD_MAP = {"day": "daily", "week": "weekly", "month": "monthly"}
_PERIOD_LABEL = {1: "近一年", 2: "近一年", 3: "近三年", 5: "近五年"}

# --- Quote candidate budgeting --------------------------------------------
# Snapshot sources are synchronous, paginated and (for EM/Sina) set no request
# timeout, so one unreachable host can consume the whole 30s tool budget and
# cancel the candidates behind it. Each quote candidate is bounded by a
# wall-clock deadline, and an interface that keeps failing is skipped for a
# cooldown instead of being retried on every call.
_QUOTE_CANDIDATE_DEADLINE_S = 8.0
_QUOTE_REQUEST_TIMEOUT_S = 8.0  # passed to interfaces that accept a timeout
_UNHEALTHY_AFTER_FAILURES = 2
_UNHEALTHY_COOLDOWN_S = 300.0


class _DeadlineExceeded(Exception):
    """A candidate did not return within its wall-clock budget."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"未在 {seconds:.1f}s 内返回")
        self.seconds = seconds


def _call_with_deadline(build: Callable[[], Any], *, deadline_s: float) -> Any:
    """Run a blocking callable with a wall-clock bound.

    ``requests`` calls without a timeout cannot be interrupted from Python, so
    the call runs on a daemon thread that is abandoned if it overruns. The
    interface failure memory keeps abandonment rare: a candidate that times out
    is skipped for a cooldown rather than restarted on every query. A daemon
    thread is deliberate — a non-daemon one would block interpreter shutdown.
    """
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = build()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(deadline_s)
    if worker.is_alive():
        raise _DeadlineExceeded(deadline_s)
    if "error" in box:
        raise box["error"]
    return box["value"]


def _import_akshare():
    """Import akshare lazily so the package stays importable without it."""
    import akshare as ak  # noqa: PLC0415 - deliberate lazy import

    return ak


def _normalize(df: pd.DataFrame, interface: str) -> pd.DataFrame:
    rename = AKSHARE_INTERFACE_COLUMNS.get(interface)
    if rename:
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _slice_years(df: pd.DataFrame, years: int, *, date_col: str = "date") -> pd.DataFrame:
    """Keep rows within the last ``years`` years; tolerates a missing date column."""
    if date_col not in df.columns or years <= 0:
        return df
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years))
    trimmed = frame[frame[date_col] >= cutoff]
    return trimmed if len(trimmed) else frame


def _label_peer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Tag the industry-median/average rows so they are never read as a company.

    EM's comparison table returns the aggregates as ordinary rows whose 代码/简称
    hold the literal labels; without a tag a consumer averaging the frame (or
    taking row 0) silently treats an aggregate as an issuer.
    """
    if PEER_ROW_TYPE_COLUMN in df.columns:
        return df
    for column in ("代码", "简称"):
        if column not in df.columns:
            continue
        labels = df[column].astype(str)
        if not labels.isin(PEER_STAT_LABELS).any():
            continue
        frame = df.copy()
        frame.insert(
            0,
            PEER_ROW_TYPE_COLUMN,
            labels.map(lambda value: PEER_STAT if value in PEER_STAT_LABELS else PEER_COMPANY),
        )
        return frame
    return df


class AkShareAdapter(DataAdapter):
    name = "akshare"

    def __init__(self, *, throttle_seconds: float = 1.0) -> None:
        self.throttle_seconds = throttle_seconds
        self._last_call = 0.0
        # Interface health, shared across the process (one adapter serves every
        # session), so a dead interface is not re-probed by each new query.
        self._failures: dict[str, int] = {}
        self._unhealthy_until: dict[str, float] = {}
        self._health_lock = threading.Lock()

    def _throttle(self) -> None:
        if self.throttle_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.throttle_seconds:
            time.sleep(self.throttle_seconds - elapsed)
        self._last_call = time.monotonic()

    # -- interface health -----------------------------------------------------
    def _interface_available(self, interface: str) -> bool:
        """False while an interface is inside its post-failure cooldown."""
        with self._health_lock:
            until = self._unhealthy_until.get(interface)
            if until is None:
                return True
            if time.monotonic() >= until:
                self._unhealthy_until.pop(interface, None)
                self._failures.pop(interface, None)
                return True
            return False

    def _note_success(self, interface: str) -> None:
        with self._health_lock:
            self._failures.pop(interface, None)
            self._unhealthy_until.pop(interface, None)

    def _note_failure(self, interface: str) -> None:
        with self._health_lock:
            count = self._failures.get(interface, 0) + 1
            self._failures[interface] = count
            if count >= _UNHEALTHY_AFTER_FAILURES:
                self._unhealthy_until[interface] = time.monotonic() + _UNHEALTHY_COOLDOWN_S

    def _call(
        self,
        interface: str,
        build: Callable[[Any], pd.DataFrame],
        *,
        deadline_s: float | None = None,
    ) -> pd.DataFrame:
        """Run one candidate interface, converting any failure to AdapterError.

        ``deadline_s`` bounds the network call (not the throttle) so a hanging
        source falls through to the next candidate instead of exhausting the
        caller's budget.
        """
        ak = _import_akshare()
        self._throttle()
        try:
            if deadline_s is None:
                df = build(ak)
            else:
                df = _call_with_deadline(lambda: build(ak), deadline_s=deadline_s)
        except _DeadlineExceeded as exc:
            raise AdapterError(f"{interface}: 超时未返回（>{exc.seconds:.0f}s）", retryable=True) from exc
        except Exception as exc:  # noqa: BLE001 - every source error funnels here
            raise AdapterError(
                f"{interface}: {type(exc).__name__}: {exc}",
                retryable=type(exc).__name__ in {"ConnectionError", "Timeout", "TimeoutError"},
            ) from exc
        if not isinstance(df, pd.DataFrame):
            raise AdapterError(f"{interface}: 返回非表格数据")
        return df

    # -- semantic fetches -----------------------------------------------------
    def fetch_quote(self, symbol: str) -> FetchResult:
        """Latest price, trying the richest snapshot source first.

        Candidates in a post-failure cooldown are skipped; if none is healthy
        the full chain is retried, so a recovered source is picked up again.
        Each attempt is deadline-bounded so a hanging source cannot consume the
        caller's whole budget.
        """
        errors: list[str] = []
        candidates = AKSHARE_ENDPOINTS["quote"]
        healthy = [interface for interface in candidates if self._interface_available(interface)]
        for interface in (healthy or list(candidates)):
            try:
                if interface == "stock_zh_a_spot_em":
                    df = _normalize(
                        self._call(
                            interface,
                            lambda ak: ak.stock_zh_a_spot_em(),
                            deadline_s=_QUOTE_CANDIDATE_DEADLINE_S,
                        ),
                        "quote",
                    )
                    if "symbol" not in df.columns:
                        raise AdapterError(f"{interface}: 缺少代码列")
                    hit = df[df["symbol"].astype(str).str.zfill(6) == symbol]
                elif interface == "stock_zh_a_spot":
                    raw = self._call(
                        interface,
                        lambda ak: ak.stock_zh_a_spot(),
                        deadline_s=_QUOTE_CANDIDATE_DEADLINE_S,
                    )
                    df = raw.rename(
                        columns={"代码": "symbol", "名称": "name", "最新价": "close",
                                 "涨跌幅": "pct_change", "成交量": "volume"}
                    )
                    if "symbol" not in df.columns:
                        raise AdapterError(f"{interface}: 缺少代码列")
                    hit = df[df["symbol"].astype(str).str.zfill(6) == symbol]
                else:
                    # Tencent daily series: the last row is the latest close.
                    end = date.today().strftime("%Y%m%d")
                    start = (date.today() - timedelta(days=14)).strftime("%Y%m%d")
                    df = self._call(
                        interface,
                        lambda ak: ak.stock_zh_a_hist_tx(
                            symbol=prefixed_symbol(symbol, lower=True),
                            start_date=start,
                            end_date=end,
                            timeout=_QUOTE_REQUEST_TIMEOUT_S,
                        ),
                        deadline_s=_QUOTE_CANDIDATE_DEADLINE_S,
                    )
                    if not len(df):
                        raise AdapterError(f"{interface}: 返回空表")
                    latest = df.tail(1).copy()
                    latest["symbol"] = symbol
                    hit = latest
                if not len(hit):
                    raise AdapterError(f"{interface}: 未找到代码 {symbol}")
                self._note_success(interface)
                return FetchResult(df=hit.reset_index(drop=True), interface=interface)
            except AdapterError as exc:
                self._note_failure(interface)
                errors.append(exc.message)
        raise AdapterError("; ".join(errors) or "quote 无可用接口")

    def fetch_kline(self, symbol: str, period: str, adjust: str | None, years: int) -> pd.DataFrame:
        em_period = _PERIOD_MAP.get(period, "daily")
        errors: list[str] = []
        for interface in AKSHARE_ENDPOINTS["kline"]:
            try:
                if interface == "stock_zh_a_hist":
                    start = (date.today() - timedelta(days=365 * max(years, 1) + 30)).strftime("%Y%m%d")
                    end = date.today().strftime("%Y%m%d")
                    df = self._call(
                        interface,
                        lambda ak: ak.stock_zh_a_hist(
                            symbol=symbol, period=em_period, start_date=start,
                            end_date=end, adjust=adjust or "",
                        ),
                    )
                else:  # sina / tencent variants need an exchange prefix
                    prefixed = prefixed_symbol(symbol, lower=True)
                    if interface == "stock_zh_a_daily":
                        df = self._call(
                            interface,
                            lambda ak: ak.stock_zh_a_daily(symbol=prefixed, adjust=adjust or ""),
                        )
                    else:
                        start = (date.today() - timedelta(days=365 * max(years, 1) + 30)).strftime("%Y%m%d")
                        end = date.today().strftime("%Y%m%d")
                        df = self._call(
                            interface,
                            lambda ak: ak.stock_zh_a_hist_tx(
                                symbol=prefixed, start_date=start, end_date=end
                            ),
                        )
                df = _normalize(df, "kline")
                if not len(df):
                    raise AdapterError(f"{interface}: 返回空表")
                # Sources differ in row order (EM ascends, Sina ascends); the
                # internal contract is newest-first so summaries and MA windows
                # always read the latest period.
                df = _slice_years(df, years)
                if "date" in df.columns:
                    df = df.sort_values("date", ascending=False)
                return FetchResult(df=df.reset_index(drop=True), interface=interface)
            except AdapterError as exc:
                errors.append(exc.message)
        raise AdapterError("; ".join(errors) or "kline 无可用接口")

    def fetch_indicators(self, symbol: str, years: int, fields: list[str] | None) -> FetchResult:
        interface = AKSHARE_ENDPOINTS["indicators"][0]
        start_year = str(date.today().year - max(years, 1))
        df = self._call(
            interface,
            lambda ak: ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year),
        )
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "date"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", ascending=False)
        if fields:
            keep, _unmatched = select_indicator_columns(df.columns, fields)
            df = df[keep]
        return FetchResult(df=df.reset_index(drop=True), interface=interface)

    def fetch_financials(self, symbol: str, statement: str, years: int) -> FetchResult:
        interface = AKSHARE_ENDPOINTS["financials"][0]
        df = self._call(interface, lambda ak: ak.stock_financial_abstract(symbol=symbol))
        # Period columns are YYYYMMDD; keep the most recent N years of them.
        period_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
        keep_periods = sorted(period_cols, reverse=True)[: max(years, 1) * 4]
        base = [c for c in df.columns if c not in period_cols]
        return FetchResult(
            df=df[base + sorted(keep_periods, reverse=True)].reset_index(drop=True),
            interface=interface,
        )

    def fetch_valuation(self, symbol: str, lookback_years: int, indicator: str) -> FetchResult:
        """One valuation series for one indicator.

        Baidu's endpoint switches metric on ``indicator`` but returns a bare
        ``date``/``value`` frame that names neither the metric nor its unit, so
        both are attached here; a market-cap figure rendered as an unlabelled
        ``value`` is how it previously read as a PE multiple.
        """
        interface = AKSHARE_ENDPOINTS["valuation"][0]
        period = _PERIOD_LABEL.get(lookback_years, "近一年")
        df = self._call(
            interface,
            lambda ak: ak.stock_zh_valuation_baidu(
                symbol=symbol, indicator=indicator, period=period
            ),
        )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date", ascending=False)
        unit = VALUATION_INDICATOR_UNITS.get(indicator, "")
        df = df.rename(columns={"value": f"{indicator}({unit})" if unit else indicator})
        return FetchResult(df=df.reset_index(drop=True), interface=interface)

    def fetch_peers(self, industry: str, fields: list[str] | None) -> FetchResult:
        """``industry`` carries the target symbol for peer comparison."""
        interface = AKSHARE_ENDPOINTS["peers"][0]
        target = prefixed_symbol(industry)
        df = self._call(interface, lambda ak: ak.stock_zh_valuation_comparison_em(symbol=target))
        df = _label_peer_rows(df)
        if fields:
            matched = [c for c in df.columns if any(f in str(c) for f in fields)]
            # Identity columns and the row tag are contract, not filter material:
            # a filtered table that cannot say whose row is whose is unusable.
            keep = [
                c for c in (PEER_ROW_TYPE_COLUMN, *PEER_IDENTITY_COLUMNS) if c in df.columns
            ]
            keep += [c for c in matched if c not in keep]
            df = df[keep]
        # Companies first, aggregates last: a whole-frame mean is then visibly
        # wrong rather than silently including the industry statistics.
        if PEER_ROW_TYPE_COLUMN in df.columns:
            rank = {PEER_COMPANY: 0, PEER_STAT: 1}
            df = df.sort_values(
                PEER_ROW_TYPE_COLUMN, key=lambda col: col.map(rank), kind="stable"
            )
        return FetchResult(df=df.reset_index(drop=True), interface=interface)

    def fetch_news(self, symbol: str | None, topic: str | None, top_n: int) -> FetchResult:
        interface = AKSHARE_ENDPOINTS["news"][0]
        query = symbol or topic or "A股"
        df = self._call(interface, lambda ak: ak.stock_news_em(symbol=query))
        return FetchResult(df=df.head(top_n).reset_index(drop=True), interface=interface)

    def fetch_announcements(self, symbol: str, since: str, top_n: int) -> FetchResult:
        interface = AKSHARE_ENDPOINTS["announcements"][0]
        begin = since.replace("-", "")
        end = date.today().strftime("%Y%m%d")
        df = self._call(
            interface,
            lambda ak: ak.stock_individual_notice_report(
                security=symbol, symbol="全部", begin_date=begin, end_date=end
            ),
        )
        return FetchResult(df=df.head(top_n).reset_index(drop=True), interface=interface)
