"""run_backtest：参数化的单序列与横截面回测。

策略/因子以*声明*方式给出，绝不以代码方式执行：单股时序回测接受内建的
``ma``/``momentum`` 或因子表达式，横截面回测接受由
:class:`~finharness.factor.engine.FactorEngine` 求值的因子表达式。这正是把
"让 agent 自行发现因子"约束在禁止任意代码执行边界内的关键。

纪律约束（docs 03.8，量化因子场景）在此强制执行，而非停留在文字约定：样本
按时间顺序切分，买入持有基准始终参与运行，成本/滑点按换手率计提，结果明确
给出是否跑赢基准，而不是掩盖失败。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finharness.data.frames import persist_frame
from finharness.data.mapping import normalize_valuation_indicator
from finharness.data.raw import RawData
from finharness.factor.engine import MARKET_VARIABLES, FactorEngine
from finharness.shared.declaration import Capability, Tier, ToolGroup, param, tool
from finharness.tools.base import BaseTool

TRADING_DAYS = 252
MAX_POOL_SIZE = 300
# 面板取数的进展上报间隔（只）。太小会淹没事件流，太大则静默期过长——
# 1.0s 全局节流下 10 只约 10 秒，短于客户端空闲看门狗，足以持续喂给它真实事件。
_PROGRESS_EVERY = 10
DEFAULT_RF_RATE = 0.015
# 试验预算：该场景的过拟合防线。超出即让本次调用失败，
# 而不是默默放任数据窥探（fishing expedition）。
MAX_TRIALS = 50


_PARAMS_HELP = (
    "策略参数：ma→{fast,slow}；momentum→{window,holding}；"
    "横截面→{groups,rebalance,max_symbols}"
)




@dataclass(slots=True)
class Perf:
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    trades: int = 0

    def render(self, label: str) -> list[str]:
        return [
            f"**{label}**：累计 {self.total_return:.2%} · 年化 {self.annual_return:.2%} · "
            f"夏普 {self.sharpe:.2f} · 最大回撤 {self.max_drawdown:.2%} · 卡玛 {self.calmar:.2f} · "
            f"胜率 {self.win_rate:.1%} · 盈亏比 {self.profit_loss_ratio:.2f} · 交易 {self.trades} 次"
        ]


def _performance(returns: pd.Series, *, rf_rate: float) -> Perf:
    """由日度净收益序列计算年化绩效指标。

    参数：
        returns：日度净收益序列（已扣除成本，小数口径，如 0.01 表示 1%）。
        rf_rate：年化无风险利率（用于 Sharpe 口径）。

    返回 Perf：累计收益、年化收益、夏普、最大回撤、卡玛、胜率、盈亏比与
    非零持仓交易次数；夏普按 252 个交易日年化，回撤取正值。
    """
    clean = returns.dropna()
    if not len(clean):
        return Perf()
    equity = (1 + clean).cumprod()
    total = float(equity.iloc[-1]) - 1
    years = len(clean) / TRADING_DAYS
    annual = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else 0.0
    std = float(clean.std())
    sharpe = ((float(clean.mean()) * TRADING_DAYS) - rf_rate) / (std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    peak = equity.cummax()
    drawdown = (equity / peak - 1).min()
    max_dd = abs(float(drawdown)) if pd.notna(drawdown) else 0.0
    calmar = annual / max_dd if max_dd > 0 else 0.0
    active = clean[clean != 0]
    wins = active[active > 0]
    losses = active[active < 0]
    win_rate = float(len(wins) / len(active)) if len(active) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = abs(float(losses.mean())) if len(losses) else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    return Perf(
        total_return=total, annual_return=annual, sharpe=sharpe,
        max_drawdown=max_dd, calmar=calmar, win_rate=win_rate,
        profit_loss_ratio=pl_ratio, trades=int(len(active)),
    )


def _max_drawdown(equity: pd.Series) -> float:
    """由净值序列计算最大回撤，返回正值（如 0.2 表示最大回撤 20%）。"""
    peak = equity.cummax()
    return abs(float((equity / peak - 1).min()))


@tool(
    name="run_backtest",
    description=(
        "运行回测并输出绩效（年化/夏普/最大回撤/卡玛/胜率/盈亏比）、样本内外分段与买入持有基准；"
        "支持单股时序策略/因子与横截面分组/IC 检验。策略或因子以参数声明，不执行任意代码。"
    ),
    capability=Capability.COMPUTE,
    # 重量级计算（面板取数、IC/分组回测）：由量化因子场景按需触发，而非随每次请求携带。
    tier=Tier.LAZY,
    group=ToolGroup.FIN_CALC,
    # 横截面面板要逐只取数，且受 data.throttle_seconds 全局节流约束：300 只成分股
    # 冷启动就可能超过 5 分钟。300s 的旧预算在这种规模下必然超时（面板取数本身
    # 未产生任何结果就已被取消），因此按最坏规模给足余量。
    timeout=900,
    output_schema_note="返回绩效表（样本外在前）、样本内外对比、基准对比与过拟合检查字段。",
)
class RunBacktestTool(BaseTool):
    # 横截面面板要逐只取数，冷启动可达数分钟；不上报进展会让客户端在静默期
    # 判定卡死而掐断连接（docs 03.12）。
    needs_progress = True

    @param("symbol", desc="单股序列模式的 6 位 A 股代码；与 pool 二选一")
    @param(
        "pool",
        desc="横截面模式的股票池：指数名/代码（沪深300/中证500/中证1000）、申万行业名，或显式 6 位代码列表",
    )
    @param(
        "factor_expr",
        desc=(
            "因子表达式（AST 白名单求值）。时序（单股）如 ts_mean(close,20)/ts_mean(close,60)-1；"
            "横截面必须含 rank/zscore/quantile，行情变量用 close 等，估值因子如 rank(-pe_ttm)"
            "（另支持 pe/pb/pcf/总市值）。未知变量会在取价前拒绝，不要原样重试。"
        ),
    )
    @param("strategy", desc="内建策略（单股）：ma 双均线 / momentum 动量")
    @param("params", desc=_PARAMS_HELP)
    @param("years", desc="回测回溯年数")
    @param("is_ratio", desc="样本内占比（按时间顺序切分，严禁随机切分）")
    @param("cost_bps", desc="单边交易成本（基点）")
    @param("slippage_bps", desc="单边滑点（基点）")
    @param("rf_rate", desc="年化无风险利率（夏普口径）")
    @param("trials", desc="本次会话累计候选因子试验次数（上限 50）")
    async def _dispatch(
        self,
        *,
        symbol: str | None = None,
        pool: list[str] | str | None = None,
        factor_expr: str | None = None,
        strategy: str | None = None,
        params: dict[str, Any] | None = None,
        years: int = 2,
        is_ratio: float = 0.7,
        cost_bps: float = 0.0,
        slippage_bps: float = 0.0,
        rf_rate: float = DEFAULT_RF_RATE,
        trials: int = 1,
    ) -> RawData:
        """回测入口：校验参数后按 symbol/pool 分派到单股或横截面回测。

        参数含成本/滑点（基点）、样本内占比、年化无风险利率与试验次数；
        返回 RawData；试验次数超上限或参数非法时抛出 ValueError。
        """
        params = dict(params or {})
        if trials > MAX_TRIALS:
            raise ValueError(
                f"候选因子试验次数 {trials} 超过上限 {MAX_TRIALS}；"
                "请减少候选或披露已试验数量，避免数据窥探。"
            )
        if not (0.1 <= is_ratio <= 0.9):
            raise ValueError("is_ratio 须在 0.1~0.9 之间")
        if symbol and pool:
            raise ValueError("symbol 与 pool 二选一")
        if not symbol and not pool:
            raise ValueError("需要 symbol（单股）或 pool（横截面）之一")
        if not factor_expr and not strategy:
            raise ValueError("需要 factor_expr 或 strategy 之一")

        if symbol:
            return await self._single_series(
                symbol=symbol, factor_expr=factor_expr, strategy=strategy,
                params=params, years=years, is_ratio=is_ratio,
                cost_bps=cost_bps, slippage_bps=slippage_bps, rf_rate=rf_rate,
                trials=trials,
            )
        return await self._cross_section(
            pool=pool, factor_expr=factor_expr, params=params, years=years,
            is_ratio=is_ratio, cost_bps=cost_bps, slippage_bps=slippage_bps,
            rf_rate=rf_rate, trials=trials,
        )

    # -- 单股时序 --------------------------------------------------------------
    async def _single_series(
        self, *, symbol: str, factor_expr: str | None, strategy: str | None,
        params: dict[str, Any], years: int, is_ratio: float, cost_bps: float,
        slippage_bps: float, rf_rate: float, trials: int,
    ) -> RawData:
        """运行单股时序回测：构造仓位、计提成本，并输出绩效与基准对比。

        策略/因子以参数声明；返回的 RawData 文本含样本内外分段、买入持有基准
        与过拟合检查，同时把净值序列（nav 与 benchmark_nav）持久化供图表引用。
        """
        raw = await self.data.kline(symbol, years=years)
        df = raw.df
        if df is None or not len(df):
            raise ValueError("未取到 K 线数据，无法回测")
        frame = self._market_frame(df)
        if len(frame) < 60:
            raise ValueError(f"有效样本仅 {len(frame)} 个交易日，不足以回测")

        variables = self._variable_frames(frame)
        if factor_expr:
            factor = FactorEngine().evaluate(factor_expr, variables).iloc[:, 0]
            position = self._signal_from_factor(factor, params)
            detail = f"因子表达式：`{factor_expr}`"
        elif strategy == "ma":
            fast = int(params.get("fast", 20))
            slow = int(params.get("slow", 60))
            if fast >= slow:
                raise ValueError("ma 策略要求 fast < slow")
            fast_ma = frame["close"].rolling(fast).mean()
            slow_ma = frame["close"].rolling(slow).mean()
            position = (fast_ma > slow_ma).astype(float)
            factor = fast_ma / slow_ma - 1
            detail = f"双均线策略：MA{fast} 上穿 MA{slow} 持有"
        elif strategy == "momentum":
            window = int(params.get("window", 20))
            holding = int(params.get("holding", 1))
            momentum = frame["close"] / frame["close"].shift(window) - 1
            raw_signal = (momentum > 0).astype(float)
            position = raw_signal.rolling(holding).max()
            factor = momentum
            detail = f"动量策略：过去 {window} 日收益为正则持有 {holding} 日"
        else:
            raise ValueError(f"未知策略：{strategy}；可选 ma/momentum")

        asset_returns = frame["close"].pct_change()
        net, benchmark, turnover = self._apply_position(
            position, asset_returns, cost_bps=cost_bps, slippage_bps=slippage_bps
        )
        split = int(len(net) * is_ratio)
        oos, is_ = net.iloc[split:], net.iloc[:split]
        perf_oos = _performance(oos, rf_rate=rf_rate)
        perf_is = _performance(is_, rf_rate=rf_rate)
        perf_all = _performance(net, rf_rate=rf_rate)
        bench = _performance(benchmark, rf_rate=rf_rate)
        positives = int(len(frame) - factor.isna().sum())

        lines = [f"# 回测结果：{symbol}", "", detail, ""]
        lines.append(f"区间：{frame.index.min().date()} ~ {frame.index.max().date()}（{len(frame)} 个交易日）")
        lines.append(
            f"假设：样本内占比 {is_ratio:.0%}（按时间切分）· 交易成本 {cost_bps:.0f}bp · "
            f"滑点 {slippage_bps:.0f}bp · 无风险利率 {rf_rate:.2%} · 复权口径以取数为准"
        )
        lines.append("")
        lines.append("## 绩效（样本外在前）")
        lines.append("")
        lines.extend(perf_oos.render("样本外"))
        lines.extend(perf_is.render("样本内"))
        lines.extend(perf_all.render("全区间"))
        lines.append("")
        lines.append("## 基准对比（买入持有）")
        lines.append("")
        lines.extend(bench.render("买入持有"))
        excess = perf_oos.annual_return - bench.annual_return
        lines.append(f"- 样本外相对基准年化超额：{excess:+.2%}")
        verdict = (
            "样本外跑赢买入持有" if excess > 0 else "**样本外未跑赢买入持有，策略无效**"
        )
        lines.append(f"- 结论：{verdict}")
        lines.append("")
        lines.append("## 过拟合检查")
        lines.append("")
        lines.append(f"- 样本内外差异：样本外年化 {perf_oos.annual_return:.2%} vs 样本内 {perf_is.annual_return:.2%}")
        lines.append(f"- 有效信号样本：{positives} 日；非零持仓交易 {perf_all.trades} 次" + ("（<30，统计意义弱）" if perf_all.trades < 30 else ""))
        lines.append(f"- 试验披露：本次共试验 {trials} 个候选（上限 {MAX_TRIALS}）")
        lines.append("")
        lines.append("> 结论限定于上述区间；不外推未来，不构成投资建议。")

        equity = pd.DataFrame(
            {
                "date": frame.index,
                "nav": (1 + net.fillna(0)).cumprod().to_numpy(),
                "benchmark_nav": (1 + benchmark.fillna(0)).cumprod().to_numpy(),
            }
        )
        # 持久化，使本会话后续的 make_chart 调用可通过引用绘制这些序列，
        # 而无需重新推导（更糟的是直接绘制原始价格）。
        parquet_path = persist_frame(
            equity, cache_dir=self._cache_dir(), name=f"backtest_{symbol}"
        )
        return RawData(
            kind="text", text="\n".join(lines), endpoint="backtest:single",
            params={"symbol": symbol, "factor_expr": factor_expr, "strategy": strategy, "trials": trials},
            df=equity,
            parquet_path=parquet_path,
        )

    # -- 横截面 ----------------------------------------------------------------
    async def _cross_section(
        self, *, pool: list[str] | str, factor_expr: str | None, params: dict[str, Any],
        years: int, is_ratio: float, cost_bps: float, slippage_bps: float,
        rf_rate: float, trials: int,
    ) -> RawData:
        """运行横截面因子回测：分组、计算 IC 与分组/多空组合净值。

        要求因子表达式包含横截面算子；返回文本含 IC 分析、分组年化收益与
        单调性判读、多空组合绩效及过拟合与生存者偏差提示；净值序列持久化
        供图表引用。
        """
        if not factor_expr:
            raise ValueError("横截面回测需要 factor_expr（因子表达式）")
        engine = FactorEngine()
        info = engine.describe(factor_expr)
        if not info.cross_sectional:
            raise ValueError("横截面回测的因子应包含 rank/zscore/quantile 等横截面算子")
        # 未知变量在取股票池和行情之前拒绝：否则会先花掉整段面板取数，
        # 再以「未知变量」失败，模型往往会原样重试直到循环防护掐断本轮。
        market_columns, valuation_vars = self._classify_factor_variables(info.variables)

        symbols = await self._resolve_pool(pool)
        resolved = len(symbols)
        limit = int(params.get("max_symbols", MAX_POOL_SIZE))
        capped = resolved > limit
        if capped:
            symbols = symbols[:limit]
        groups = max(int(params.get("groups", 5)), 2)
        rebalance = max(int(params.get("rebalance", 5)), 1)

        # 池一旦确定就上报规模与是否截断：这是用户最快能得到的反馈，也避免
        # "检验了中证500"实为前 300 只却无从察觉的误导。
        await self._report_progress(
            {
                "phase": "pool",
                "resolved": resolved,
                "capped": capped,
                "total": len(symbols),
                "groups": groups,
                "rebalance": rebalance,
                "years": years,
            }
        )

        panels = await self._panel(symbols, years=years, columns=market_columns)
        closes = panels.get("close")
        if closes is None or closes.shape[1] < groups:
            raise ValueError(f"有效股票数不足（{0 if closes is None else closes.shape[1]}），无法分组")
        missing = [name for name in market_columns if name not in panels]
        if missing:
            raise ValueError(f"数据中缺少变量：{'、'.join(missing)}")

        variables = dict(panels)
        if valuation_vars:
            variables.update(
                await self._valuation_panels(
                    list(closes.columns),
                    years=years,
                    index=closes.index,
                    variables=valuation_vars,
                )
            )
        factor = engine.evaluate(factor_expr, variables)
        returns = closes.pct_change()
        # 调仓窗口内的前向收益：t 时刻的信号预测 t+1..t+h。
        forward = closes.shift(-rebalance) / closes - 1

        lines = ["# 横截面因子回测", "", f"因子表达式：`{factor_expr}`", ""]
        lines.append(f"股票池：{len(closes.columns)} 只 · 区间 {closes.index.min().date()} ~ {closes.index.max().date()}")
        lines.append(f"分组：{groups} 组 · 调仓周期：{rebalance} 日 · 成本 {cost_bps:.0f}bp + 滑点 {slippage_bps:.0f}bp")
        # 截断必须写在结果里：否则 500 只的指数被静默取前 300 只，用户会把
        # 结论误当作对完整成分股池的检验。
        if capped:
            lines.append(
                f"- **股票池截断**：解析到 {resolved} 只，受上限 {limit} 限制仅纳入前 "
                f"{len(closes.columns)} 只有效成分；可用 `params.max_symbols` 调整上限。"
            )
        lines.append("")

        # IC：因子行与前向收益之间的 Spearman 秩相关。
        ic = factor.corrwith(forward, axis=1, method="spearman").dropna()
        split = int(len(ic) * is_ratio)
        ic_is, ic_oos = ic.iloc[:split], ic.iloc[split:]
        lines.append("## IC 分析（样本外在前）")
        lines.append("")
        lines.extend(self._ic_lines("样本外", ic_oos))
        lines.extend(self._ic_lines("样本内", ic_is))
        lines.extend(self._ic_lines("全区间", ic))
        lines.append("")

        # 分组净值：每 `rebalance` 日调仓一次，等权。
        group_returns = self._group_returns(factor, forward, returns, groups=groups, rebalance=rebalance)
        cost = (cost_bps + slippage_bps) / 10000
        lines.append("## 分组净值（年化收益，单调性判读）")
        lines.append("")
        annualized: list[float] = []
        for idx in range(groups):
            series = group_returns[idx].fillna(0) - cost
            perf = _performance(series, rf_rate=rf_rate)
            annualized.append(perf.annual_return)
            lines.append(f"- 第 {idx + 1} 组：年化 {perf.annual_return:.2%} · 夏普 {perf.sharpe:.2f} · 最大回撤 {perf.max_drawdown:.2%}")
        monotonic = all(annualized[i] >= annualized[i + 1] for i in range(len(annualized) - 1))
        lines.append(f"- 单调性：{'各组收益单调（质量较好）' if monotonic else '各组收益不单调，因子质量偏弱'}")
        lines.append("")

        long_short = group_returns[groups - 1].fillna(0) - group_returns[0].fillna(0) - 2 * cost
        ls_perf = _performance(long_short, rf_rate=rf_rate)
        lines.append("## 多空组合（Top − Bottom）")
        lines.append("")
        lines.extend(ls_perf.render("多空"))
        lines.append("")
        lines.append("## 过拟合与局限")
        lines.append("")
        lines.append(f"- 样本划分：样本内 {is_ratio:.0%}（按时间切分）；试验披露：本次共试验 {trials} 个候选（上限 {MAX_TRIALS}）")
        lines.append("- **生存者偏差**：股票池为当前成分，历史回测含前视偏差，结论需打折")
        lines.append("- 结论限定于该池该区间；不外推全市场，不构成投资建议。")

        nav = pd.DataFrame({"date": closes.index})
        for idx in range(groups):
            nav[f"group_{idx + 1}_nav"] = (1 + (group_returns[idx].fillna(0) - cost)).cumprod().to_numpy()
        nav["long_short_nav"] = (1 + long_short.fillna(0)).cumprod().to_numpy()
        parquet_path = persist_frame(
            nav, cache_dir=self._cache_dir(), name="backtest_cross_section"
        )
        return RawData(
            kind="text", text="\n".join(lines), endpoint="backtest:cross_section",
            params={"pool_size": len(closes.columns), "factor_expr": factor_expr, "trials": trials},
            df=nav,
            parquet_path=parquet_path,
        )

    # -- 辅助函数 --------------------------------------------------------------
    def _cache_dir(self) -> str:
        """数据缓存目录，派生数据帧持久化于此。"""
        settings = getattr(self.data, "settings", None)
        if settings is None:
            return "data_cache"
        return str(settings.data.cache_dir)

    @staticmethod
    def _market_frame(df: pd.DataFrame) -> pd.DataFrame:
        """把 K 线数据帧规范化为以日期为索引、按时间升序的 OHLCV。"""
        frame = df.copy()
        if "date" not in frame.columns or "close" not in frame.columns:
            raise ValueError("K 线数据缺少 date/close 列")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").set_index("date")
        for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    @staticmethod
    def _variable_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """把每个行情变量封装为单列的「日期 x 标的」数据帧。"""
        variables: dict[str, pd.DataFrame] = {}
        for column in frame.columns:
            variables[str(column)] = frame[[column]].rename(columns={column: "SYM"})
        return variables

    @staticmethod
    def _classify_factor_variables(
        names: tuple[str, ...],
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """把表达式变量分成行情列与估值指标。

        行情列随 K 线一并取出；估值名经 ``normalize_valuation_indicator`` 变成
        规范指标（``pe_ttm`` → ``市盈率(TTM)``）。两者都不是的名字立刻拒绝，
        避免先取面板再失败。
        """
        market: list[str] = []
        valuation: dict[str, str] = {}
        unknown: list[str] = []
        for name in names:
            if name in MARKET_VARIABLES:
                market.append(name)
                continue
            try:
                valuation[name] = normalize_valuation_indicator(name)
            except ValueError:
                unknown.append(name)
        if unknown:
            joined = "、".join(unknown)
            raise ValueError(
                f"未知变量：{joined}。横截面因子目前只支持行情变量"
                f"（{'/'.join(MARKET_VARIABLES)}）与估值序列"
                "（pe、pe_ttm、pb、pcf、总市值及其别名）。"
                "财报字段（如 ROE）不在面板中，请更换表达式，不要原样重试。"
            )
        return tuple(market), valuation

    async def _valuation_panels(
        self,
        symbols: list[str],
        *,
        years: int,
        index: pd.Index,
        variables: dict[str, str],
    ) -> dict[str, pd.DataFrame]:
        """按规范指标各取一次估值面板，再挂回表达式里的变量名。"""
        by_indicator: dict[str, list[str]] = {}
        for name, indicator in variables.items():
            by_indicator.setdefault(indicator, []).append(name)
        frames: dict[str, pd.DataFrame] = {}
        for indicator, names in by_indicator.items():
            panel = await self._valuation_panel(
                symbols, years=years, indicator=indicator, index=index
            )
            if panel is None or panel.dropna(how="all").empty:
                raise ValueError(
                    f"未取到估值序列 {indicator}（变量 {'、'.join(names)}），无法计算因子"
                )
            for name in names:
                frames[name] = panel
        return frames

    async def _valuation_panel(
        self,
        symbols: list[str],
        *,
        years: int,
        indicator: str,
        index: pd.Index,
    ) -> pd.DataFrame | None:
        """逐标的取一条估值序列，对齐到收盘价日期（向前填充）。"""

        async def one(symbol: str) -> tuple[str, pd.Series] | None:
            try:
                raw = await self.data.valuation(symbol, lookback_years=years, indicator=indicator)
            except Exception:
                return None
            series = self._valuation_series(raw.df)
            if series is None:
                return None
            return symbol, series.reindex(index.union(series.index)).sort_index().ffill().reindex(index)

        return await self._gather_panel(symbols, one, phase="valuation")

    @staticmethod
    def _valuation_series(df: pd.DataFrame | None) -> pd.Series | None:
        """从估值表抽出按日期升序的数值列（指标列名带单位，不依赖固定列名）。"""
        if df is None or not len(df) or "date" not in df.columns:
            return None
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frame = frame.drop_duplicates("date", keep="last")
        numeric = [
            column
            for column in frame.columns
            if column != "date" and pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        if not numeric:
            return None
        series = pd.to_numeric(frame[numeric[-1]], errors="coerce")
        series.index = pd.DatetimeIndex(frame["date"])
        return series

    @staticmethod
    def _signal_from_factor(factor: pd.Series, params: dict[str, Any]) -> pd.Series:
        """把因子序列映射为仓位；默认因子 > 0 时做多（long_only）。

        params.signal="long_short" 时按因子符号取值（±1）。返回仓位序列。
        """
        mode = str(params.get("signal", "long_only"))
        if mode == "long_short":
            return np.sign(factor).fillna(0.0)
        threshold = float(params.get("threshold", 0.0))
        return (factor > threshold).astype(float).fillna(0.0)

    @staticmethod
    def _apply_position(
        position: pd.Series, asset_returns: pd.Series, *, cost_bps: float, slippage_bps: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """按换手率计提成本；返回 (net, benchmark, turnover)。

        仓位延后一期参与收益（pos.shift(1)），成本按换手率乘以
        (cost_bps + slippage_bps) 基点计提。
        """
        pos = position.fillna(0.0)
        gross = pos.shift(1).fillna(0.0) * asset_returns
        turnover = pos.diff().abs().fillna(pos.abs())
        cost = turnover * (cost_bps + slippage_bps) / 10000
        net = gross - cost
        benchmark = asset_returns.fillna(0.0)
        return net, benchmark, turnover

    @staticmethod
    def _ic_lines(label: str, ic: pd.Series) -> list[str]:
        clean = ic.dropna()
        if not len(clean):
            return [f"- {label}：无有效样本"]
        mean = float(clean.mean())
        std = float(clean.std())
        icir = mean / std if std > 0 else 0.0
        win = float((clean > 0).mean())
        return [
            f"- {label}：IC 均值 {mean:.3f} · ICIR {icir:.2f} · IC>0 占比 {win:.1%}",
        ]

    @staticmethod
    def _group_returns(
        factor: pd.DataFrame, forward: pd.DataFrame, returns: pd.DataFrame,
        *, groups: int, rebalance: int,
    ) -> list[pd.Series]:
        """等权分组收益，每 ``rebalance`` 日调仓一次。

        信号日在每隔 ``rebalance`` 行处取样；随后一个窗口的分组收益为该组
        成员前向收益的均值。按调仓窗口取平均、窗口之间保持持仓不变，可避免
        把「逐日调仓」的结果误读为因子本身的超额收益。
        """
        group_series = [pd.Series(0.0, index=factor.index) for _ in range(groups)]
        for pos in range(0, len(factor), rebalance):
            row = factor.iloc[pos].dropna()
            fwd = forward.iloc[pos]
            valid = row.index.intersection(fwd.dropna().index)
            if len(valid) < groups:
                continue
            ranks = row[valid].rank(method="first")
            labels = np.ceil(ranks / len(valid) * groups).clip(1, groups).astype(int)
            window_end = min(pos + rebalance, len(factor))
            for group in range(1, groups + 1):
                members = labels[labels == group].index
                if not len(members):
                    continue
                # 窗口期内该组成员的逐日平均收益。
                window = returns.iloc[pos:window_end][members]
                if window.empty:
                    continue
                group_series[group - 1].iloc[pos:window_end] = window.mean(axis=1).to_numpy()
        return group_series

    async def _resolve_pool(self, pool: list[str] | str) -> list[str]:
        """把股票池解析为代码列表（指数/行业成分，或字面代码列表）。"""
        if isinstance(pool, str):
            key = pool.strip()
            # 也接受显式的逗号/空格分隔列表。
            if any(sep in key for sep in (",", "，", " ")):
                parts = [p for p in re.split(r"[,\s，]+", key) if p]
                return self._validate_symbols(parts)
            try:
                raw = await self.data.index_constituents(key)
                return self._symbols_from(raw.df)
            except Exception:
                raw = await self.data.industry_constituents(key)
                return self._symbols_from(raw.df)
        return self._validate_symbols([str(s) for s in pool])

    @staticmethod
    def _symbols_from(df: pd.DataFrame | None) -> list[str]:
        if df is None or not len(df):
            return []
        column = "symbol" if "symbol" in df.columns else ("证券代码" if "证券代码" in df.columns else None)
        if column is None:
            return []
        values = df[column].astype(str).str.zfill(6)
        return [v for v in values.tolist() if v.isdigit() and len(v) == 6]

    @staticmethod
    def _validate_symbols(symbols: list[str]) -> list[str]:
        cleaned = []
        for symbol in symbols:
            value = str(symbol).strip().zfill(6)
            if value.isdigit() and len(value) == 6 and value not in cleaned:
                cleaned.append(value)
        if not cleaned:
            raise ValueError("股票池为空或代码格式非法（需 6 位 A 股代码）")
        return cleaned

    async def _panel(
        self, symbols: list[str], *, years: int, columns: tuple[str, ...] = ()
    ) -> dict[str, pd.DataFrame]:
        """逐标的取行情并对齐为「日期 x 标的」面板，至少含 close。

        ``columns`` 是表达式额外点名的行情变量（open/volume 等），与 close
        同一次 K 线取回。面板构建是横截面回测里最慢的一段（逐只网络取数且受
        全局节流约束），因此按完成数上报进度。
        """
        wanted = tuple(dict.fromkeys(("close", *columns)))

        async def one(symbol: str) -> tuple[str, dict[str, pd.Series]] | None:
            try:
                raw = await self.data.kline(symbol, years=years)
            except Exception:
                return None
            df = raw.df
            if df is None or not len(df) or "date" not in df.columns or "close" not in df.columns:
                return None
            keep = ["date", *[column for column in wanted if column in df.columns]]
            frame = df[keep].copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values("date")
            frame = frame.drop_duplicates("date", keep="last").set_index("date")
            series = {
                column: pd.to_numeric(frame[column], errors="coerce")
                for column in wanted
                if column in frame.columns
            }
            if "close" not in series:
                return None
            return symbol, series

        buckets: dict[str, dict[str, pd.Series]] = {column: {} for column in wanted}

        def accept(result: tuple[str, dict[str, pd.Series]]) -> None:
            symbol, series = result
            for column, values in series.items():
                buckets[column][symbol] = values

        await self._collect(
            symbols,
            one,
            phase="panel",
            accept=accept,
            available=lambda: len(buckets["close"]),
        )
        return {
            column: pd.DataFrame(mapping).sort_index()
            for column, mapping in buckets.items()
            if mapping
        }

    async def _gather_panel(self, symbols: list[str], one, *, phase: str) -> pd.DataFrame | None:
        """把 ``one(symbol) -> (symbol, series) | None`` 收成「日期 x 标的」表。"""
        series: dict[str, pd.Series] = {}

        def accept(result: tuple[str, pd.Series]) -> None:
            series[result[0]] = result[1]

        await self._collect(symbols, one, phase=phase, accept=accept, available=lambda: len(series))
        if not series:
            return None
        return pd.DataFrame(series).sort_index()

    async def _collect(self, symbols, one, *, phase: str, accept, available) -> None:
        """并发取数并按完成数上报进度；取消本协程时一并取消未完成的任务。"""
        total = len(symbols)
        started = time.monotonic()
        fetched = 0
        tasks = [asyncio.ensure_future(one(symbol)) for symbol in symbols]
        try:
            for completed in asyncio.as_completed(tasks):
                result = await completed
                fetched += 1
                if result is not None:
                    accept(result)
                if fetched % _PROGRESS_EVERY == 0 or fetched == total:
                    elapsed = time.monotonic() - started
                    await self._report_progress(
                        {
                            "phase": phase,
                            "fetched": fetched,
                            "total": total,
                            "available": available(),
                            "elapsed_s": round(elapsed, 1),
                            # 线性外推，仅作量级提示；取数快慢随命中缓存与网络波动。
                            "eta_s": round(elapsed / fetched * (total - fetched), 1) if fetched else None,
                        }
                    )
        finally:
            # 工具级的 asyncio.wait_for 超时会取消本协程；此时未完成的取数任务必须
            # 一并取消，否则它们会继续在后台跑并占用线程池。
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _report_progress(self, payload: dict[str, Any]) -> None:
        """上报一次中间进展；未注入回调时静默跳过（如单测直接调用工具）。"""
        report = getattr(self, "progress", None)
        if report is None:
            return
        await report(payload)
