"""效率评分器：token 消耗、步数与时延相对于预算的表现。

每个子指标给出 [0, 1] 区间内的比率（预算/实际，上限为 1），三者按配置权重
合成。绝对值作为备注附上，以便即使用例通过也能观察到趋势。
"""

from __future__ import annotations

from finharness.eval.config import EvalConfig
from finharness.eval.runner import CaseRun
from finharness.eval.schema import EvalCase
from finharness.eval.scorers.base import DimensionScore


def _ratio(actual: float, budget: float | None) -> tuple[float, bool]:
    """预算/实际，上限为 1；缺预算时判为通过且不设加分上限。"""
    if budget is None or budget <= 0:
        return 1.0, True
    if actual <= 0:
        return 1.0, True
    return min(budget / actual, 1.0), actual <= budget


def score_efficiency(
    case: EvalCase, run: CaseRun, config: EvalConfig
) -> DimensionScore:
    """对一次运行的效率打分：token、步数与时延相对预算的比率加权合成。"""
    score = DimensionScore(dimension="efficiency")
    totals = run.totals()

    max_tokens = case.budget.max_tokens or config.efficiency.default_max_tokens
    max_rounds = case.budget.max_rounds or config.efficiency.default_max_rounds
    max_seconds = case.budget.max_seconds or config.efficiency.default_max_seconds

    token_ratio, token_ok = _ratio(totals["total_tokens"], max_tokens)
    steps = totals["rounds"] or totals["tool_calls"]
    step_budget = case.budget.max_steps or max_rounds
    step_ratio, step_ok = _ratio(steps, step_budget)
    seconds = run.duration_ms / 1000.0
    latency_ratio, latency_ok = _ratio(seconds, max_seconds)

    score.add(
        "token_budget",
        token_ok,
        f"tokens {totals['total_tokens']} / 预算 {max_tokens}",
    )
    score.add("step_budget", step_ok, f"steps {steps} / 预算 {step_budget}")
    score.add("latency_budget", latency_ok, f"{seconds:.1f}s / 预算 {max_seconds}s")

    weights = config.efficiency.sub_weights
    score.score = (
        weights.tokens * token_ratio
        + weights.steps * step_ratio
        + weights.latency * latency_ratio
    )
    score.notes.append(
        f"绝对值：in={totals['input_tokens']} out={totals['output_tokens']} "
        f"rounds={totals['rounds']} tool_calls={totals['tool_calls']} "
        f"duration={seconds:.1f}s"
    )
    return score


__all__ = ["score_efficiency"]
