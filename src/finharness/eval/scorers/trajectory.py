"""轨迹评分器：记录的路径是否与期望路径一致？

每个声明的组成部分按其权重计分；执行了禁调工具是一票否决，无论其余部分如何
都会将整个维度归零。“答案对了”并不蕴含“过程合理”——这正是轨迹要与答案分开
记录的原因。
"""

from __future__ import annotations

from finharness.eval.runner import CaseRun
from finharness.eval.scorers.base import DimensionScore
from finharness.eval.schema import EvalCase

# 各组成部分的相对权重；按实际出现的权重做归一化。
_W_REQUIRED = 0.40
_W_FORBIDDEN = 0.30
_W_ORDER = 0.20
_W_REDUNDANCY = 0.10


def _subsequence_in_order(sequence: list[str], expected: list[str]) -> bool:
    """``expected`` 是否作为有序子序列出现在 ``sequence`` 中。"""
    iterator = iter(sequence)
    return all(item in iterator for item in expected)


def _strictest_min(current: int | None, incoming: int) -> int:
    """在已有上限与新的上限之间取更严格（更小）者。"""
    return incoming if current is None else min(current, incoming)


def score_trajectory(case: EvalCase, run: CaseRun) -> DimensionScore:
    """对轨迹正确性打分：按声明的工具、顺序、技能、计划与冗余约束加权计分。

    命中禁调/禁成功项时一票否决，该维度归零。
    """
    score = DimensionScore(dimension="trajectory")

    # 合并每一轮的期望：一个用例的路径是作为整体来评判的。
    must: list[str] = []
    any_of: list[str] = []
    must_not: list[str] = []
    skills_must: list[str] = []
    skills_must_not: list[str] = []
    orders: list[list[str]] = []
    must_not_succeed: list[str] = []
    max_repeats: int | None = None
    max_rounds: int | None = None
    max_tool_calls: int | None = None
    plan_required: bool | None = None

    for turn in case.turns:
        trajectory = turn.expect.trajectory
        must.extend(trajectory.tools_must)
        any_of.extend(trajectory.tools_any)
        must_not.extend(trajectory.tools_must_not)
        skills_must.extend(trajectory.skills_must)
        skills_must_not.extend(trajectory.skills_must_not)
        orders.extend(trajectory.order)
        must_not_succeed.extend(trajectory.must_not_succeed)
        if trajectory.max_repeats is not None:
            max_repeats = _strictest_min(max_repeats, trajectory.max_repeats)
        if trajectory.max_rounds is not None:
            max_rounds = _strictest_min(max_rounds, trajectory.max_rounds)
        if trajectory.max_tool_calls is not None:
            max_tool_calls = _strictest_min(max_tool_calls, trajectory.max_tool_calls)
        if trajectory.plan_required is not None:
            plan_required = trajectory.plan_required

    declared = any(
        [must, any_of, must_not, skills_must, skills_must_not, orders, must_not_succeed]
    ) or plan_required is not None or max_repeats is not None or max_tool_calls is not None
    if not declared:
        score.score = 1.0
        score.notes.append("用例未声明轨迹约束")
        return score

    executed = run.executed_sequence()
    weighted = 0.0
    total_weight = 0.0
    vetoed = False

    if must:
        total_weight += _W_REQUIRED
        unique = list(dict.fromkeys(must))
        hit = [name for name in unique if name in executed]
        recall = len(hit) / len(unique)
        weighted += _W_REQUIRED * recall
        score.add(
            "tools_must",
            recall == 1.0,
            f"必调工具召回 {len(hit)}/{len(unique)}：缺 {sorted(set(unique) - set(hit))}",
        )
    if any_of:
        total_weight += _W_REQUIRED
        ok = any(name in executed for name in any_of)
        weighted += _W_REQUIRED * (1.0 if ok else 0.0)
        score.add("tools_any", ok, f"应至少调用 {any_of} 之一；实际 {executed}")
    if must_not:
        total_weight += _W_FORBIDDEN
        violated = [name for name in dict.fromkeys(must_not) if name in executed]
        vetoed = vetoed or bool(violated)
        weighted += 0.0 if violated else _W_FORBIDDEN
        score.add(
            "tools_must_not",
            not violated,
            f"禁调工具被调用：{violated}" if violated else "未调用禁调工具",
        )
    if must_not_succeed:
        total_weight += _W_FORBIDDEN
        succeeded_forbidden = [
            name for name in dict.fromkeys(must_not_succeed) if name in executed
        ]
        vetoed = vetoed or bool(succeeded_forbidden)
        weighted += 0.0 if succeeded_forbidden else _W_FORBIDDEN
        score.add(
            "must_not_succeed",
            not succeeded_forbidden,
            f"下列调用本不应成功执行：{succeeded_forbidden}",
        )
    if orders:
        total_weight += _W_ORDER
        passed_orders = sum(
            1 for expected_order in orders if _subsequence_in_order(executed, expected_order)
        )
        weighted += _W_ORDER * (passed_orders / len(orders))
        score.add(
            "order",
            passed_orders == len(orders),
            f"顺序满足 {passed_orders}/{len(orders)}；实际执行序列 {executed}",
        )
    if skills_must:
        total_weight += _W_REQUIRED
        loaded = set(run.loaded_skills())
        missing = [name for name in dict.fromkeys(skills_must) if name not in loaded]
        weighted += _W_REQUIRED * (1.0 if not missing else 0.0)
        score.add("skills_must", not missing, f"缺失技能：{missing}")
    if skills_must_not:
        total_weight += _W_FORBIDDEN
        loaded = set(run.loaded_skills())
        present = [name for name in dict.fromkeys(skills_must_not) if name in loaded]
        vetoed = vetoed or bool(present)
        weighted += 0.0 if present else _W_FORBIDDEN
        score.add("skills_must_not", not present, f"不应加载：{present}")
    if plan_required is not None:
        total_weight += _W_REQUIRED
        has_plan = run.plan_present()
        ok = has_plan if plan_required else not has_plan
        weighted += _W_REQUIRED * (1.0 if ok else 0.0)
        score.add(
            "plan_required",
            ok,
            f"plan_required={plan_required}；实际{'有' if has_plan else '无'}计划",
        )
    if max_repeats is not None:
        total_weight += _W_REDUNDANCY
        largest = run.max_repeat_count()
        ok = largest <= max_repeats
        weighted += _W_REDUNDANCY * (1.0 if ok else 0.0)
        score.add("max_repeats", ok, f"同参重复最多 {largest} 次，上限 {max_repeats}")
    if max_tool_calls is not None:
        total_weight += _W_REDUNDANCY
        used = len(executed)
        ok = used <= max_tool_calls
        weighted += _W_REDUNDANCY * (1.0 if ok else 0.0)
        score.add("max_tool_calls", ok, f"工具调用 {used} 次，上限 {max_tool_calls}")
    if max_rounds is not None:
        total_weight += _W_REDUNDANCY
        rounds = sum(int(getattr(turn.outcome, "rounds", 0)) for turn in run.turns)
        ok = rounds <= max_rounds
        weighted += _W_REDUNDANCY * (1.0 if ok else 0.0)
        score.add("max_rounds", ok, f"轮数 {rounds}，上限 {max_rounds}")

    score.score = 0.0 if vetoed else (weighted / total_weight if total_weight else 1.0)
    if vetoed:
        score.notes.append("命中禁调/禁成功项，轨迹分归零（一票否决）")
    return score


__all__ = ["score_trajectory"]
