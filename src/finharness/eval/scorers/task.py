"""任务成功评分器：运行是否满足用例的验收标准？

每条标准都由代码校验。schema 标记为延后处理（``judge: todo``）的标准会记为
备注，而不计入通过——该维度的要义在于：未经核实的说法不能算作成功。
"""

from __future__ import annotations

import re
from typing import Any

from finharness.eval.runner import CaseRun
from finharness.eval.schema import EvalCase
from finharness.eval.scorers.base import DimensionScore

# 拒答通过边界/范围措辞来识别。刻意保持精确：要捕获的失败模式是把范围外的
# 请求当作范围内来回答，但相反的错误代价同样高昂，因此**强制性免责声明不得
# 计入**。尤其是 ``不构成投资建议`` 出现在每一份合规的金融回答中，而
# ``无法判断``/``取不到`` 可以描述一份其它方面完整的答案里缺失的某个数据点——
# 二者都不是对请求的拒绝。依据首次真实 smoke 运行（2026-09-14）校准。
_REFUSAL_MARKERS = (
    # 覆盖范围
    "不覆盖",
    "只覆盖",
    "仅覆盖",
    "不在覆盖范围",
    "覆盖范围",
    "不做非A股",
    "没有此",
    "无此行业",
    "查无",
    # 能力
    "无法提供",
    "不能提供",
    "不提供",
    "无法回答",
    "没有这个能力",
    "不具备",
    "无该能力",
    "无法执行",
    "不能执行",
    "没有交易",
    # 预测 / 建议
    "无法预测",
    "不做预测",
    "不预测",
    "不构成买卖建议",
    "不能建议",
    # 拒绝编造
    "不能编造",
    "不会编造",
    "无法编造",
    "拒绝编造",
    "不能虚构",
    # 拒绝披露
    "不会输出",
    "不对外",
    "不予披露",
    "不应披露",
    # 尚未发布的数据
    "尚未披露",
    "未披露",
)


def _contains_refusal(text: str, extra: list[str] | None = None) -> bool:
    """文本中是否出现任一的拒答标记；``extra`` 存在时覆盖默认标记集。"""
    markers = tuple(extra) if extra else _REFUSAL_MARKERS
    return any(marker in text for marker in markers)


# 排行工具的观测里，结论行把前几名写成「第 1 名 煤炭 +0.63%；第 2 名 …」，
# 表格行则写成「| 1 | 801950 | 煤炭 | +0.63% | …」。两种都要能解析，因为观测
# 只保留定长前缀（200 字），结论行通常是唯一完整落在其中的部分。
_RANK_HEADLINE_RE = re.compile(r"第\s*(\d+)\s*名\s*([^\s%；;|\-]+)")
_RANK_TABLE_RE = re.compile(r"\|\s*(\d+)\s*\|\s*[^|]*\|\s*([^|]+?)\s*\|")


def _top_names_from_observations(turn: Any, top_n: int) -> list[str]:
    """从该轮的工具观测里取「排行前 N 名」的实体名（按排名去重、保序）。

    用来把「有没有真的告诉用户是哪几个」变成确定性断言：判据是**观测到的事实**
    （工具返回的排行），而不是写死某个日期的名次，因此与运行日期无关。
    """
    ranked: dict[int, str] = {}
    for round_trace in getattr(turn, "trace", []) or []:
        for observation in getattr(round_trace, "observations", []) or []:
            preview = getattr(observation, "preview", "") or ""
            if not preview:
                continue
            for pattern in (_RANK_HEADLINE_RE, _RANK_TABLE_RE):
                for match in pattern.finditer(preview):
                    rank = int(match.group(1))
                    name = match.group(2).strip()
                    if rank and rank not in ranked and name:
                        ranked[rank] = name
    return [ranked[rank] for rank in sorted(ranked)[:top_n]]


def score_task(case: EvalCase, run: CaseRun) -> DimensionScore:
    """对任务完成率打分：逐轮校验答案、引用与产物等验收标准。"""
    score = DimensionScore(dimension="task")

    if run.error is not None:
        score.add("run_completed", False, f"用例执行异常：{run.error}")
        return score
    score.add("run_completed", True)

    # 逐轮的答案断言；某一轮的期望只作用于该轮。
    for index, turn in enumerate(case.all_turns):
        if index >= len(run.turns):
            score.add(f"turn{index + 1}_present", False, "该轮未执行")
            continue
        answer = run.turns[index].answer
        expect = turn.expect.answer

        if expect.refusal is True:
            score.add(
                f"turn{index + 1}_refusal",
                _contains_refusal(answer, expect.refusal_markers),
                f"应明确拒绝/说明边界；回答首 80 字：{answer[:80]}",
            )
        elif expect.refusal is False:
            score.add(
                f"turn{index + 1}_no_refusal",
                not _contains_refusal(answer, expect.refusal_markers),
                "不应拒答；但回答含拒答措辞",
            )

        if expect.contains_any:
            hit = [term for term in expect.contains_any if term in answer]
            score.add(
                f"turn{index + 1}_contains_any",
                bool(hit),
                f"需至少命中其一 {expect.contains_any}；命中 {hit}",
            )
        if expect.contains_all:
            missing = [term for term in expect.contains_all if term not in answer]
            score.add(
                f"turn{index + 1}_contains_all",
                not missing,
                f"缺失关键词：{missing}",
            )
        if expect.contains_none:
            present = [term for term in expect.contains_none if term in answer]
            score.add(
                f"turn{index + 1}_contains_none",
                not present,
                f"出现了禁止词：{present}",
            )
        if expect.matches_any:
            matched = [p for p in expect.matches_any if re.search(p, answer)]
            score.add(
                f"turn{index + 1}_matches_any",
                bool(matched),
                f"需命中其一正则 {expect.matches_any}；命中 {matched}",
            )
        if expect.top_names_from_observations:
            names = _top_names_from_observations(run.turns[index], expect.top_names_from_observations)
            missing = [name for name in names if name not in answer]
            score.add(
                f"turn{index + 1}_top_names_in_answer",
                bool(names) and not missing,
                f"排行前 {expect.top_names_from_observations} 名未在答案中点名：{missing}"
                if missing
                else f"点名了排行前 {expect.top_names_from_observations} 名：{names}",
            )
        if expect.min_chars is not None:
            score.add(
                f"turn{index + 1}_min_chars",
                len(answer) >= expect.min_chars,
                f"回答长度 {len(answer)} < {expect.min_chars}",
            )
        if expect.max_chars is not None:
            score.add(
                f"turn{index + 1}_max_chars",
                len(answer) <= expect.max_chars,
                f"回答长度 {len(answer)} > {expect.max_chars}",
            )

    # 引用要求属于“该声明有据可查”的一部分。
    produced = sum(
        len(getattr(turn.outcome, "citations", []) or []) for turn in run.turns
    )
    if case.all_turns and case.all_turns[-1].expect.citations.required:
        score.add(
            "citations_required",
            produced > 0,
            f"引用数 {produced}",
        )
    minimum = min(
        (turn.expect.citations.min for turn in case.all_turns), default=0
    )
    if minimum:
        score.add("citations_min", produced >= minimum, f"引用数 {produced} < {minimum}")

    # 产物期望同样属于任务成功：报告要么存在且干净，要么该用例未成功。
    for turn in case.all_turns:
        expected = turn.expect.artifacts
        if not expected.report_exported:
            continue
        exported = bool(run.exports)
        score.add("report_exported", exported, f"产物：{run.exports}")
        if not exported:
            continue
        if expected.no_unsourced_numbers:
            residual = len(re.findall(r"\[!无来源:\d+\]", run.report_text))
            score.add(
                "report_no_unsourced_numbers",
                residual == 0,
                f"报告残留无来源标记 {residual} 处",
            )
        if expected.no_tool_names:
            leaked = _leaked_tool_names(run.report_text, run)
            score.add(
                "report_no_tool_names",
                not leaked,
                f"正文出现内部工具名：{leaked}",
            )

    if case.judge == "todo":
        score.notes.append(
            "该用例含模糊判定标准，v1 仅做结构近似；建议后续接入 LLM 裁判复核。"
        )
    # 任务成功是已声明验收标准中通过项所占的比例；未声明任何标准的用例视为
    # 自动满足，而非记为零分。
    if score.checks:
        score.score = sum(1 for check in score.checks if check.passed) / len(
            score.checks
        )
    else:
        score.score = 1.0
        score.notes.append("用例未声明任务验收标准")
    return score


# 绝不允许出现在面向读者正文中的工具名。取自本次运行实际用过的工具目录加上
# 始终存在的名称，从而任何提到内部管线的报告都会被捕获。
_ALWAYS_INTERNAL = (
    "get_quote",
    "get_kline",
    "get_indicators",
    "get_financials",
    "get_valuation",
    "get_peers",
    "get_market_news",
    "make_chart",
    "write_report",
    "search_tools",
    "research_plan",
    "calc_metrics",
)


def _leaked_tool_names(text: str, run: CaseRun) -> list[str]:
    """找出文本中泄露的内部工具名（始终内部 + 实际调用 + 尝试调用）。"""
    candidates = set(_ALWAYS_INTERNAL) | set(run.called_tools()) | set(run.attempted_tools())
    return sorted(name for name in candidates if name and name in text)


__all__ = ["score_task"]
