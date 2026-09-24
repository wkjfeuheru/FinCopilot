"""计划进展的软信号：标的偏离、能力错配、停滞（docs 03.6.2）。

这些方法只读取 ``self`` 上的会话状态（``ctx.plan``、``_last_user_msg`` 与几个
一次性报告集合），不写任何持久化内容，也不驱动控制流——产出的是一段追加到
下一次请求的**软提示**，从不阻断调用。因此它们可以作为一个 ``AgentLoop`` 的
mixin 单独存在，使这份 200 行的判定逻辑与循环主体分开阅读与测试。

``AgentLoop`` 提供 ``ctx`` / ``settings`` / ``_last_user_msg`` 以及
``_reported_drift`` / ``_reported_mismatch`` / ``_plan_signature`` /
``_plan_stall_turns`` 这几个实例状态；本模块只消费它们。
"""

from __future__ import annotations

import re
from typing import Any

from finharness.context.session import Plan
from finharness.shared.capabilities import (
    Capability,
    UnknownCapabilityError,
    capabilities_in_text,
    capabilities_of,
    capability_of,
    is_research_capability,
)
from finharness.types import ToolUse


class PlanProgressMixin:
    """计划进展判定；与 ``AgentLoop`` 组合使用（见模块说明）。"""

    # -- 计划进展（docs 03.6.2） ----------------------------------------------
    # 偏离是相对于*任务*来判定的，而非相对于工具白名单。抓取数据、读取缓存、
    # 查看公告、为结果作图——这些每一项都是研究任务得以开展的方式，因此没有
    # 任何一项本身构成偏离。可能偏离目标的是标的：去拉取任务从未点名的 symbol。
    # 这是此处唯一检查的事情，且仅当计划本身点名了 symbol 时才检查
    # （见 ``_plan_scope_drift``）。
    _SYMBOL_RE = re.compile(r"\b\d{6}\b")

    def _plan_prose(self, plan: Plan) -> str:
        """计划的自由文本——目标与各步骤动作，而非其工具提示。

        提示是模型填写的簿记信息；散文式的表述才是其真实意图。下面的两个信号
        从该表述中读取意图（点名的 symbol、隐含的 capability），因此一份恰好
        不完整的提示列表不会让一个正当操作显得错误。
        """
        return " ".join([plan.goal, *(step.action for step in plan.steps)])

    def _plan_declared_symbols(self, plan: Plan) -> set[str]:
        """计划明确锁定的 symbol。

        从目标与各步骤动作——即任务描述——中读取，而 **不** 从工具提示中读取。
        提示携带的是工具名；出现在其中的 symbol 只是偶然，把它当作 scope 正是
        导致一份提示写着 "get_peers" 的计划看起来像点名了研究标的的原因。
        """
        return set(self._SYMBOL_RE.findall(self._plan_prose(plan)))

    def _question_named_symbol(self) -> str | None:
        """用户问题所点名的那个 symbol，前提是它恰好只点名了一个。

        这是单标的任务的真正 scope。计划可能碰巧点了该 symbol（也可能没有），
        但用户的问题才是事实依据，因此无论计划的表述选择写下什么，针对该点名
        symbol 的操作都不会被解读为偏离。
        """
        prompt = str(getattr(self, "_last_user_msg", "") or "")
        found = self._SYMBOL_RE.findall(prompt)
        unique = list(dict.fromkeys(found))
        return unique[0] if len(unique) == 1 else None

    def _plan_scope_drift(self, tool_uses: list[ToolUse]) -> list[str]:
        """本轮触及但任务从未点名的 symbol。

        任务自身的 symbol——问题点名的单个 symbol 加上计划表述中点名的 symbol
        ——构成 scope。除此之外的任何内容都是对研究标的的真实扩大。仅在 scope
        可知时才判定：行业或宏观问题不点名任何 symbol，因此没有 symbol 可能是
        偏离目标的，整个信号被跳过。一个 symbol 每次运行最多报告一次，因此一个
        跑偏的标的只产生一次提示，而不是每轮都唠叨。
        """
        plan = self.ctx.plan
        if plan is None:
            return []
        scope = set(self._plan_declared_symbols(plan))
        named = self._question_named_symbol()
        if named:
            scope.add(named)
        if not scope:
            return []
        drift: list[str] = []
        for tool_use in tool_uses:
            args = tool_use.args if isinstance(tool_use.args, dict) else {}
            symbol = args.get("symbol")
            if not symbol:
                continue
            value = str(symbol)
            if value in scope or value in self._reported_drift:
                continue
            self._reported_drift.add(value)
            drift.append(value)
        return drift

    def _plan_expected_capabilities(self, plan: Plan) -> set[Capability]:
        """本轮允许使用的 capability，来自计划的*意图*。

        两个来源取并集：计划表述所隐含的 capability（某步骤写着"分析估值"
        就需要估值 capability），以及其工具提示所点名的 capability。读取表述
        正是为了防止一份提示单薄的计划把任务显然需要的工具标记为问题。
        流程/展示类 capability 从不计入——``make_chart`` 的提示并不能说明任务
        需要哪些证据。
        """
        hinted = [
            hint for step in plan.steps for hint in step.tool_hint if hint
        ]
        expected = capabilities_of(hinted) | capabilities_in_text(self._plan_prose(plan))
        question = str(getattr(self, "_last_user_msg", "") or "")
        expected |= capabilities_in_text(question)
        return {
            capability for capability in expected if is_research_capability(capability)
        }

    def _plan_capability_mismatch(self, tool_uses: list[ToolUse]) -> list[str]:
        """本轮使用但计划从未指向的 capability。

        这是 capability 层面的"工具与任务不匹配"信号：当每一步都关乎公告时，
        某一轮却去抓新闻，这是真正的替换；而用不同工具名读取同一数据、
        ``detail=full``，或后续的计算调用则不是。

        与偏离遵循同样的纪律：仅当计划确实提示了工具时才判定（否则没有任何明示
        的期望），绝不阻断调用，且一个 capability 每次运行最多报告一次。
        """
        plan = self.ctx.plan
        if plan is None:
            return []
        expected = self._plan_expected_capabilities(plan)
        if not expected:
            return []
        mismatch: list[str] = []
        for tool_use in tool_uses:
            try:
                capability = capability_of(tool_use.name)
            except UnknownCapabilityError:
                continue
            if not is_research_capability(capability):
                continue
            label = capability.value
            if capability in expected or label in self._reported_mismatch:
                continue
            self._reported_mismatch.add(label)
            mismatch.append(label)
        return mismatch

    def _plan_progress(self, tool_uses: list[ToolUse]) -> dict[str, Any] | None:
        """在一轮之后总结计划进展；没有计划时返回 ``None``。

        计算三个软信号，其中任何一个都不阻断调用：

        * ``drift``——本轮触及但任务从未点名的标的。仅在计划声明了 symbol 时
          判定（见 ``_plan_scope_drift``）。
        * ``mismatch``——本轮使用但没有任何步骤提示的 capability，判定在
          capability 层面，从而使策略/模式选择不受约束（见
          ``_plan_capability_mismatch``）。
        * ``stalled_turns``——计划的 (revision, statuses) 签名连续未发生变化的
          轮数，即运行在推进却没有记录进展。
        """
        plan = self.ctx.plan
        if plan is None:
            return None
        signature = (plan.revision, tuple(step.status for step in plan.steps))
        if signature == self._plan_signature:
            self._plan_stall_turns += 1
        else:
            self._plan_signature = signature
            self._plan_stall_turns = 0

        return {
            **self._plan_snapshot(),
            "stalled_turns": self._plan_stall_turns,
            "drift": self._plan_scope_drift(tool_uses),
            "mismatch": self._plan_capability_mismatch(tool_uses),
        }

    def _plan_snapshot(self) -> dict[str, Any]:
        """返回可安全交给客户端与回放存储的计划台账。

        工具和技能提示属于模型内部的执行线索，前端只需用户可读的
        研究目标、步骤、状态与依赖关系。
        """
        plan = self.ctx.plan
        if plan is None:
            return {}
        done, total = plan.progress()
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "revision": plan.revision,
            "done": done,
            "total": total,
            "steps": [
                {
                    "seq": step.seq,
                    "action": step.action,
                    "status": step.status,
                    "dep": list(step.dep),
                }
                for step in plan.steps
            ],
        }

    def _plan_hint_text(self, progress: dict[str, Any]) -> str:
        """渲染追加到下一次请求的软性偏离/不匹配/停滞提示。"""
        parts: list[str] = []
        drift = progress.get("drift") or []
        if drift:
            targets = "、".join(str(item) for item in drift)
            parts.append(
                f"【目标核对】本轮分析了任务未点名的标的（{targets}）："
                "若这些是该任务需要的新方向，请用 research_plan 修订计划并说明其必要性；"
                "若是在扩展范围，请回到任务目标本身，勿让中间结果带偏。"
            )
        mismatch = progress.get("mismatch") or []
        if mismatch:
            used = "、".join(str(item) for item in mismatch)
            parts.append(
                f"【工具核对】本轮使用了计划步骤未涉及的能力（{used}）："
                "若确为任务所需（如计划漏了该类数据），请用 research_plan 修订计划；"
                "若是用错了能力（如查公告却用了新闻），请换回计划指向的工具，"
                "不要用相近但不同类的数据替代。"
            )
        limit = self.settings.context.plan_stall_turns
        if progress.get("stalled_turns", 0) >= limit:
            parts.append(
                f"【停滞提示】计划已连续 {progress['stalled_turns']} 轮无进展"
                f"（{progress['done']}/{progress['total']}）："
                "请用 update_plan_step 回写已完成/已放弃的步骤，"
                "或用 research_plan 修订计划，不要只推进不回写。"
            )
        return "\n".join(parts)

