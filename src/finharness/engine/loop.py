"""会话内的 agent 循环：一次一个模型轮次，且仅使用只读工具。"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from finharness.config.settings import Settings
from finharness.context.compaction import AutoCompactor, CompactionResult
from finharness.context.memory.short_term import Episode, ShortTermMemory
from finharness.context.memory.summary import SummaryLayer
from finharness.context.memory.working import WorkingMemory
from finharness.context.session import Plan, ResearchContext
from finharness.data.cache import make_lookup_key
from finharness.data.citation import CitationRegistry, fingerprint_frame, fingerprint_text
from finharness.engine.cost import SessionStats
from finharness.engine.retry import RetryPolicy, stream_with_retry
from finharness.hooks.base import HookChain
from finharness.observability import NullObserver
from finharness.observability.context import update_turn
from finharness.permissions.gate import ReadOnlyGate
from finharness.permissions.modes import Verdict
from finharness.provider.base import Provider
from finharness.tools.capabilities import (
    Capability,
    UnknownCapabilityError,
    capabilities_in_text,
    capabilities_of,
    capability_of,
    is_research_capability,
)
from finharness.tools.registry import ToolRegistry
from finharness.types import (
    AgentTurnOutcome,
    EngineEvent,
    ModelUsage,
    Msg,
    ObservedCall,
    OutputSink,
    RoundTrace,
    StreamEvent,
    ToolResult,
    ToolUse,
)


class _CompactionMarker:
    """用于描述一次 compaction 的审计记录的替身工具。"""

    name = "context_compaction"


@dataclass(slots=True)
class RepeatVerdict:
    """针对单次 tool 调用的重复检查结果。"""

    count: int
    escalate: bool
    message: str


class LoopDetected(RuntimeError):
    """当同一 tool 调用重复次数超出允许阈值时抛出。"""

    def __init__(self, message: str, *, tool: str = "") -> None:
        super().__init__(message)
        self.tool = tool


class AgentLoop:
    """持有一个会话的对话、usage 与工具预算。"""

    TRUNCATION_MARKER = "\n[truncated]"

    def __init__(
        self,
        *,
        provider: Provider,
        registry: ToolRegistry,
        settings: Settings,
        system: str,
        output: OutputSink | None = None,
        retry_policy: RetryPolicy | None = None,
        stats: SessionStats | None = None,
        cite: CitationRegistry | None = None,
        session_id: str | None = None,
        ctx: ResearchContext | None = None,
        gate: Any | None = None,
        hooks: HookChain | None = None,
        interactive: Any | None = None,
        counter: Any | None = None,
        conversation_id: str | None = None,
        store: Any | None = None,
        user_id: str = "",
        coordinator: Any | None = None,
        observer: Any | None = None,
        call_type: str = "main",
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.system = system
        self.output = output
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.stats = stats if stats is not None else SessionStats()
        self.cite = cite if cite is not None else CitationRegistry()
        self.session_id = session_id or "local"
        # 观测默认关闭：不注入 observer 的调用方（测试替身、eval）行为不变。
        self.observer = observer if observer is not None else NullObserver()
        # 主循环是 "main"，子代理是 "subagent"。它决定 LLM 指标的 call_type，
        # 以及 ``run()`` 是否算作一次用户请求（子代理不算）。
        self.call_type = call_type
        self.ctx = (
            ctx
            if ctx is not None
            else ResearchContext(cite=self.cite, settings=settings)
        )
        # 默认值对不注入任何内容的调用方保持治理引入之前的行为。
        self.gate = gate if gate is not None else ReadOnlyGate()
        self.hooks = hooks if hooks is not None else HookChain()
        self.interactive = interactive
        # sub-agent 协调器（docs 03.10）。按需从 registry 的 DataAccess 构建，
        # 因此不注入任何内容的调用方也能获得风险评审。
        # 记账在下方绑定，此时本循环自身的计数器已存在。
        self.coordinator = (
            coordinator if coordinator is not None else self._build_coordinator(counter=counter)
        )
        # L1 memory 拥有对话记录及其记账；循环负责编排。
        # 可以注入一个 counter，以便调用方共享同一份词表缓存。
        self.memory = WorkingMemory(ctx=self.ctx, settings=settings, counter=counter)
        # 让 ctx.append_user/append_tool_result 转发到这里（docs 03.6.2）。
        self.ctx.memory = self.memory
        # 对话记忆（docs 03.6.4）：对话记录及建立在它之上的记忆按会话持久化，
        # 因此重启后无需重新摘要即可恢复。若没有 store，循环便和以前一样无记忆。
        self.conversation_id = conversation_id or session_id or "local"
        self.store = store
        # 记忆作用域的用户归属（docs 03.13）：对话落库与偏好笔记都按它隔离。
        # 默认空串保持 CLI/eval 直连（无 HTTP 层）时的既有行为。
        self.user_id = user_id
        self.short_term = ShortTermMemory(cap=settings.context.short_mem_cap)
        self.summary = SummaryLayer.load(
            conversation_id=self.conversation_id,
            store=store,
            counter=self.memory.counter,
            budget_tokens=int(
                settings.context.context_window_tokens
                * settings.context.summary_budget_ratio
            ),
        )
        self.ctx.short_term = self.short_term
        self.ctx.summary = self.summary
        self.ctx.store = store
        self.ctx.user_id = self.user_id
        self._memory_loaded = False
        self.usage = ModelUsage()
        # 评审 token 必须出现在会话总量中，因此既然计数器都已存在，就把协调器
        # 接到计数器上。测试替身可能没有该绑定钩子，这无妨——它只是不计费而已。
        binding = getattr(self.coordinator, "bind_accounting", None)
        if binding is not None:
            binding(usage=self.usage, stats=self.stats, observer=self.observer)
        self.turn = 0
        self.compactions: list[CompactionResult] = []
        # 循环防护状态，每次运行重置：完全相同的 (tool, args) 调用重复超过阈值后
        # 不再提供新信息，因此会被拒绝。
        self._call_counts: dict[str, int] = {}
        self._reminded: set[str] = set()
        # 计划进展状态，每次运行重置。``_plan_hint`` 搭载在下一次请求的 state
        # 块中（在一轮结束后设置，这样一轮中的三次测量看到的文本一致）；
        # signature/stall 计数器用于检测计划已停止推进。
        self._plan_hint = ""
        self._plan_signature: tuple[Any, ...] | None = None
        self._plan_stall_turns = 0
        # 本次运行中已被标记为偏离目标的目标，这样单个跑偏的 symbol 只会提示一次，
        # 而不是它在 findings 中每停留一轮就提示一次。
        self._reported_drift: set[str] = set()
        # 本次运行中已被标记为不在计划内的 capability，同样遵循一次性规则。
        self._reported_mismatch: set[str] = set()
        # 当前问题，在 run() 中设置；scope 信号从它读取任务自身的 symbol 与
        # capability，而非计划的重述。
        self._last_user_msg: str = ""
        # 本次运行的 Thought/Action/Observation 记录（docs 03.3）。按运行保存，
        # 以便与单个问题对齐；``_call_meta`` 将每次调用的结果/耗时从
        # ``_execute_one`` 带回轮次记录。
        self.trace: list[RoundTrace] = []
        self.rounds = 0
        self._call_meta: dict[str, ObservedCall] = {}

    @property
    def messages(self) -> list[Msg]:
        """对话记录的只读视图（保留给调用方与测试）。"""
        return self.memory.snapshot()

    def _build_coordinator(self, *, counter: Any | None) -> Any | None:
        """在需要时，从 registry 的数据访问构建一个协调器。

        仅在目录中确实包含请求它的工具时才构建，这使两种情况都正确：传入
        registry 替身的调用方不会获得 sub-agent 机制；评审者自己收窄后的
        registry——其中没有 ``write_report``——无法递归地再构建出第二个评审者。

        延迟导入避免了模块加载时 engine→coordinator 的循环依赖。
        """
        tools = getattr(self.registry, "tools", None)
        if not tools or not any(
            getattr(tool, "needs_coordinator", False) for tool in tools.values()
        ):
            return None
        data = getattr(self.registry, "data", None)
        if data is None:
            return None
        try:
            from finharness.coordinator import Coordinator
        except Exception:  # noqa: BLE001 - 缺少协调器并不致命
            return None
        return Coordinator(
            provider=self.provider,
            data=data,
            settings=self.settings,
            cite=self.cite,
            counter=counter,
        )

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.output is not None:
            await self.output.emit(EngineEvent(kind, data))

    def _on_retry(self, error: BaseException, index: int, delay: float) -> None:
        """provider 重试回调：计入会话统计，并留下一条可排查的日志。"""
        self.stats.add_retry()
        logger = getattr(self.observer, "log", None)
        if logger is not None:
            logger.warning(
                "llm_retry",
                extra={
                    "retry_index": index,
                    "delay_s": round(delay, 3),
                    "turn": self.turn,
                    "error": type(error).__name__,
                },
            )

    def _account_compaction_usage(self, input_tokens: int, output_tokens: int) -> None:
        """把压缩摘要调用的用量并入会话总量。

        与子 Agent 不同，压缩没有独立的 ``per_agent`` 维度——它是会话自身的
        维护成本，因此只累加到总量。
        """
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.stats.add_usage(input_tokens, output_tokens)

    def _system_prompt(self) -> str:
        """静态系统提示词（docs 03.3）。

        刻意不含任何按轮次变化的内容：它在每个请求间逐字节相同，因此 provider
        的 prefix cache 可以将其与不断增长的消息历史一并保留。会话的研究状态
        过去被拼接在这里，这把缓存边界移到了对话最前端，导致每一轮都要重新
        处理整段历史。
        """
        return self.system

    def _state_text(self) -> str:
        """会话的研究状态，为本请求重新渲染（docs 03.6.2）。

        偏离/停滞提示追加在这里，而非存储在 ``ctx`` 上：它和其余状态一样是
        本请求的一个视图，并且在一轮中三次测量 prompt 时必须保持一致。
        """
        state = self.ctx.state_block()
        if self._plan_hint:
            state = f"{state}\n\n{self._plan_hint}" if state else self._plan_hint
        return state

    def _request_messages(self) -> list[Msg]:
        """历史记录，外加作为末尾消息的研究状态。

        状态放在最后而非最前，有两个原因：它让历史记录成为缓存可复用的稳定
        前缀，同时把最新的状态放在模型注意力最强之处。它从不追加到 ``raw``
        ——它是本请求的一个视图，而不是会持久化或被摘要的对话记录条目。
        """
        messages = self.memory.snapshot()
        state = self._state_text()
        if state:
            messages.append(Msg(role="user", content=state))
        return messages

    # -- 对话记忆（docs 03.6.4） ----------------------------------------------
    async def _load_memory_if_first_turn(self, user_msg: str) -> None:
        """在本循环的首次 prompt 之前，一次性加载已持久化的记忆。

        这里加载的所有内容都挂在 ``ctx`` 上，以便后续轮次渲染出相同的文本——
        prompt 每轮要测量三次，这些测量必须一致，而重新查询可能会破坏一致性。
        """
        if self._memory_loaded or self.store is None:
            return
        self._memory_loaded = True

        self.store.ensure_conversation(
            self.conversation_id, user_id=self.user_id, title=_title_from(user_msg)
        )
        # 对话记录，让模型从会话中途续接，而不是从头开始。
        # track=False：重放的历史已经存储过；再缓冲它会以重复的序号再次写入。
        for message in self.store.load_messages(self.conversation_id):
            self.memory.append(message, track=False)
        # Citation 必须保留其 id：已存储的摘要会引用它们。
        self.cite.restore(self.store.load_citations(self.conversation_id))
        self.ctx.prior_conclusions = self.store.load_conclusions(self.conversation_id)
        self.ctx.notes = self.store.get_notes(user_id=self.user_id)
        for symbol in self.store.load_symbols(self.conversation_id):
            self.ctx.remember_symbol(symbol)
        self.summary = SummaryLayer.load(
            conversation_id=self.conversation_id,
            store=self.store,
            counter=self.memory.counter,
            budget_tokens=int(
                self.settings.context.context_window_tokens
                * self.settings.context.summary_budget_ratio
            ),
        )
        self.ctx.summary = self.summary
        self.ctx.refresh_recall()

    def _persist_turn(self) -> None:
        """在一个事务中写入本轮的消息与记忆。

        在轮次结束时批量写入（而非逐条消息）使写入更廉价；崩溃最多损失当前
        这一轮，对于旨在辅助而非审计的进程记忆而言这是可接受的。
        """
        pending = self.memory.pending
        if self.store is None or not pending:
            return
        self.store.append_messages(self.conversation_id, pending)
        self.memory.pending = []
        self.store.save_citations(self.conversation_id, self.cite.all())
        for symbol in self.ctx.symbols:
            self.store.upsert_symbol(self.conversation_id, symbol)
        for conclusion in self.ctx.conclusions:
            self.store.save_conclusion(
                self.conversation_id,
                subject=self._conclusion_subject(conclusion.cids),
                text=conclusion.text,
                cids=conclusion.cids,
            )
        self.store.touch_conversation(self.conversation_id)

    def _conclusion_subject(self, cids: list[str]) -> str:
        """从被引用数据的 symbol 生成回想的 key，使同一事实的 key 保持稳定。"""
        for cid in cids:
            citation = self.cite.get(cid)
            if citation is not None and citation.symbol:
                return str(citation.symbol)
        return self.conversation_id

    def _remember_episode(
        self, *, kind: str, subject: str, summary: str, ref: dict | None = None
    ) -> None:
        """为 L2 回想记录一个结构化事件。"""
        if not subject:
            return
        self.short_term.add(
            Episode(kind=kind, subject=subject, summary=summary, ref=dict(ref or {}))
        )
        # 让 symbol 池与实际抓取过的内容保持一致。仅从 citation 推导会漏掉
        # 未产生 citation 的抓取（纯文本 payload），使回想没有可供检索的 subject。
        if kind == "data":
            self.ctx.remember_symbol(subject)
        self.ctx.refresh_recall()

    async def _maybe_compact(self) -> None:
        """当下一次请求将超出窗口预算时，对历史进行折叠压缩。"""
        compactor = AutoCompactor(
            provider=self.provider,
            memory=self.memory,
            settings=self.settings,
            system=self._system_prompt(),
            tools=self.registry.schemas(),
            summary=self.summary,
            state_text=self._state_text(),
            observer=self.observer,
            # 摘要调用的 token 计入会话总量，使成本视图完整。
            on_usage=lambda i, o: self._account_compaction_usage(i, o),
        )
        if not compactor.needs_compaction():
            return
        result = await compactor.compact()
        if not result.compacted and result.warning is None:
            return
        self.compactions.append(result)
        await self._emit(
            "context_compacted",
            {
                "removed": result.removed,
                "before_tokens": result.before_tokens,
                "after_tokens": result.after_tokens,
                "degraded": result.degraded,
                "warning": result.warning,
                "duration_ms": result.duration_ms,
            },
        )
        # Compaction 必须出现在审计轨迹中（docs 4.3，action=compact）。
        if self.hooks.hooks:
            try:
                await self.hooks.post(
                    _CompactionMarker(),
                    {},
                    ToolResult(content="", ok=True),
                    action="compact",
                    verdict="allow",
                    duration_ms=result.duration_ms,
                    turn=self.turn,
                )
            except Exception:  # noqa: BLE001 - 审计是尽力而为
                pass

    async def _audit_detection(self, detected: LoopDetected) -> None:
        """在审计轨迹中记录该中止（尽力而为，绝不致命）。"""
        if not self.hooks.hooks:
            return
        try:
            await self.hooks.post(
                _CompactionMarker(),
                {"tool": detected.tool},
                ToolResult(content="", ok=False, error=str(detected)),
                action="loop_detected",
                verdict="abort",
                turn=self.turn,
            )
        except Exception:  # noqa: BLE001 - 审计是尽力而为
            pass

    def _window_tokens(self) -> int:
        return self.memory.request_tokens(
            system=self._system_prompt(),
            tools=self.registry.schemas(),
            extra_text=self._state_text(),
        )

    # -- 循环防护 -------------------------------------------------------------
    def _call_fingerprint(self, tool_use: ToolUse) -> str:
        """(tool, args) 对的稳定标识；相同的 key 意味着相同的结果。

        复用缓存的 key 构建器，使参数归一化（排序、JSON 安全渲染）与数据层
        处理调用的既有方式保持一致。
        """
        return make_lookup_key(kind=tool_use.name, params=dict(tool_use.args or {}))

    def _check_repeat(self, tool_use: ToolUse) -> RepeatVerdict | None:
        """统计一次调用，并决定是提醒还是升级处理。

        计数在整次运行中累计：A/B/A/B 的模式也是一种循环，而且重复的相同调用
        永远不提供新信息，因为其结果已经可用。
        """
        limit = self.settings.context.max_identical_tool_calls
        key = self._call_fingerprint(tool_use)
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        count = self._call_counts[key]
        if count < limit:
            return None

        complex_task = self.ctx.plan is not None
        if key not in self._reminded:
            # 首次违规：提醒，让模型自行纠正。
            self._reminded.add(key)
            if complex_task:
                message = (
                    f"相同参数的 {tool_use.name} 已调用 {count} 次，结果已在上文。"
                    "若当前方向行不通，请用 research_plan 修订计划后继续，不要重复取数。"
                )
            else:
                message = (
                    f"相同参数的 {tool_use.name} 已调用 {count} 次，结果已在上文（或对应 citation），"
                    "请直接复用该结果作答，不要重复取数。"
                )
            return RepeatVerdict(count=count, escalate=False, message=message)

        # 同一调用形态的第二次违规：停止；本次运行无法继续推进。
        return RepeatVerdict(
            count=count,
            escalate=True,
            message=f"相同参数的 {tool_use.name} 重复调用 {count} 次，已中止本轮。",
        )

    # 运行停止的原因，以面向读者的措辞表述。每一种不成功的终态都会配上一份
    # 部分答案，因此原因必须点明真正的起因：用“检测到重复调用”来描述轮次预算
    # 耗尽就是错的。
    _STOP_REASONS = {
        "loop_detected": "检测到重复调用",
        "max_turns_exhausted": "达到轮次上限",
        "provider_error": "模型调用失败",
    }

    def _partial_answer(self, reason: str = "loop_detected") -> str:
        """总结本次运行在停止前设法确立的内容。

        在每一种不成功的终态都会调用，因此失败的运行绝不会空手而归：它形成的
        结论和抓取的数据，是已经付出成本的轮次所留下的有用残余。
        """
        citations = self.cite.all()
        conclusions = self.ctx.conclusions
        cause = self._STOP_REASONS.get(reason, reason)
        lines = [f"本轮已提前结束（{cause}）。"]
        if conclusions:
            lines.append("已形成的结论：")
            lines.extend(
                f"- {item.text}（依据 {'、'.join(item.cids) or '无'}）"
                for item in conclusions[-5:]
            )
        if citations:
            lines.append(f"已获取 {len(citations)} 份数据，可用 citations 精读。")
        if len(lines) == 1:
            lines.append("尚未形成可复述的结论，未能完成该请求。")
        return "\n".join(lines)

    def _has_partial_findings(self) -> bool:
        """停止时是否产生了值得交还的内容。"""
        return bool(self.cite.all() or self.ctx.conclusions)

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
        done, total = plan.progress()

        signature = (plan.revision, tuple(step.status for step in plan.steps))
        if signature == self._plan_signature:
            self._plan_stall_turns += 1
        else:
            self._plan_signature = signature
            self._plan_stall_turns = 0

        return {
            "revision": plan.revision,
            "done": done,
            "total": total,
            "stalled_turns": self._plan_stall_turns,
            "drift": self._plan_scope_drift(tool_uses),
            "mismatch": self._plan_capability_mismatch(tool_uses),
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

    def _done_payload(
        self, *, succeeded: bool, reason: str | None, tool_calls: int
    ) -> dict[str, Any]:
        snapshot = self.stats.snapshot()
        plan = self.ctx.plan
        plan_payload = (
            {
                "revision": plan.revision,
                "done": plan.progress()[0],
                "total": plan.progress()[1],
            }
            if plan is not None
            else None
        )
        return {
            "succeeded": succeeded,
            "reason": reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                # 上方输入的 prefix-cache 拆分，前提是 provider 报告它；
                # 该优化的效果只能在这里核查。
                "cache_hit_tokens": snapshot.cache_hit_tokens,
                "cache_miss_tokens": snapshot.cache_miss_tokens,
                "cache_hit_ratio": self.stats.cache_hit_ratio(),
            },
            # 与上方的累计 usage 不同：这是当前窗口的充满程度，
            # compaction 本应降低它。
            "window_tokens": self._window_tokens(),
            "compactions": len(self.compactions),
            "tool_calls": tool_calls,
            # 实际经历的轮数：效率评估读取的步数，与 tool_calls 不同，
            # 因为一轮可能调用零个或多个工具。
            "rounds": self.rounds,
            "retry_count": snapshot.retry_count,
            "tool_duration_ms": snapshot.tool_duration_ms,
            # sub-agent 开销，已包含在上方的 ``usage`` 中；这里是明细拆分，
            # 便于客户端展示 token 的去向（docs 03.10）。
            "per_agent": {name: dict(entry) for name, entry in snapshot.per_agent.items()},
            "citations": [item.cid for item in self.cite.all()],
            # 运行结束时的计划进展：让重新规划对客户端可见（revision），
            # 并展示计划完成了多少。
            "plan": plan_payload,
        }

    def _citation_ids(self) -> list[str]:
        return [item.cid for item in self.cite.all()]

    async def _fail(
        self,
        *,
        kind: str,
        message: str,
        reason: str,
        tool_calls: int,
        error: str | None,
        answer: str = "",
    ) -> AgentTurnOutcome:
        """以不成功的方式结束本次运行；``answer`` 可能携带部分发现。"""
        await self._emit("error", {"kind": kind, "message": message, "reason": reason})
        if answer:
            # 运行失败了，但并非空手而归——把发现的内容呈现出来。
            await self._emit("answer", {"text": answer})
            # 也将其持久化：客户端只在事件中看到的答案在重载后会消失，因此
            # 部分发现也写为本轮的 assistant 消息。``run()`` 在我们返回后
            # 刷新该缓冲区。
            self.memory.append_assistant(Msg(role="assistant", content=answer))
        await self._emit(
            "done",
            self._done_payload(succeeded=False, reason=reason, tool_calls=tool_calls),
        )
        return AgentTurnOutcome(
            answer=answer,
            succeeded=False,
            usage=self.usage,
            error=error,
            reason=reason,
            tool_calls=tool_calls,
            retry_count=self.stats.retry_count,
            tool_duration_ms=self.stats.snapshot().tool_duration_ms,
            citations=self._citation_ids(),
            trace=list(self.trace),
            rounds=self.rounds,
        )

    async def run(self, user_msg: str) -> AgentTurnOutcome:
        """运行一次完整的 agent 轮次循环，返回本轮结果。

        重置各项按运行维护的状态，在需要时加载持久化记忆，然后交由
        ``_run_turns`` 逐步执行模型调用与工具调用，最后持久化本轮消息。

        本方法持有一轮请求的根 Span：它是所有入口（服务端、eval、脚本）共同
        的执行边界，因此请求级耗时只在这里统计一次，不会重复计数。
        """
        model = getattr(self.provider, "model", "") or ""
        with self.observer.request_span(
            message=user_msg,
            model=model,
            emit_metric=self.call_type == "main",
        ) as span:
            outcome = await self._run_once(user_msg)
            # 根 Span 的成功状态取自引擎的结论，而不是"没有抛异常"。
            span.set_attribute("status", "ok" if outcome.succeeded else "error")
            span.set_attribute("reason", outcome.reason or "ok")
            span.set_attribute("tool_calls", outcome.tool_calls)
            span.set_attribute("rounds", outcome.rounds)
            span.set_attribute(
                "usage",
                {
                    "input_tokens": outcome.usage.input_tokens,
                    "output_tokens": outcome.usage.output_tokens,
                },
            )
            if not outcome.succeeded:
                self.observer.record_error(
                    error_type=outcome.reason or "error", detail=outcome.error or ""
                )
            return outcome

    async def _run_once(self, user_msg: str) -> AgentTurnOutcome:
        """``run`` 的实际执行体；根 Span 由调用方持有。"""
        # 记住该问题：scope/偏离信号从它读取任务自身的 symbol 与 capability，
        # 这才是事实依据，而非计划可能不完整的重述。
        self._last_user_msg = user_msg or ""
        # 在对话首轮，于构建 prompt 之前加载已持久化的记忆。后续轮次复用
        # 它（见 ctx）而非重新查询：系统提示词每轮要测量三次，这些测量必须一致。
        await self._load_memory_if_first_turn(user_msg)
        # 在追加之前打开本轮的写缓冲区，这样用户消息会被缓存以供持久化，
        # 而不会被重置操作丢弃。
        self.memory.pending = []
        if user_msg:
            self.memory.append_user(user_msg)

        # 按运行维护的状态：轮次预算按请求计算，因此计数器随之重置。
        self.turn = 0
        self._call_counts = {}
        self._reminded = set()
        self._plan_hint = ""
        self._plan_signature = None
        self._plan_stall_turns = 0
        self._reported_drift = set()
        self._reported_mismatch = set()
        # 轨迹记录按问题保存，与其他按运行维护的状态一样。
        self.trace = []
        self.rounds = 0
        self._call_meta = {}

        outcome = await self._run_turns()
        self._persist_turn()
        return outcome

    async def _run_turns(self) -> AgentTurnOutcome:
        """驱动模型轮次直到产出最终答案、耗尽轮次预算或触发中止。

        每轮做窗口维护、流式请求模型、执行工具调用并记录轨迹；某个调用触发
        循环防护时以部分答案提前结束，轮次耗尽时同样如此。
        """
        tool_calls_total = 0
        model = getattr(self.provider, "model", "") or ""
        # 完整 Prompt 只在明确要求时构造，避免默认开启追踪时白算一遍消息历史。
        capture_messages = bool(getattr(self.observer, "tracing_capture_payloads", False))
        for _ in range(self.settings.context.max_turns):
            self.turn += 1
            update_turn(self.turn)
            # 窗口维护发生在轮次之间，绝不在请求中途，并且即使摘要失败也不能
            # 中断对话。
            await self._maybe_compact()
            deltas: list[str] = []
            tool_uses: list[ToolUse] = []
            # 为轨迹记录按轮捕获耗时/token。时钟在请求之前启动，因此首 token
            # 延迟是按轮计算的，而非按运行；token 是本轮的，与模型看到的一致。
            round_started = self.stats.now()
            first_token_at: float | None = None
            round_input = 0
            round_output = 0
            # LLM 子 Span 必须包住整个 ``async for`` 消费过程，而不只是
            # provider.stream() 返回的生成器对象——后者在第一次迭代前不会发起
            # 请求，越过消费边界关 Span 会得到 0 耗时。
            async with self.observer.llm_span(
                model=model,
                call_type=self.call_type,
                messages=self._request_messages() if capture_messages else None,
                tool_count=len(self.registry.schemas()),
            ) as llm_span:
                try:
                    async for chunk in stream_with_retry(
                        lambda: self.provider.stream(
                            system=self._system_prompt(),
                            messages=self._request_messages(),
                            tools=self.registry.schemas(),
                            usage=self.usage,
                        ),
                        policy=self.retry_policy,
                        on_retry=lambda error, index, delay: self._on_retry(error, index, delay),
                    ):
                        if chunk.event is StreamEvent.TEXT_DELTA:
                            # 每个 delta 一到就流式输出，使答案在生成过程中逐步显现，
                            # 而不是在结束时一次性涌出。仍然保留缓冲区，以组装
                            # 最终答案和持久化的消息。
                            if first_token_at is None:
                                first_token_at = self.stats.now()
                                llm_span.mark_first_token()
                            deltas.append(chunk.data)
                            await self._emit("text_delta", {"text": chunk.data})
                        elif chunk.event is StreamEvent.MESSAGE_END and isinstance(
                            chunk.data, ModelUsage
                        ):
                            tool_uses = list(chunk.data.tool_uses)
                            round_input += chunk.data.input_tokens
                            round_output += chunk.data.output_tokens
                            self.usage.input_tokens += chunk.data.input_tokens
                            self.usage.output_tokens += chunk.data.output_tokens
                            self.usage.cache_hit_tokens += chunk.data.cache_hit_tokens
                            self.usage.cache_miss_tokens += chunk.data.cache_miss_tokens
                            self.stats.add_usage(
                                chunk.data.input_tokens,
                                chunk.data.output_tokens,
                                cache_hit_tokens=chunk.data.cache_hit_tokens,
                                cache_miss_tokens=chunk.data.cache_miss_tokens,
                            )
                            # token 指标的唯一权威发射点；Span 结束时统一上报，
                            # 避免主循环与子 Agent 两处各发一次导致双计。
                            llm_span.set_usage(chunk.data)
                except Exception as exc:
                    # 本轮的请求从未完成；记录它实际产出的内容，使失败运行的轨迹
                    # 能显示失败点。
                    llm_span.fail(str(exc))
                    self._record_round(
                        thought="".join(deltas),
                        actions=[],
                        results=[],
                        input_tokens=round_input,
                        output_tokens=round_output,
                        llm_first_ms=(
                            round((first_token_at - round_started) * 1000)
                            if first_token_at is not None
                            else 0
                        ),
                        llm_ms=round((self.stats.now() - round_started) * 1000),
                    )
                    return await self._fail(
                        kind=type(exc).__name__,
                        message=str(exc),
                        reason="provider_error",
                        tool_calls=tool_calls_total,
                        error=str(exc),
                        # 交还早先轮次已确立的内容；第 N 轮发生 provider 失败
                        # 绝不能丢弃第 1..N-1 轮。
                        answer=(
                            self._partial_answer(reason="provider_error")
                            if self._has_partial_findings()
                            else ""
                        ),
                    )

            stream_ended = self.stats.now()
            llm_first_ms = (
                round((first_token_at - round_started) * 1000)
                if first_token_at is not None
                else 0
            )
            llm_ms = round((stream_ended - round_started) * 1000)

            if tool_uses:
                # 本轮结果是一次工具调用，因此此前流出的任何文本都只是草稿，
                # 而非答案。在展示工具活动之前告知客户端丢弃它；最终答案会在
                # 后续某一轮流式输出。
                if deltas:
                    await self._emit("text_reset", {})
                self.memory.append_assistant(
                    Msg(role="assistant", content=None, tool_uses=tool_uses)
                )
                try:
                    results = list(
                        await asyncio.gather(
                            *(self._execute_one(tool_use) for tool_use in tool_uses)
                        )
                    )
                except LoopDetected as detected:
                    # 为每个调用配对，使对话记录保持格式良好，然后带着已经确立
                    # 的内容结束本次运行。
                    aborted = self._aborted_results(tool_uses)
                    self.memory.append(
                        Msg(role="tool_result", content=None, tool_results=aborted)
                    )
                    # 记录触发中止的这一轮：是哪个调用触发了防护、带着什么参数，
                    # 这是核心的轨迹观测。
                    self._record_round(
                        thought="".join(deltas),
                        actions=tool_uses,
                        results=aborted,
                        input_tokens=round_input,
                        output_tokens=round_output,
                        llm_first_ms=llm_first_ms,
                        llm_ms=llm_ms,
                    )
                    await self._audit_detection(detected)
                    return await self._fail(
                        kind="loop_detected",
                        message=str(detected),
                        reason="loop_detected",
                        tool_calls=tool_calls_total,
                        error=str(detected),
                        answer=self._partial_answer(reason="loop_detected"),
                    )
                except BaseException:
                    # 被中止的一轮绝不能让 assistant 帧缺少配对的 tool 消息，
                    # 否则下一次请求格式错误。
                    self.memory.append(
                        Msg(
                            role="tool_result",
                            content=None,
                            tool_results=self._aborted_results(tool_uses),
                        )
                    )
                    raise
                self.memory.append(
                    Msg(role="tool_result", content=None, tool_results=results)
                )
                tool_calls_total += len(results)
                self._record_round(
                    thought="".join(deltas),
                    actions=tool_uses,
                    results=results,
                    input_tokens=round_input,
                    output_tokens=round_output,
                    llm_first_ms=llm_first_ms,
                    llm_ms=llm_ms,
                )
                # 计划进展在刚运行完的这一轮上评估，然后搭载到下一次请求。
                # 它是一个信号，绝不是闸门：一轮绝不会因为偏离计划而被拒绝。
                progress = self._plan_progress(tool_uses)
                if progress is not None:
                    self._plan_hint = self._plan_hint_text(progress)
                    await self._emit("plan_progress", progress)
                continue

            answer = "".join(deltas)
            self._record_round(
                thought="",
                actions=[],
                results=[],
                input_tokens=round_input,
                output_tokens=round_output,
                llm_first_ms=llm_first_ms,
                llm_ms=llm_ms,
                answer=answer,
            )
            await self._emit("answer", {"text": answer})
            await self._emit(
                "done",
                self._done_payload(
                    succeeded=True, reason=None, tool_calls=tool_calls_total
                ),
            )
            self.memory.append_assistant(Msg(role="assistant", content=answer))
            return AgentTurnOutcome(
                answer=answer,
                usage=self.usage,
                tool_calls=tool_calls_total,
                retry_count=self.stats.retry_count,
                tool_duration_ms=self.stats.snapshot().tool_duration_ms,
                citations=self._citation_ids(),
                trace=list(self.trace),
                rounds=self.rounds,
            )

        return await self._fail(
            kind="max_turns_exhausted",
            message="maximum agent turns exhausted",
            reason="max_turns_exhausted",
            tool_calls=tool_calls_total,
            error=None,
            # 预算耗尽并不等于空跑：报告它已经确立的内容。
            answer=self._partial_answer(reason="max_turns_exhausted"),
        )

    def _truncate(self, content: str) -> str:
        """将单个工具结果限制在 ``context.max_result_tokens`` 个 token 以内。

        该设置以 token 为单位命名，因此截断依据的是真实的 token 计数而非字符
        长度（字符上限会让中文文本膨胀到约两倍的预期预算）。
        """
        limit = self.settings.context.max_result_tokens
        if self.memory.counter.count(content).tokens <= limit:
            return content
        # 二分查找预算内最长的前缀；计数是单调的。
        low, high = 0, len(content)
        while low < high:
            middle = (low + high + 1) // 2
            if self.memory.counter.count(content[:middle]).tokens <= limit:
                low = middle
            else:
                high = middle - 1
        prefix = content[:low]
        if limit <= 4:  # 没有空间放置标记
            return prefix
        return prefix.rstrip() + self.TRUNCATION_MARKER

    def _encode(self, result: ToolResult) -> str:
        payload: dict[str, Any] = {
            "ok": bool(result.ok),
            "content": self._truncate(result.content),
            "error": result.error,
        }
        if result.citations:
            payload["citations"] = list(result.citations)
        return json.dumps(payload, ensure_ascii=False)

    # -- 轨迹记录（docs 03.3） ------------------------------------------------
    # 一轮 = 一次模型调用，加上（若有）它的工具调用及其结果。以有界的小记录
    # 列表保存，使评估可以在不携带完整对话记录的情况下，按预期路径重放本次运行。
    OBSERVATION_PREVIEW_CHARS = 200

    def _note_call(
        self,
        tool_use: ToolUse,
        *,
        ok: bool,
        error: str | None = None,
        content: str = "",
        duration_ms: int = 0,
    ) -> None:
        """记录单次调用的可观测结果，无论其是已执行还是被拒绝。"""
        self._call_meta[tool_use.call_id] = ObservedCall(
            call_id=tool_use.call_id,
            name=tool_use.name,
            ok=ok,
            error=error,
            preview=content[: self.OBSERVATION_PREVIEW_CHARS],
            duration_ms=duration_ms,
        )

    def _record_round(
        self,
        *,
        thought: str,
        actions: list[ToolUse],
        results: list[tuple[str, str]],
        input_tokens: int = 0,
        output_tokens: int = 0,
        llm_first_ms: int = 0,
        llm_ms: int = 0,
        answer: str = "",
    ) -> None:
        """向本次运行的 trace 追加一个 Thought/Action/Observation 轮次。

        工具轮次从答案中丢弃的文本仍会作为 ``thought`` 保存在这里：它是模型
        陈述的推理过程，也是轨迹值得与答案分开记录的原因。观测从 ``_call_meta``
        读取，``_execute_one``/``_reject`` 即便对从未执行的调用（被拒绝、惰性、
        受循环防护）也会填充该表。
        """
        self.rounds += 1
        observations: list[ObservedCall] = []
        for call_id, rendered in results:
            meta = self._call_meta.get(call_id)
            if meta is None:
                ok, error, preview = self._decode_observation(rendered)
                meta = ObservedCall(
                    call_id=call_id, name="", ok=ok, error=error, preview=preview
                )
            observations.append(meta)
        self.trace.append(
            RoundTrace(
                turn=self.rounds,
                thought=thought,
                actions=list(actions),
                observations=observations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                llm_first_ms=llm_first_ms,
                llm_ms=llm_ms,
                answer=answer,
            )
        )

    @staticmethod
    def _decode_observation(rendered: str) -> tuple[bool, str | None, str]:
        """从编码后的工具结果中读回 ok/error/preview。"""
        limit = AgentLoop.OBSERVATION_PREVIEW_CHARS
        try:
            payload = json.loads(rendered)
        except (ValueError, TypeError):
            return True, None, str(rendered)[:limit]
        if not isinstance(payload, dict):
            return True, None, str(payload)[:limit]
        error = payload.get("error")
        return (
            bool(payload.get("ok", True)),
            str(error) if error else None,
            str(payload.get("content") or "")[:limit],
        )

    def _aborted_results(self, tool_uses: list[ToolUse]) -> list[tuple[str, str]]:
        """占位失败结果，使被取消的一轮仍能为每个 call id 配对。"""

        return [
            (
                tool_use.call_id,
                json.dumps(
                    {
                        "ok": False,
                        "content": "",
                        "error": f"tool call cancelled: {tool_use.name}",
                    },
                    ensure_ascii=False,
                ),
            )
            for tool_use in tool_uses
        ]

    async def _reject(
        self, tool_use: ToolUse, message: str, *, duration_ms: int | None = None
    ) -> tuple[str, str]:
        """拒绝一次工具调用：发送失败状态、记录观测并返回编码后的结果。"""
        status: dict[str, Any] = {
            "call_id": tool_use.call_id,
            "name": tool_use.name,
            "status": "failed",
            "ok": False,
            "error": message,
        }
        if duration_ms is not None:
            status["duration_ms"] = duration_ms
        await self._emit("tool_status", status)
        # 拒绝（被拒、惰性未激活、受循环防护、超时）同样是观测：只记录执行的
        # 轨迹无法区分一个被阻断的高风险调用和一个从未尝试过的调用。
        self._note_call(
            tool_use, ok=False, error=message, duration_ms=int(duration_ms or 0)
        )
        return tool_use.call_id, json.dumps(
            {"ok": False, "content": "", "error": message}, ensure_ascii=False
        )

    async def _execute_one(self, tool_use: ToolUse) -> tuple[str, str]:
        """执行一次工具调用，工具 Span 由本方法持有，内部实现见 ``_execute_inner``。

        状态标签在 Span 上累积（ok/error/timeout/denied/blocked/unknown），
        退出时一次性上报 ``agent_tool_calls_total`` 与耗时直方图。
        """
        self.observer.log_tool_call(
            tool=tool_use.name, args=tool_use.args, turn=self.turn
        )
        async with self.observer.tool_span(
            name=tool_use.name, args=tool_use.args, call_id=tool_use.call_id
        ) as span:
            try:
                result = await self._execute_inner(tool_use, span)
            finally:
                # 循环防护升级会抛出 LoopDetected；即便那样也要留下这次调用的
                # 计数，否则"触发中止的那次调用"将从指标里消失。
                self.observer.tool_finished(
                    tool=tool_use.name,
                    status=span.attributes.get("status", "error"),
                    span=span,
                )
        return result

    async def _execute_inner(self, tool_use: ToolUse, span: Any) -> tuple[str, str]:
        """执行一次工具调用，返回 (call_id, 编码后的结果)。

        依次经过解析工具、惰性激活闸门、循环防护、权限与 hook 治理链，再以
        超时保护运行工具；任何一步的拒绝或失败都会生成配对的失败结果，
        而不是抛出异常（循环防护的升级除外）。
        """
        self.stats.record_tool_request(tool_use.name)
        tool = self.registry.resolve(tool_use.name)
        if tool is None:
            span.set_attribute("status", "unknown")
            return await self._reject(tool_use, f"unknown tool: {tool_use.name}")

        # 惰性激活闸门：惰性工具的 schema 在被激活之前从未注入，因此现在调用
        # 它意味着模型绕过了两阶段规则。在结构上拒绝并指出修复方式，而不是
        # 去执行一个模型从未见过的工具。仅在 registry 确实对工具分层时才应用
        # （收窄后的 sub-agent registry 不携带任何分层）。
        lazy = getattr(self.registry, "lazy_names", None)
        is_active = getattr(self.registry, "is_active", None)
        if (
            callable(lazy)
            and callable(is_active)
            and tool_use.name in set(lazy())
            and not is_active(tool_use.name)
        ):
            span.set_attribute("status", "not_activated")
            return await self._reject(
                tool_use,
                f"tool not activated: {tool_use.name}；请先用 load_tool 激活，"
                "下一轮才能调用。",
            )

        # 循环防护：超过阈值的相同调用无法增加信息（其结果已在对话记录和缓存
        # 中），因此用一次提醒拒绝它，而不是再次执行。
        guard = self._check_repeat(tool_use)
        if guard is not None:
            await self._emit(
                "loop_guard",
                {
                    "call_id": tool_use.call_id,
                    "name": tool_use.name,
                    "count": guard.count,
                    "action": "refused" if not guard.escalate else "would_abort",
                },
            )
            if guard.escalate:
                # 该调用形态的第二次违规：停止本次运行，而不是继续为一轮无法
                # 推进的调用付费。
                self._note_call(tool_use, ok=False, error=guard.message)
                raise LoopDetected(
                    f"tool {tool_use.name} repeated with identical arguments "
                    f"{guard.count} times"
                )
            self._note_call(tool_use, ok=False, error=guard.message)
            span.set_attribute("status", "loop_guard")
            return tool_use.call_id, json.dumps(
                {"ok": False, "content": "", "error": guard.message}, ensure_ascii=False
            )

        # 治理链（docs 03.3.3）：先做权限判定，再执行 pre-hooks。
        decision = await self.gate.check(tool, tool_use.args)
        if decision.verdict is Verdict.DENY:
            await self._audit(tool, tool_use.args, action="denied", verdict="deny")
            span.set_attribute("status", "denied")
            return await self._reject(
                tool_use, decision.reason or f"tool denied: {tool_use.name}"
            )
        if not await self.hooks.pre(tool, tool_use.args, turn=self.turn):
            await self._audit(tool, tool_use.args, action="denied", verdict="blocked")
            span.set_attribute("status", "blocked")
            return await self._reject(
                tool_use, f"tool blocked by hook: {tool_use.name}"
            )

        await self._emit(
            "tool_status",
            {"call_id": tool_use.call_id, "name": tool_use.name, "status": "started"},
        )
        # 运维方的覆盖优先；否则使用工具声明的预算，若工具未声明则回退到
        # 全局默认值（docs 03.3.3）。
        default_timeout = self.settings.tools.timeout_default_s
        timeout = self.settings.tools.timeout_overrides.get(
            tool_use.name, tool.timeout or default_timeout
        )
        started_at = self.stats.now()
        if getattr(tool, "needs_interactive", False) and self.interactive is not None:
            tool.interactive = self.interactive
        # 对派发 sub-agent 的工具采用同样的模式（docs 03.10）：工具自身无法
        # 构建协调器，因此由循环把会话的协调器交给它。
        if getattr(tool, "needs_coordinator", False) and self.coordinator is not None:
            tool.coordinator = self.coordinator
        try:
            result = await asyncio.wait_for(tool.run(**tool_use.args), timeout)
        except TimeoutError:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            span.set_attribute("status", "timeout")
            await self._audit(
                tool,
                tool_use.args,
                action="run",
                verdict=decision.verdict.value,
                ok=False,
                duration_ms=duration_ms,
            )
            return await self._reject(
                tool_use,
                f"tool timeout after {timeout}s: {tool_use.name}",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
            span.set_attribute("status", "error")
            await self._audit(
                tool,
                tool_use.args,
                action="run",
                verdict=decision.verdict.value,
                ok=False,
                duration_ms=duration_ms,
            )
            return await self._reject(
                tool_use, f"tool failed: {exc}", duration_ms=duration_ms
            )

        duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
        span.set_attribute("status", "ok" if result.ok else "error")
        result.citations = self._register_citations(tool_use, result)
        await self._audit(
            tool,
            tool_use.args,
            action="run",
            verdict=decision.verdict.value,
            ok=bool(result.ok),
            duration_ms=duration_ms,
            citations=result.citations,
            result=result,
        )
        await self._emit(
            "tool_status",
            {
                "call_id": tool_use.call_id,
                "name": tool_use.name,
                "status": "completed" if result.ok else "failed",
                "ok": bool(result.ok),
                "duration_ms": duration_ms,
                "citations": list(result.citations),
                # 产出的文件（图表、报告）一并附上，使客户端无需二次查询
                # 即可提供它们。
                "attachments": list(result.attachments),
            },
        )
        self._note_call(
            tool_use,
            ok=bool(result.ok),
            error=result.error,
            content=result.content,
            duration_ms=duration_ms,
        )
        return tool_use.call_id, self._encode(result)

    async def _audit(
        self,
        tool,
        args: dict,
        *,
        action: str,
        verdict: str,
        ok: bool = True,
        duration_ms: float = 0.0,
        citations: list[str] | None = None,
        result: ToolResult | None = None,
    ) -> None:
        """发出审计记录；治理失败绝不能中断一轮。"""
        if not self.hooks.hooks:
            return
        endpoint, rows, cols = "", 0, 0
        sources = getattr(result, "sources", None) if result is not None else None
        if sources:
            first = sources[0]
            endpoint = getattr(first, "endpoint", "") or ""
            df = getattr(first, "df", None)
            if df is not None:
                rows, cols = int(len(df)), int(len(df.columns))
        try:
            await self.hooks.post(
                tool,
                args,
                result if result is not None else ToolResult(content="", ok=ok),
                action=action,
                verdict=verdict,
                duration_ms=duration_ms,
                citations=list(citations or []),
                turn=self.turn,
                endpoint=endpoint,
                rows=rows,
                cols=cols,
            )
        except Exception:  # noqa: BLE001 - 审计是尽力而为
            pass

    def _register_citations(self, tool_use: ToolUse, result: ToolResult) -> list[str]:
        """将工具的原始 payload 转换为会话跟踪的 citation id。"""
        cids: list[str] = []
        symbol = (
            tool_use.args.get("symbol") if isinstance(tool_use.args, dict) else None
        )
        for source in getattr(result, "sources", []) or []:
            df = getattr(source, "df", None)
            citation = self.cite.register(
                tool=tool_use.name,
                endpoint=getattr(source, "endpoint", "") or tool_use.name,
                symbol=str(symbol) if symbol is not None else None,
                params=dict(getattr(source, "params", {}) or {}),
                rows=int(len(df)) if df is not None else 0,
                cols=int(len(df.columns)) if df is not None else 0,
                # 纯文本 payload 没有可供摘要的 frame，因此直接对文本本身做
                # 哈希；否则每个文本 citation 都会共享同一个指纹。
                fingerprint=(
                    fingerprint_frame(df)
                    if df is not None
                    else fingerprint_text(getattr(source, "text", None))
                ),
                from_cache=bool(getattr(source, "from_cache", False)),
                parquet_path=getattr(source, "parquet_path", None),
            )
            cids.append(citation.cid)
        # L2：记录这份数据已被抓取，并附带指向它的指针，以便后续的跟进能复用它
        # 而非重新抓取。按调用记录而非按 source 记录，因为即使 payload 是文本
        # （没有 DataFrame）而非表格，抓取也确实发生过。
        if symbol and result.ok:
            self._remember_episode(
                kind="data",
                subject=str(symbol),
                summary=f"{tool_use.name} 已取数"
                + (f"（{len(cids)} 份）" if cids else ""),
                ref={"cids": list(cids)},
            )
        return cids


def _title_from(user_msg: str) -> str | None:
    """从对话的开场消息推导出一个对话标题。"""
    text = (user_msg or "").strip()
    if not text:
        return None
    return text[:40]
