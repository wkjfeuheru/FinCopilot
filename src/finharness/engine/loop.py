"""会话内的 agent 循环：一次一个模型轮次，且仅使用只读工具。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from finharness.config.settings import Settings
from finharness.context.compaction import AutoCompactor, CompactionResult
from finharness.context.memory.short_term import Episode, ShortTermMemory
from finharness.context.memory.state_store import MemoryStateStore, SqliteAgentStateStore
from finharness.context.memory.store import MemoryStore
from finharness.context.memory.summary import SummaryLayer
from finharness.context.memory.working import WorkingMemory
from finharness.context.session import Plan, ResearchContext
from finharness.data.cache import make_lookup_key
from finharness.data.citation import (
    CitationRegistry,
    fingerprint_frame,
    fingerprint_text,
)
from finharness.engine.cost import SessionStats
from finharness.engine.machine import AgentStateMachine
from finharness.engine.plan_progress import PlanProgressMixin
from finharness.engine.retry import RetryPolicy, stream_with_retry
from finharness.engine.runner import AgentRunner, EffectResult, RunnerEffects
from finharness.engine.state import (
    AgentPhase,
    AgentState,
    CallStatus,
    CompactionFinished,
    ConfirmationRequested,
    ConfirmationResolved,
    HydrationFinished,
    ModelFinished,
    ResumeRequested,
    RunFailed,
    StopRequested,
    ToolBatchFinished,
    ToolCallFinished,
    ToolCallStarted,
    UsageDelta,
    new_agent_state,
)
from finharness.hooks.base import HookChain
from finharness.observability import NullObserver
from finharness.observability.context import update_turn
from finharness.observability.logs import get_logger
from finharness.permissions.gate import ConfirmationSpec, GateDecision, ReadOnlyGate
from finharness.permissions.modes import Verdict
from finharness.provider.base import Provider
from finharness.shared.budget import resolve_result_budget
from finharness.shared.capabilities import (
    capabilities_in_text,
)
from finharness.shared.fencing import close_dangled
from finharness.tools.meta.skills import (
    SkillError,
    SkillRegistry,
    report_requested,
    route,
)
from finharness.tools.registry import ToolRegistry
from finharness.types import (
    STATE_VIEW_META,
    AgentTurnOutcome,
    EngineEvent,
    ModelUsage,
    Msg,
    ObservedCall,
    OutputSink,
    RoundTrace,
    StopSignal,
    StreamEvent,
    ToolResult,
    ToolUse,
)
from finharness.utils.clock import utc_now_iso


class _CompactionMarker:
    """用于描述一次 compaction 的审计记录的替身工具。"""

    name = "context_compaction"


log = get_logger("finharness.engine.loop")


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


class AgentLoop(PlanProgressMixin):
    """持有一个会话的对话、usage 与工具预算。"""

    TRUNCATION_MARKER = "\n[truncated]"
    # 压缩记录只留最近若干条；该列表按会话存活，逐轮追加会无界增长。
    COMPACTION_HISTORY = 20

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
        semantic_index: Any | None = None,
        compute: Any | None = None,
        call_type: str = "main",
        route_skills: bool = True,
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
        # 场景路由（docs 03.8）是主循环的职责：它读用户问题与计划意图，据此注入方法论。
        # 子代理拿到的是一个孤立片段——它的输入是任务文本，没有用户的原始问题，也没有
        # 计划——在那里推断意图只会给一个只读核查者塞进与任务无关的方法论。
        self.route_skills = route_skills
        self.ctx = (
            ctx
            if ctx is not None
            else ResearchContext(cite=self.cite, settings=settings)
        )
        # 默认值对不注入任何内容的调用方保持治理引入之前的行为。
        self.gate = gate if gate is not None else ReadOnlyGate()
        self.hooks = hooks if hooks is not None else HookChain()
        self.interactive = interactive
        # 记忆作用域的用户归属（docs 03.13）：对话落库与偏好笔记都按它隔离。
        # 默认空串保持 CLI/eval 直连（无 HTTP 层）时的既有行为。先于协调器
        # 赋值：子代理的审计归属取自这里。
        self.user_id = user_id
        # sub-agent 协调器（docs 03.10）。按需从 registry 的 DataAccess 构建，
        # 因此不注入任何内容的调用方也能获得风险评审。
        # 记账在下方绑定，此时本循环自身的计数器已存在。
        self.coordinator = (
            coordinator if coordinator is not None else self._build_coordinator(counter=counter)
        )
        # 隔离计算通道（docs 03.15）：由服务端注入会话绑定的 SessionCompute，
        # 未注入（CLI/eval/本地）时为 None，声明了 needs_compute 的工具回退进程内。
        self.compute = compute
        # L1 memory 拥有对话记录及其记账；循环负责编排。
        # 可以注入一个 counter，以便调用方共享同一份词表缓存。
        self.memory = WorkingMemory(ctx=self.ctx, settings=settings, counter=counter)
        # 让 ctx.append_user/append_tool_result 转发到这里（docs 03.6.2）。
        self.ctx.memory = self.memory
        # 对话记忆（docs 03.6.4）：对话记录及建立在它之上的记忆按会话持久化，
        # 因此重启后无需重新摘要即可恢复。若没有 store，循环便和以前一样无记忆。
        self.conversation_id = conversation_id or session_id or "local"
        self.store = store
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
        # 语义记忆的向量索引（docs 03.6.4 LTM）：由调用方注入（服务端按配置
        # 构建 Qdrant/BLOB 后端）；未注入或未配 embedding 时语义召回关闭，
        # 记忆本体与键匹配检索仍然完全可用。
        self.semantic_index = semantic_index
        self.ctx.semantic_index = semantic_index
        self._memory_loaded = False
        self.usage = ModelUsage()
        # 评审 token 必须出现在会话总量中，因此既然计数器都已存在，就把协调器
        # 接到计数器上。测试替身可能没有该绑定钩子，这无妨——它只是不计费而已。
        binding = getattr(self.coordinator, "bind_accounting", None)
        if binding is not None:
            binding(usage=self.usage, stats=self.stats, observer=self.observer)
        self.turn = 0
        # 只保留最近若干次压缩记录：该列表按会话存活，逐轮追加会无界增长。
        # 累计次数单独计数，``done`` 载荷上报的是它。
        self.compactions: deque[CompactionResult] = deque(
            maxlen=self.COMPACTION_HISTORY
        )
        self.compaction_count = 0
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
        # 协作式停止信号（docs 03.3）。默认 None，使 eval、脚本与子代理等
        # 不经过服务层的调用方行为完全不变：没有信号就永远不检查。
        self.stop_signal: StopSignal | None = None
        # FSM 运行期暂存：答案不在 AgentState 上。
        self._pending_answer: str = ""
        self._pending_error: str | None = None
        self._pending_reason: str | None = None
        self._pending_fail_kind: str = ""
        self._pending_fail_message: str = ""
        self._think_deltas: list[str] = []
        self._think_round_input: int = 0
        self._think_round_output: int = 0
        self._think_llm_first_ms: int = 0
        self._think_llm_ms: int = 0
        self._machine: AgentStateMachine | None = None
        self._is_resume: bool = False
        self._resume_phase: AgentPhase | None = None
        self._finished_call_ids: set[str] = set()
        # FSM confirmation: ephemeral prompt payload + one-shot answers (not persisted).
        self._pending_confirm_spec: ConfirmationSpec | None = None
        self._confirmation_approved_ids: set[str] = set()
        self._interaction_answer: str | None = None
        self._skip_permission_ids: set[str] = set()
        # Queued by InteractivePort; emitted in _after_dispatch after ConfirmationResolved.
        self._pending_interaction_resolved: dict[str, Any] | None = None

    def _stopped(self) -> bool:
        """是否已收到停止请求；无信号时恒为 False。"""
        return self.stop_signal is not None and self.stop_signal.requested
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
        # 子代理的审计与主循环同一写入器、同一用户：audit.jsonl 里每个
        # run/denied 行因此都能独立回答"是谁"（session_id 带 focus 前缀，
        # 可与主会话区分）。写入器取自主循环的审计 hook；没有审计 hook 的
        # 调用方（测试替身）子代理照常运行，只是不留痕。
        audit_writer = None
        for hook in getattr(self.hooks, "hooks", []) or []:
            audit_writer = getattr(hook, "writer", None)
            if audit_writer is not None:
                break

        def audit_hook_factory(session_id: str):
            from finharness.hooks.audit import AuditHook

            return AuditHook(audit_writer, session_id=session_id, user_id=self.user_id)

        return Coordinator(
            provider=self.provider,
            data=data,
            settings=self.settings,
            cite=self.cite,
            counter=counter,
            # 停止信号在 loop 构造之后才由服务层装上，因此这里传一个取值回调
            # 而非当时的快照：子代理据此在每次运行前拿到最新信号，用户在长时间
            # 的子代理扇出期间按停止也能及时生效。
            stop_signal_provider=lambda: self.stop_signal,
            audit_hook_factory=audit_hook_factory if audit_writer is not None else None,
            user_id=self.user_id,
        )

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.output is not None:
            await self.output.emit(EngineEvent(kind, data))

    def _progress_emitter(self, tool_use: ToolUse):
        """构造工具的进展回调：补齐调用标识后转发为 ``tool_progress`` 事件。

        回调刻意"尽力而为"：上报失败只记一条日志，绝不上抛——进展通道断掉不该
        连累正在进行的计算。取消仍照常传播，否则超时取消会被吞掉。
        """

        async def report(payload: dict[str, Any]) -> None:
            data = {"call_id": tool_use.call_id, "name": tool_use.name, **payload}
            try:
                await self._emit("tool_progress", data)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 进展上报失败不中断工具执行
                log.warning(
                    "tool_progress_failed",
                    extra={"tool": tool_use.name, "turn": self.turn},
                )

        return report

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

        子代理不接收研究状态：它的输入只有任务文本，一个孤立的跑批不该因为共享了
        ``state_block`` 的渲染路径而多出主会话的上下文。当前日期随状态块注入主循环
        （它是判断数据时效的锚点），子代理则维持"只拿到任务"的既有边界。
        """
        if self.call_type != "main":
            return ""
        state = self.ctx.state_block()
        if self._plan_hint:
            state = f"{state}\n\n{self._plan_hint}" if state else self._plan_hint
        return state

    def _request_messages(self) -> list[Msg]:
        """历史记录，外加作为末尾消息的研究状态。

        状态放在最后而非最前，有两个原因：它让历史记录成为缓存可复用的稳定
        前缀，同时把最新的状态放在模型注意力最强之处。它从不追加到 ``raw``
        ——它是本请求的一个视图，而不是会持久化或被摘要的对话记录条目。

        该视图以 user 角色发送，因此带上 :data:`STATE_VIEW_META` 标记：否则
        「最后一条用户消息」的读法会把状态块当成用户的提问。
        """
        messages = self.memory.snapshot()
        state = self._state_text()
        if state:
            messages.append(Msg(role="user", content=state, metadata={STATE_VIEW_META: True}))
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

        # 对话行由 ``run()`` 在 ``machine.start()`` 之前创建（外键）。
        # 对话记录，让模型从会话中途续接，而不是从头开始。
        # track=False：重放的历史已经存储过；再缓冲它会以重复的序号再次写入。
        #
        # 长对话不把整份历史读进内存：只回放最近若干轮，更早的由摘要分段承载。
        # 但仅当被跳过的前缀确实已被摘要覆盖时才这么做——否则宁可全量加载，
        # 也不能静默丢掉没人记得的历史（此类对话 token 很小，或下一轮压缩即
        # 折叠前缀，之后重载自然有界）。
        messages = self._load_recent_history()
        for message in messages:
            self.memory.append(message, track=False)
        # Citation 必须保留其 id：已存储的摘要会引用它们。
        self.cite.restore(self.store.load_citations(self.conversation_id, user_id=self.user_id))
        self.ctx.prior_conclusions = self.store.load_conclusions(self.conversation_id)
        self.ctx.notes = self.store.get_notes(user_id=self.user_id)
        # 跨对话长期记忆（docs 03.6.4 LTM）：最近情节被动注入（首轮一次），
        # 之后由标的池驱动刷新（refresh_ltm_recall，与 L2 recall 同模式）。
        self.ctx.ltm_recent = self.store.list_ltm_episodes(
            user_id=self.user_id,
            limit=int(self.settings.ltm.recent_episodes),
        )
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
        self.ctx.refresh_ltm_recall()

    def _restore_checkpoint(self) -> None:
        """从上次中断的断点恢复研究计划（docs 03.3）。

        这是"恢复"的全部内容：消息与引用已由常规回放载入，唯一无法从消息
        重建的执行态就是计划。把原计划装回 ``ctx`` 后，模型看到完整历史 + 原有
        计划与各步状态，于是能接着做而不是从头重来——它也因此不会重复调用那些
        已标记为 done 的步骤。

        只在断点确实处于"可继续"（用户停止）时恢复：``completed`` 表示上轮已
        交付完毕，其计划只是上一轮的遗物，重新装上会让新一轮无端继承旧目标。

        由 ``_run_once`` 在**每次** ``run()`` 时调用，而不是只在会话首轮：断点
        属于"一轮"，而一个会话可以承载多轮。会话复用时记忆已经加载过，若把它
        挂在首轮加载里，"同一窗口内继续"就会既读不到断点、也消费不掉它。
        """
        if self.store is None:
            return
        try:
            checkpoint = self.store.load_latest_checkpoint(self.conversation_id)
        except Exception:  # noqa: BLE001 - 断点读失败按"没有断点"处理
            log.exception("checkpoint_load_failed")
            return
        if checkpoint is None:
            return
        # 已消费：新一轮一旦开始，旧的"可继续"就不再成立，界面不该在运行途中
        # 仍显示续做入口。本轮收尾时会写入属于它自己的新断点。
        try:
            self.store.clear_checkpoint(self.conversation_id)
        except Exception:  # noqa: BLE001 - 清理失败不影响恢复本身
            log.exception("checkpoint_clear_failed")
        if not checkpoint.recoverable or not checkpoint.plan:
            return
        restored = Plan.from_dict(checkpoint.plan)
        if restored is None:
            return
        self.ctx.plan = restored
        log.info(
            "checkpoint_plan_restored",
            extra={
                "conversation_id": self.conversation_id,
                "plan_id": restored.plan_id,
                "revision": restored.revision,
                "rounds": checkpoint.rounds,
            },
        )

    def _load_recent_history(self) -> list[Msg]:
        """回放对话记录；长对话只取最近若干轮，并让序号计数与存储对齐。

        只有被跳过的前缀确实已被摘要分段覆盖时才截断加载——摘要才是那部分
        历史的载体，没有它就是不声不响地遗忘。覆盖不足时回退全量加载。
        """
        limit = int(self.settings.context.max_loaded_rounds)
        total = self.store.count_messages(self.conversation_id)
        if limit <= 0 or total == 0:
            return self.store.load_messages(self.conversation_id)
        bounded = self.store.load_messages(self.conversation_id, recent_rounds=limit)
        # seq 在单个对话内从 1 连续递增，因此被跳过的条数就等于切点前一位。
        skipped = total - len(bounded)
        if skipped <= 0:
            return bounded
        if self.store.count_covered_prefix(self.conversation_id) < skipped:
            # 前缀还没被摘要收下，不能丢；全量加载，交由压缩去折叠它。
            return self.store.load_messages(self.conversation_id)
        # 让后续压缩的 seq_from/seq_to 与真实消息序号对齐，而不是从 1 重数——
        # 否则会把已覆盖的区间再摘要一遍。
        self.memory.discarded = skipped
        return bounded

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
        self.store.save_citations(self.conversation_id, self.cite.all(), user_id=self.user_id)
        for symbol in self.ctx.symbols:
            self.store.upsert_symbol(self.conversation_id, symbol)
        # 跨对话情节记忆（docs 03.6.4 LTM）：这是**唯一**无约定环节的全局写入
        # 路径，因此默认关闭（ltm.auto_task_episodes）。关闭后跨对话记忆只来自
        # 显式写入与懒蒸馏（decision/excerpt）。对话内结论照常按对话隔离落库。
        # add_ltm_episode 按 (kind, subject, summary) 幂等，与 save_conclusion
        # 的对话内去重互不干扰。标题只查一次（每条情节都要冗余它）。
        title = self._conversation_title()
        for conclusion in self.ctx.pending_conclusions:
            subject = self._conclusion_subject(conclusion.cids)
            self.store.save_conclusion(
                self.conversation_id,
                subject=subject,
                text=conclusion.text,
                cids=conclusion.cids,
            )
            if not self.settings.ltm.auto_task_episodes:
                continue
            try:
                self.store.add_ltm_episode(
                    kind="task_result",
                    summary=conclusion.text,
                    user_id=self.user_id,
                    subject=subject,
                    cids=conclusion.cids,
                    source_conversation_id=self.conversation_id,
                    source_title=title,
                    source_ts=conclusion.ts,
                )
            except Exception:  # noqa: BLE001 - 情节落库失败不影响对话持久化
                pass
        self.ctx.pending_conclusions = []
        self.store.touch_conversation(self.conversation_id)

    def _conversation_title(self) -> str:
        """当前对话的标题（情节溯源用）；未持久化时为空。"""
        record = (
            self.store.get_conversation(self.conversation_id)
            if self.store is not None
            else None
        )
        return (record.title if record is not None else "") or ""

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
        # 新 symbol 进入对话：历史上关于该标的的跨对话情节随之带出
        # （docs 03.6.4 LTM 标的驱动召回，与 L2 recall 同一触发点）。
        self.ctx.refresh_ltm_recall()

    def _note_review_outcome(self, result: ToolResult) -> None:
        """把工具声明的风险终审结局记入会话状态（docs 03.10.7）。

        终审的编排在工具内部完成，但"这份报告还留着未处理的严重问题"必须在**后续
        轮次**依然可见，否则模型可以在拿到意见后的任意一轮里把它忘掉并照常交付。
        ``metadata["review"]`` 是工具声明的载荷（审计钩子读同一份），这里只把其中
        的未结事项搬进 ``ctx``；消解（``unresolved_note`` 为 None）会清除旧条目。

        写入时机在本轮所有 prompt 度量之后（工具只在请求返回后执行），因此一轮内
        三次度量仍看到相同的状态文本。
        """
        review = result.metadata.get("review")
        if not isinstance(review, dict):
            return
        topic = str(review.get("topic") or "")
        if not topic:
            return
        note = review.get("unresolved_note")
        self.ctx.note_review_finding(topic, str(note) if note else None)

    def _build_compactor(self) -> AutoCompactor:
        return AutoCompactor(
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

    async def _apply_compaction_result(self, result: CompactionResult) -> None:
        """Emit/audit one compaction result (degrade still continues)."""
        if not result.compacted and result.warning is None:
            return
        self.compactions.append(result)
        self.compaction_count += 1
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
            except Exception:  # noqa: BLE001 - 审计失败绝不能中断一轮
                self._log_audit_failure("compact")

    async def _maybe_compact(self) -> None:
        """当下一次请求将超出窗口预算时，对历史进行折叠压缩。"""
        compactor = self._build_compactor()
        if not compactor.needs_compaction():
            return
        result = await compactor.compact()
        await self._apply_compaction_result(result)

    def _log_audit_failure(self, action: str, tool_name: str = "?") -> None:
        """审计写入失败时必须留下痕迹。

        审计是治理链路里唯一的记录，因此"这一轮没被记下来"本身是一个必须可见的
        事件。写失败仍然不致命（照旧不中断一轮），但不能再是 ``pass``。
        """
        log.warning(
            "审计写入失败：tool=%s action=%s", tool_name, action, exc_info=True
        )
        # 同时计数：审计失败率是需要告警的治理信号（隔离方案 Phase 2）。
        try:
            self.observer.record_governance_event(kind="audit_write_failure")
        except Exception:  # noqa: BLE001 - 观测绝不能再抛
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
        except Exception:  # noqa: BLE001 - 审计失败绝不能中断一轮
            self._log_audit_failure("loop_detected")

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
        "user_stopped": "已按你的要求停止",
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

    def _done_payload(
        self, *, succeeded: bool, reason: str | None, tool_calls: int
    ) -> dict[str, Any]:
        snapshot = self.stats.snapshot()
        plan_payload = self._plan_snapshot() or None
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
            "compactions": self.compaction_count,
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

    def _checkpoint_plan(self) -> dict | None:
        """本次运行的研究计划，序列化用于断点（docs 03.3）。

        计划是唯一无法从已落库的消息重建的执行态——它只活在 ``ctx`` 里，
        因此"恢复后在同一份计划上继续"必须靠它。
        """
        plan = self.ctx.plan
        return plan.to_dict() if plan is not None else None

    def _save_checkpoint(self, *, status: str, reason: str = "", partial_answer: str = "") -> None:
        """写下本轮的断点；无存储的调用方（eval/脚本/子代理）为空操作。

        同步 SQLite 写入：这一点在取消路径上很关键——``CancelledError`` 之下
        任何 ``await`` 都可能被立刻打断，而一次 sqlite 调用不会。
        """
        if self.store is None:
            return
        try:
            persisted = self.store.message_seq_range(self.conversation_id)[1]
            self.store.save_checkpoint(
                self.conversation_id,
                status=status,
                reason=reason,
                rounds=self.rounds,
                turn_index=self.turn,
                plan=self._checkpoint_plan(),
                partial_answer=partial_answer,
                persisted_seq=persisted,
            )
        except Exception:  # noqa: BLE001 - 断点写失败不该影响本轮收尾
            log.exception("checkpoint_save_failed")

    async def _stop_turn(self, *, tool_calls: int) -> AgentTurnOutcome:
        """按用户要求结束本次运行，并保留既有成果（docs 03.3）。

        与 ``_fail`` 的关键差别：停止不是失败，因此**不发** ``error`` 事件——
        用户按了停止却看到一条错误提示是错的。已确立的内容照常以 ``answer``
        交付并落库，使停止不会让已付出的取数与结论白费。
        """
        reason = "user_stopped"
        answer = self._partial_answer(reason=reason) if self._has_partial_findings() else ""
        if answer:
            await self._emit("answer", {"text": answer})
            # 与 ``_fail`` 同理：只在事件里见过的答案重载后会消失，因此把部分
            # 发现也写为本轮的 assistant 消息，随本轮一起落库。
            self.memory.append_assistant(Msg(role="assistant", content=answer))
        else:
            # 没有可交付的发现：清掉流式中途的半句话。否则用户会看到一句
            # 被截断的草稿，误以为那是一个答案。
            await self._emit("text_reset", {})
        # ``resumable`` 告知客户端：这不是终点，可以基于断点继续。
        await self._emit(
            "done",
            {
                **self._done_payload(
                    succeeded=False, reason=reason, tool_calls=tool_calls
                ),
                "resumable": True,
            },
        )
        return AgentTurnOutcome(
            answer=answer,
            succeeded=False,
            usage=self.usage,
            error=None,
            reason=reason,
            tool_calls=tool_calls,
            retry_count=self.stats.retry_count,
            tool_duration_ms=self.stats.snapshot().tool_duration_ms,
            citations=self._citation_ids(),
            trace=list(self.trace),
            rounds=self.rounds,
        )

    async def run(self, user_msg: str, *, resume: bool = False) -> AgentTurnOutcome:
        """运行一次完整的 agent 轮次循环，返回本轮结果。

        重置各项按运行维护的状态，在需要时加载持久化记忆，然后交由
        ``AgentRunner`` 按显式阶段执行，最后持久化本轮消息。

        本方法持有一轮请求的根 Span：它是所有入口（服务端、eval、脚本）共同
        的执行边界，因此请求级耗时只在这里统计一次，不会重复计数。
        """
        model = getattr(self.provider, "model", "") or ""
        with self.observer.request_span(
            message=user_msg,
            model=model,
            emit_metric=self.call_type == "main",
        ) as span:
            outcome = await self._run_once(user_msg, resume=resume)
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

    def _make_state_store(self) -> MemoryStateStore | SqliteAgentStateStore:
        if isinstance(self.store, MemoryStore):
            return SqliteAgentStateStore(self.store)
        return MemoryStateStore()

    def _reset_run_locals(self) -> None:
        """按运行维护的状态：轮次预算按请求计算，因此计数器随之重置。"""
        self.turn = 0
        self._call_counts = {}
        self._reminded = set()
        self._plan_hint = ""
        self._plan_signature = None
        self._plan_stall_turns = 0
        self._reported_drift = set()
        self._reported_mismatch = set()
        self.trace = []
        self.rounds = 0
        self._call_meta = {}
        self._pending_answer = ""
        self._pending_error = None
        self._pending_reason = None
        self._pending_fail_kind = ""
        self._pending_fail_message = ""
        self._think_deltas = []
        self._think_round_input = 0
        self._think_round_output = 0
        self._think_llm_first_ms = 0
        self._think_llm_ms = 0
        self._finished_call_ids = set()
        self._pending_confirm_spec = None
        self._confirmation_approved_ids = set()
        self._interaction_answer = None
        self._skip_permission_ids = set()
        self._pending_interaction_resolved = None

    def queue_interaction_resolved(self, payload: dict[str, Any]) -> None:
        """InteractivePort stashes resolved metadata; emit after ConfirmationResolved."""
        self._pending_interaction_resolved = payload

    def _failed_resume_outcome(self) -> AgentTurnOutcome:
        reason = "no_resumable_state"
        message = "no resumable agent state for this conversation"
        return AgentTurnOutcome(
            answer="",
            succeeded=False,
            usage=self.usage,
            error=message,
            reason=reason,
            tool_calls=0,
            retry_count=self.stats.retry_count,
            tool_duration_ms=self.stats.snapshot().tool_duration_ms,
            citations=self._citation_ids(),
            trace=list(self.trace),
            rounds=self.rounds,
        )

    async def _after_dispatch(self, messages: tuple[Msg, ...]) -> None:
        """Apply committed Msgs; emit queued interaction_resolved after persist."""
        known = {id(message) for message in self.memory.raw}
        for message in messages:
            if id(message) not in known:
                self.memory.append(message, track=False)
        committed = {id(message) for message in messages}
        if committed:
            self.memory.pending = [
                message
                for message in self.memory.pending
                if id(message) not in committed
            ]
        pending = self._pending_interaction_resolved
        if pending is not None:
            self._pending_interaction_resolved = None
            await self._emit("interaction_resolved", pending)

    def _tool_permission(self, name: str) -> str:
        resolve = getattr(self.registry, "resolve", None)
        tool = resolve(name) if callable(resolve) else None
        if tool is None:
            return "unknown"
        is_read_only = getattr(self.registry, "is_read_only", None)
        if callable(is_read_only) and is_read_only(name):
            return "read"
        return "write"

    async def _run_once(self, user_msg: str, *, resume: bool = False) -> AgentTurnOutcome:
        """``run`` 的实际执行体；根 Span 由调用方持有。"""
        # 记住该问题：scope/偏离信号从它读取任务自身的 symbol 与 capability，
        # 这才是事实依据，而非计划可能不完整的重述。
        self._last_user_msg = user_msg or ""
        state_store = self._make_state_store()
        self._is_resume = False
        self._resume_phase = None

        if resume:
            snapshot = state_store.latest_resumable(self.conversation_id, self.user_id)
            if snapshot is None:
                return self._failed_resume_outcome()
            self._reset_run_locals()
            self._is_resume = True
            self._resume_phase = snapshot.phase
            machine = AgentStateMachine(
                snapshot, store=state_store, output=self.output
            )
            self._machine = machine
            await machine.dispatch(ResumeRequested(at=utc_now_iso()))
        else:
            self._reset_run_locals()
            # A normal new message abandons any older resumable run before
            # hydrate, so eval/scripts share the same supersede semantics.
            state_store.abandon_latest(
                self.conversation_id, self.user_id, now=utc_now_iso()
            )
            if isinstance(self.store, MemoryStore):
                self.store.ensure_conversation(
                    self.conversation_id,
                    user_id=self.user_id,
                    title=_title_from(user_msg),
                )
            initial = new_agent_state(
                run_id=str(uuid.uuid4()),
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                now=utc_now_iso(),
            )
            machine = AgentStateMachine(
                initial, store=state_store, output=self.output
            )
            self._machine = machine
            await machine.start()

        effects = RunnerEffects(
            hydrate=self._effect_hydrate,
            compact=self._effect_compact,
            think=self._effect_think,
            use_tools=self._effect_tooluse,
            await_confirmation=self._effect_await_confirmation,
            finish=self._finish_run,
        )
        runner = AgentRunner(
            machine, effects, after_dispatch=self._after_dispatch
        )
        try:
            outcome = await runner.run()
        except BaseException:
            # 硬取消（连接被掐断、任务被取消、进程即将退出）也必须留下本轮成果。
            # ``CancelledError`` 继承自 ``BaseException``，会穿出 runner
            # 里所有 ``except Exception``；若不在这里兜住，下面的持久化将被跳过，
            # 用户消息与已完成的取数会一并消失——这正是"中断即丢数据"的成因。
            # ``_persist_turn``/``_save_checkpoint`` 是同步 SQLite 写入，在被取消
            # 的任务里仍能跑完。
            self._persist_turn()
            self._save_checkpoint(status="stopped", reason="interrupted")
            raise
        finally:
            self._machine = None
        self._persist_turn()
        # 断点是"可继续"状态的载体：正常交付记 completed，运行失败（含用户停止）
        # 记 stopped。判据取自引擎结论而非异常与否——停止本就不抛异常。
        self._save_checkpoint(
            status="completed" if outcome.succeeded else "stopped",
            reason=outcome.reason or "",
            partial_answer=outcome.answer if not outcome.succeeded else "",
        )
        return outcome

    async def _effect_hydrate(self, state: AgentState) -> EffectResult:
        """Load memory, restore checkpoint, recall, route; decide compaction."""
        await self._load_memory_if_first_turn(self._last_user_msg)
        # 断点属于"一轮"，而一个会话可承载多轮，因此它必须在每次 run() 都检查：
        # 会话复用时上面的加载会提前返回，挂在里面就会漏掉"同一窗口内继续"。
        # 放在这里（而非更晚）是为了让本轮第一次 prompt 组装就能看到恢复的计划。
        self._restore_checkpoint()
        # 语义召回按**本轮问题**刷新（docs 03.6.4 LTM）：与情节的标的驱动不同，
        # "哪个知识跟当前问题相关"只有拿到问题文本才能判断。它发生在 prompt
        # 组装之前，因此本轮三次测量看到的是同一份文本。
        self.ctx.refresh_semantic_recall(self._last_user_msg, index=self.semantic_index)
        # 在追加之前打开本轮的写缓冲区，这样用户消息会被缓存以供持久化，
        # 而不会被重置操作丢弃。
        self.memory.pending = []
        # Resume must not duplicate the user frame already in durable history.
        if not self._is_resume and self._last_user_msg:
            self.memory.append_user(self._last_user_msg)

        # 路由：从问题本身推断需要哪些方法论，并注入其正文。放在轮次循环之前，因此
        # 第一次模型调用就已经看得到方法，而不是等到某一轮才补上——简单提问没有计划，
        # 这是它唯一能拿到方法论的时机。
        if self.route_skills and not self._is_resume:
            await self._route_methodology()

        mid_tool_round = self._resume_phase in {
            AgentPhase.TOOL_USE,
            AgentPhase.AWAITING_CONFIRMATION,
        }
        if mid_tool_round:
            needs_compaction = False
            resume_phase = self._resume_phase
        else:
            needs_compaction = self._build_compactor().needs_compaction()
            allowed = {
                AgentPhase.THINKING,
                AgentPhase.TOOL_USE,
                AgentPhase.AWAITING_CONFIRMATION,
            }
            candidate = self._resume_phase or state.resume_phase
            resume_phase = candidate if candidate in allowed else None
        return EffectResult(
            HydrationFinished(
                needs_compaction=needs_compaction,
                resume_phase=resume_phase,
                at=utc_now_iso(),
            )
        )

    async def _route_methodology(self) -> list[str]:
        """按问题意图注入方法论正文，返回本次新注入的目标键。

        注入是幂等的（``ctx.inject_methodology`` 按目标去重），因此一条追问不会重复
        注入同一份方法；只有首次命中才计入 ``context_routed`` 事件，使"为什么这段方法
        会在这里"可归因。
        """
        capabilities = capabilities_in_text(self._last_user_msg)
        report = report_requested(self._last_user_msg)
        if self.ctx.plan is not None:
            plan = self.ctx.plan
            capabilities |= capabilities_in_text(self._plan_prose(plan))
            hinted = tuple(
                name for step in plan.steps for name in step.skill_hint if name
            )
        else:
            hinted = ()

        targets = route(capabilities=capabilities, hinted=hinted, report=report)
        if not targets:
            return []

        registry = SkillRegistry(self.settings.paths.skills_dir)
        injected: list[str] = []
        for target in targets:
            try:
                _meta, body, _record = registry.load(target.skill, file=target.file)
            except SkillError:
                # 技能包缺失或格式错误只是"这份方法这次没有"，不该让整个回合失败。
                continue
            if body.strip() and self.ctx.inject_methodology(target.key, body):
                injected.append(target.key)

        if injected:
            await self._emit(
                "context_routed",
                {"skills": injected, "capabilities": sorted(c.value for c in capabilities)},
            )
        return injected

    async def _effect_compact(self, state: AgentState) -> EffectResult:
        """Run compaction body; degrade still continues to thinking."""
        del state
        compactor = self._build_compactor()
        result = await compactor.compact()
        await self._apply_compaction_result(result)
        return EffectResult(CompactionFinished(at=utc_now_iso()))

    async def _effect_think(self, state: AgentState) -> EffectResult:
        """One provider stream; return ModelFinished / StopRequested / RunFailed."""
        now = utc_now_iso()
        if state.turn >= self.settings.context.max_turns:
            self._pending_reason = "max_turns_exhausted"
            self._pending_fail_kind = "max_turns_exhausted"
            self._pending_fail_message = "maximum agent turns exhausted"
            self._pending_error = None
            self._pending_answer = self._partial_answer(reason="max_turns_exhausted")
            return EffectResult(
                RunFailed(
                    kind="max_turns_exhausted",
                    message="maximum agent turns exhausted",
                    at=now,
                )
            )

        if self._stopped():
            self._pending_reason = "user_stopped"
            return EffectResult(StopRequested(at=now))

        self.turn = state.turn + 1
        update_turn(self.turn)
        # Mid-run window maintenance (after tool rounds). First-turn compact is
        # an explicit COMPACT phase when hydrate reported needs_compaction.
        await self._maybe_compact()

        model = getattr(self.provider, "model", "") or ""
        capture_messages = bool(getattr(self.observer, "tracing_capture_payloads", False))
        deltas: list[str] = []
        tool_uses: list[ToolUse] = []
        round_started = self.stats.now()
        first_token_at: float | None = None
        round_input = 0
        round_output = 0
        stop_requested = False

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
                    on_retry=lambda error, index, delay: self._on_retry(
                        error, index, delay
                    ),
                ):
                    if self._stopped():
                        stop_requested = True
                        break
                    if chunk.event is StreamEvent.TEXT_DELTA:
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
                        llm_span.set_usage(chunk.data)
            except Exception as exc:
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
                self._pending_reason = "provider_error"
                self._pending_fail_kind = type(exc).__name__
                self._pending_fail_message = str(exc)
                self._pending_error = str(exc)
                self._pending_answer = (
                    self._partial_answer(reason="provider_error")
                    if self._has_partial_findings()
                    else ""
                )
                return EffectResult(
                    RunFailed(
                        kind=type(exc).__name__,
                        message=str(exc),
                        at=utc_now_iso(),
                    )
                )

        stream_ended = self.stats.now()
        llm_first_ms = (
            round((first_token_at - round_started) * 1000)
            if first_token_at is not None
            else 0
        )
        llm_ms = round((stream_ended - round_started) * 1000)
        self._think_deltas = deltas
        self._think_round_input = round_input
        self._think_round_output = round_output
        self._think_llm_first_ms = llm_first_ms
        self._think_llm_ms = llm_ms

        if stop_requested:
            self._record_round(
                thought="".join(deltas),
                actions=[],
                results=[],
                input_tokens=round_input,
                output_tokens=round_output,
                llm_first_ms=llm_first_ms,
                llm_ms=llm_ms,
            )
            self._pending_reason = "user_stopped"
            return EffectResult(StopRequested(at=utc_now_iso()))

        usage = UsageDelta(input_tokens=round_input, output_tokens=round_output)
        if tool_uses:
            if deltas:
                await self._emit("text_reset", {})
            permissions = {
                tool_use.call_id: self._tool_permission(tool_use.name)
                for tool_use in tool_uses
            }
            assistant = Msg(role="assistant", content=None, tool_uses=list(tool_uses))
            # Keep pending until dispatch commits; clear in _after_dispatch.
            pending = tuple(self.memory.pending)
            return EffectResult(
                ModelFinished(
                    answer="",
                    tool_uses=tuple(tool_uses),
                    usage=usage,
                    at=utc_now_iso(),
                    permissions=permissions,
                ),
                messages=pending + (assistant,),
            )

        answer = "".join(deltas)
        self._pending_answer = answer
        self._pending_reason = None
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
        return EffectResult(
            ModelFinished(
                answer=answer,
                tool_uses=(),
                usage=usage,
                at=utc_now_iso(),
            )
        )

    async def _effect_tooluse(self, state: AgentState) -> EffectResult:
        """Execute/recover persisted tool calls; finalize one tool-result Msg."""
        machine = self._machine
        if machine is None:
            raise RuntimeError("tooluse effect requires an active state machine")

        now = utc_now_iso()
        deltas = self._think_deltas
        round_input = self._think_round_input
        round_output = self._think_round_output
        llm_first_ms = self._think_llm_first_ms
        llm_ms = self._think_llm_ms
        self._finished_call_ids = {
            call.call_id
            for call in state.calls
            if call.status in {CallStatus.COMPLETED, CallStatus.FAILED}
            and call.result_json is not None
        }

        if self._is_resume:
            # Consume resume classification once: after confirmation resolves,
            # re-entering tooluse must execute (not re-prompt uncertain writes).
            self._is_resume = False
            uncertain_ids: list[str] = []
            for call in state.calls:
                if call.status in {CallStatus.COMPLETED, CallStatus.FAILED}:
                    continue
                if call.status is CallStatus.UNCERTAIN:
                    uncertain_ids.append(call.call_id)
                    continue
                if call.status in {CallStatus.PENDING, CallStatus.RUNNING}:
                    if call.permission == "read":
                        continue
                    uncertain_ids.append(call.call_id)

            if uncertain_ids:
                for call_id in uncertain_ids:
                    current = next(
                        c for c in machine.state.calls if c.call_id == call_id
                    )
                    if current.status is not CallStatus.UNCERTAIN:
                        await machine.dispatch(
                            ToolCallFinished(
                                call_id=call_id,
                                status=CallStatus.UNCERTAIN,
                                result_json=None,
                                at=utc_now_iso(),
                            )
                        )
                return EffectResult(
                    ConfirmationRequested(
                        prompt="上次执行结果未知，是否重试该写操作？",
                        options=("y", "n"),
                        call_ids=tuple(uncertain_ids),
                        at=utc_now_iso(),
                        kind="permission",
                        category="write",
                    )
                )

        tool_uses = [
            ToolUse(call_id=call.call_id, name=call.name, args=dict(call.args))
            for call in state.calls
        ]
        results_by_id: dict[str, str] = {
            call.call_id: call.result_json
            for call in state.calls
            if call.status in {CallStatus.COMPLETED, CallStatus.FAILED}
            and call.result_json is not None
        }

        async def run_one(call) -> tuple[str, str]:
            tool_use = ToolUse(
                call_id=call.call_id, name=call.name, args=dict(call.args)
            )
            await machine.dispatch(
                ToolCallStarted(call_id=call.call_id, at=utc_now_iso())
            )
            call_id, result_json = await self._execute_one(tool_use)
            status = CallStatus.COMPLETED
            try:
                payload = json.loads(result_json)
                if isinstance(payload, dict) and not payload.get("ok", True):
                    status = CallStatus.FAILED
            except (TypeError, ValueError):
                pass
            await machine.dispatch(
                ToolCallFinished(
                    call_id=call_id,
                    status=status,
                    result_json=result_json,
                    at=utc_now_iso(),
                )
            )
            self._finished_call_ids.add(call_id)
            return call_id, result_json

        to_run = [
            call
            for call in state.calls
            if call.call_id not in results_by_id
            and call.status
            in {
                CallStatus.PENDING,
                CallStatus.RUNNING,
            }
        ]

        allow_calls, deny_calls, confirm_specs = await self._partition_tool_calls(
            to_run
        )
        for call in machine.state.calls:
            if call.result_json is not None:
                results_by_id[call.call_id] = call.result_json
        if confirm_specs:
            if allow_calls or deny_calls:
                abort = await self._run_allow_calls(
                    allow_calls,
                    deny_calls=deny_calls,
                    run_one=run_one,
                    results_by_id=results_by_id,
                    machine=machine,
                    state=state,
                    tool_uses=tool_uses,
                    deltas=deltas,
                    round_input=round_input,
                    round_output=round_output,
                    llm_first_ms=llm_first_ms,
                    llm_ms=llm_ms,
                )
                if abort is not None:
                    return abort
            # One homogeneous group (same kind+category) per ConfirmationRequested;
            # leave other pending confirms for the next tooluse pass.
            first_spec = confirm_specs[0][1]
            group = [
                (call_id, spec)
                for call_id, spec in confirm_specs
                if spec.kind == first_spec.kind and spec.category == first_spec.category
            ]
            call_ids = tuple(call_id for call_id, _ in group)
            merged = ConfirmationSpec(
                kind=first_spec.kind,
                prompt=first_spec.prompt,
                options=first_spec.options,
                category=first_spec.category,
                call_ids=call_ids,
                multi_select=first_spec.multi_select,
            )
            self._pending_confirm_spec = merged
            return EffectResult(
                ConfirmationRequested(
                    prompt=merged.prompt,
                    options=merged.options,
                    call_ids=call_ids,
                    at=utc_now_iso(),
                    kind=merged.kind,
                    category=merged.category,
                    multi_select=merged.multi_select,
                )
            )

        abort = await self._run_allow_calls(
            allow_calls,
            deny_calls=deny_calls,
            run_one=run_one,
            results_by_id=results_by_id,
            machine=machine,
            state=state,
            tool_uses=tool_uses,
            deltas=deltas,
            round_input=round_input,
            round_output=round_output,
            llm_first_ms=llm_first_ms,
            llm_ms=llm_ms,
        )
        if abort is not None:
            return abort

        for call in state.calls:
            if (
                call.status is CallStatus.FAILED
                and call.call_id not in results_by_id
            ):
                tool_use = ToolUse(
                    call_id=call.call_id, name=call.name, args=dict(call.args)
                )
                # Populate _call_meta so refused_tools / eval scorers see the name.
                self._note_call(
                    tool_use,
                    ok=False,
                    error="用户已拒绝该工具调用",
                    duration_ms=0,
                )
                results_by_id[call.call_id] = json.dumps(
                    {
                        "ok": False,
                        "content": "",
                        "error": "用户已拒绝该工具调用",
                    },
                    ensure_ascii=False,
                )

        ordered = [
            (call.call_id, results_by_id[call.call_id]) for call in state.calls
        ]
        tool_msg = Msg(role="tool_result", content=None, tool_results=ordered)
        self._record_round(
            thought="".join(deltas),
            actions=tool_uses,
            results=ordered,
            input_tokens=round_input,
            output_tokens=round_output,
            llm_first_ms=llm_first_ms,
            llm_ms=llm_ms,
        )
        if self._stopped():
            self._pending_reason = "user_stopped"
            return EffectResult(
                StopRequested(at=utc_now_iso()), messages=(tool_msg,)
            )
        progress = self._plan_progress(tool_uses)
        if progress is not None:
            self._plan_hint = self._plan_hint_text(progress)
            progress["turn"] = self.turn
            await self._emit("plan_progress", progress)
        return EffectResult(ToolBatchFinished(at=now), messages=(tool_msg,))

    async def _run_allow_calls(
        self,
        allow_calls: list,
        *,
        deny_calls: list | None = None,
        run_one,
        results_by_id: dict[str, str],
        machine: AgentStateMachine,
        state: AgentState,
        tool_uses: list,
        deltas: list[str],
        round_input: int,
        round_output: int,
        llm_first_ms: int,
        llm_ms: int,
    ) -> EffectResult | None:
        """Reject DENY sequentially via run_one, then gather ALLOW calls.

        DENY still goes through ``_execute_one`` (observer/stats/tool_status), but
        never races under ``asyncio.gather`` with ALLOW — SSE failed-before-started
        ordering for mixed batches stays deterministic.
        """
        deny_calls = list(deny_calls or [])
        try:
            for call in deny_calls:
                call_id, result_json = await run_one(call)
                results_by_id[call_id] = result_json
            if not allow_calls:
                return None
            finished = await asyncio.gather(*(run_one(call) for call in allow_calls))
            for call_id, result_json in finished:
                results_by_id[call_id] = result_json
        except LoopDetected as detected:
            aborted = await self._persist_aborted_results(machine, state.calls)
            for call_id, result_json in aborted:
                results_by_id.setdefault(call_id, result_json)
            ordered = [
                (call.call_id, results_by_id[call.call_id])
                for call in state.calls
                if call.call_id in results_by_id
            ]
            tool_msg = Msg(role="tool_result", content=None, tool_results=ordered)
            self._record_round(
                thought="".join(deltas),
                actions=tool_uses,
                results=ordered,
                input_tokens=round_input,
                output_tokens=round_output,
                llm_first_ms=llm_first_ms,
                llm_ms=llm_ms,
            )
            await self._audit_detection(detected)
            self._pending_reason = "loop_detected"
            self._pending_fail_kind = "loop_detected"
            self._pending_fail_message = str(detected)
            self._pending_error = str(detected)
            self._pending_answer = self._partial_answer(reason="loop_detected")
            return EffectResult(
                RunFailed(
                    kind="loop_detected",
                    message=str(detected),
                    at=utc_now_iso(),
                ),
                messages=(tool_msg,),
            )
        except BaseException:
            aborted = await self._persist_aborted_results(machine, state.calls)
            for call_id, result_json in aborted:
                results_by_id.setdefault(call_id, result_json)
            ordered = [
                (call.call_id, results_by_id[call.call_id])
                for call in state.calls
                if call.call_id in results_by_id
            ]
            tool_msg = Msg(role="tool_result", content=None, tool_results=ordered)
            await machine.dispatch(
                ToolBatchFinished(at=utc_now_iso()), messages=(tool_msg,)
            )
            await self._after_dispatch((tool_msg,))
            raise
        return None

    async def _partition_tool_calls(
        self, to_run: list
    ) -> tuple[list, list, list[tuple[str, ConfirmationSpec]]]:
        """Split pending calls into ALLOW, DENY, and confirmation-needed.

        DENY calls are returned separately so the caller can reject them
        sequentially via ``run_one`` → ``_execute_one`` (unified observer/stats/
        tool_status bookkeeping) before ``asyncio.gather`` on ALLOW. CONFIRM /
        ask_user leave tooluse via ``ConfirmationRequested`` (no await inside
        gather).
        """
        allow_calls: list = []
        deny_calls: list = []
        confirm_specs: list[tuple[str, ConfirmationSpec]] = []
        machine = self._machine
        if machine is None:
            return list(to_run), [], []

        for call in to_run:
            if call.call_id in self._confirmation_approved_ids:
                self._confirmation_approved_ids.discard(call.call_id)
                self._skip_permission_ids.add(call.call_id)
                allow_calls.append(call)
                continue

            tool = self.registry.resolve(call.name)
            if tool is None:
                allow_calls.append(call)
                continue

            if getattr(tool, "needs_interactive", False):
                options = call.args.get("options") or []
                if not isinstance(options, list):
                    options = list(options) if options else []
                spec = ConfirmationSpec(
                    kind="question",
                    prompt=str(call.args.get("question") or ""),
                    options=tuple(str(item) for item in options),
                    category="question",
                    call_ids=(call.call_id,),
                    multi_select=bool(call.args.get("multi_select", False)),
                )
                confirm_specs.append((call.call_id, spec))
                continue

            decide = getattr(self.gate, "decide", None)
            if callable(decide):
                decision = decide(tool, dict(call.args))
            else:
                decision = await self.gate.check(tool, dict(call.args))

            if decision.verdict is Verdict.DENY:
                deny_calls.append(call)
                continue

            if decision.verdict is Verdict.CONFIRM and decision.confirmation is not None:
                confirm_specs.append((call.call_id, decision.confirmation))
                continue

            allow_calls.append(call)

        return allow_calls, deny_calls, confirm_specs

    async def _persist_aborted_results(
        self, machine: AgentStateMachine, calls: tuple
    ) -> list[tuple[str, str]]:
        """Persist FAILED/aborted results for every call still lacking a result.

        Trust durable ``machine.state.calls`` (result_json or terminal status),
        not in-process ``_finished_call_ids`` — cancel can land after save and
        before local bookkeeping, and must not clobber a completed result.
        """
        aborted: list[tuple[str, str]] = []
        terminal = {
            CallStatus.COMPLETED,
            CallStatus.FAILED,
            CallStatus.UNCERTAIN,
        }
        for call in calls:
            current = next(
                (c for c in machine.state.calls if c.call_id == call.call_id),
                call,
            )
            if current.result_json is not None:
                aborted.append((call.call_id, current.result_json))
                continue
            if current.status in terminal:
                # Terminal without a result payload (e.g. UNCERTAIN): do not invent cancel.
                continue
            result_json = json.dumps(
                {
                    "ok": False,
                    "content": "",
                    "error": f"tool call cancelled: {call.name}",
                },
                ensure_ascii=False,
            )
            await machine.dispatch(
                ToolCallFinished(
                    call_id=call.call_id,
                    status=CallStatus.FAILED,
                    result_json=result_json,
                    at=utc_now_iso(),
                )
            )
            self._finished_call_ids.add(call.call_id)
            aborted.append((call.call_id, result_json))
        return aborted

    async def _effect_await_confirmation(self, state: AgentState) -> EffectResult:
        """Prompt via InteractivePort; resolve; return ConfirmationResolved.

        ``state`` has already been persisted as awaitingconfirmation before this
        effect runs, so public ``state`` precedes ``interactive_request``.
        """
        # Confirmation already encodes the resume decision; do not re-classify
        # writes as uncertain when tooluse runs after resolve.
        self._is_resume = False
        conf = state.confirmation
        now = utc_now_iso()
        if conf is None:
            return EffectResult(ConfirmationResolved(approved=False, at=now))

        spec = self._pending_confirm_spec
        if spec is None:
            spec = ConfirmationSpec(
                kind=conf.kind,
                prompt=conf.prompt,
                options=tuple(conf.options),
                category=conf.category,
                call_ids=tuple(conf.call_ids),
                multi_select=conf.multi_select,
            )
        self._pending_confirm_spec = None

        answer: str | None = None
        port = self.interactive
        if port is not None and hasattr(port, "prompt"):
            answer = await port.prompt(spec)
        elif callable(port):
            kind = "question" if spec.kind == "question" else "confirm"
            answer = await port(
                kind,
                spec.prompt,
                list(spec.options),
                multi_select=spec.multi_select,
            )
        elif spec.kind != "question":
            # Legacy PermissionGate(confirm=...) without InteractivePort.
            callback = getattr(self.gate, "confirm", None)
            if callable(callback):
                for call in state.calls:
                    if call.call_id not in (spec.call_ids or conf.call_ids):
                        continue
                    approved_bool = await callback(call.name, dict(call.args))
                    answer = "y" if approved_bool else "n"
                    break

        if spec.kind == "question":
            self._interaction_answer = answer
            approved = True
        else:
            resolve = getattr(self.gate, "resolve", None)
            if callable(resolve):
                decision = resolve(spec, answer)
                approved = decision.verdict is Verdict.ALLOW
            else:
                approved = answer in {"y", "y_remember", "y_session"}

        if approved:
            self._confirmation_approved_ids.update(spec.call_ids or conf.call_ids)

        return EffectResult(
            ConfirmationResolved(approved=approved, answer=answer, at=now)
        )

    async def _finish_run(self, state: AgentState) -> AgentTurnOutcome:
        """Emit answer/error/done and build AgentTurnOutcome from loop + counters."""
        tool_calls = state.tool_calls
        if state.phase is AgentPhase.COMPLETE and state.outcome is not None:
            if state.outcome.kind == "stopped":
                return await self._stop_turn(tool_calls=tool_calls)
            if state.outcome.kind == "succeeded":
                answer = self._pending_answer
                await self._emit("answer", {"text": answer})
                await self._emit(
                    "done",
                    self._done_payload(
                        succeeded=True, reason=None, tool_calls=tool_calls
                    ),
                )
                self.memory.append_assistant(Msg(role="assistant", content=answer))
                return AgentTurnOutcome(
                    answer=answer,
                    usage=self.usage,
                    tool_calls=tool_calls,
                    retry_count=self.stats.retry_count,
                    tool_duration_ms=self.stats.snapshot().tool_duration_ms,
                    citations=self._citation_ids(),
                    trace=list(self.trace),
                    rounds=self.rounds,
                )
            # Other complete kinds (e.g. abandoned): treat as unsuccessful finish.
            reason = state.outcome.reason or state.outcome.kind
            return await self._fail(
                kind=state.outcome.kind,
                message=reason or state.outcome.kind,
                reason=reason,
                tool_calls=tool_calls,
                error=None,
                answer=self._pending_answer,
            )

        if state.phase is AgentPhase.ERROR and state.error is not None:
            reason = self._pending_reason or state.error.kind
            return await self._fail(
                kind=self._pending_fail_kind or state.error.kind,
                message=self._pending_fail_message or state.error.message,
                reason=reason,
                tool_calls=tool_calls,
                error=self._pending_error,
                answer=self._pending_answer,
            )

        # Defensive fallback — should not be reached after a normal runner exit.
        return await self._fail(
            kind="internal_error",
            message="agent runner finished without a terminal outcome",
            reason="internal_error",
            tool_calls=tool_calls,
            error="internal_error",
            answer=self._pending_answer,
        )

    def _truncate(self, content: str, limit: int, *, recovery_path: str | None = None) -> str:
        """将单个工具结果限制在 ``limit`` 个 token 以内。

        预算以 token 为单位命名，因此截断依据的是真实的 token 计数而非字符长度
        （字符上限会让中文文本膨胀到约两倍的预期预算）。``limit`` 由
        :func:`finharness.shared.budget.resolve_result_budget` 解析后传入，使渲染侧与
        编码侧用同一个数字——过去两侧各自读取全局设置，导致 ``detail="full"`` 放宽
        渲染出的内容又被这里砍回原预算。

        标记必须可据以行动：它给出被省略的规模与完整内容所在（若工具已落盘），
        而不只是一个 ``[truncated]``。这正是"只看到研报首页"曾经无法被诊断的原因——
        读者既不知道丢了多少，也不知道去哪里拿回来。
        """
        if limit <= 0 or self.memory.counter.count(content).tokens <= limit:
            return content
        # 截断可能把 </web_result> 切掉而留下悬空围栏——"围栏内是引文"的边界
        # 由此退化为含糊。补齐闭合优先于截断标记：完整性是安全属性，标记只是
        # 信息。渲染层的中和保证此刻前缀里的 <web_result 只能是真围栏，而围栏
        # 总是配对发出（悬空的至多最后一段），因此扣除闭合开销重截一轮即可收敛；
        # 三轮上限只是对异常输入的防御。
        budget = limit
        for _ in range(3):
            prefix = self._fit_prefix(content, budget)
            closed = close_dangled(prefix)
            over = self.memory.counter.count(closed).tokens - limit
            if over <= 0:
                break
            budget -= over + 1
        prefix = close_dangled(prefix)
        marker = self._truncation_marker(prefix, content, recovery_path=recovery_path)
        if self.memory.counter.count(marker).tokens >= limit:
            return prefix  # 连标记都放不下：宁可无标记，也不超出预算
        return prefix.rstrip() + marker

    def _fit_prefix(self, content: str, limit: int) -> str:
        """二分查找 token 预算内最长的前缀；计数是单调的。"""
        low, high = 0, len(content)
        while low < high:
            middle = (low + high + 1) // 2
            if self.memory.counter.count(content[:middle]).tokens <= limit:
                low = middle
            else:
                high = middle - 1
        return content[:low]

    def _truncation_marker(
        self, prefix: str, content: str, *, recovery_path: str | None
    ) -> str:
        """说明被省略了什么，以及完整内容在哪里可以取回。

        ``TRUNCATION_MARKER`` 作为可检索的哨兵前缀保留（日志/断言据它识别截断），
        其后追加本次的实际规模与取回位置。
        """
        kept = self.memory.counter.count(prefix).tokens
        total = self.memory.counter.count(content).tokens
        note = f"{self.TRUNCATION_MARKER}已省略约 {total - kept}/{total} token"
        if recovery_path:
            note += f"；完整内容见 {recovery_path}（可用 read_file / read_pdf 分页读取）"
        return note

    def _encode(self, result: ToolResult, limit: int) -> str:
        payload: dict[str, Any] = {
            "ok": bool(result.ok),
            "content": self._truncate(
                result.content, limit, recovery_path=self._recovery_path(result)
            ),
            "error": result.error,
        }
        if result.citations:
            payload["citations"] = list(result.citations)
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _recovery_path(result: ToolResult) -> str | None:
        """结果被裁剪时，指向可读回完整内容的本地载荷。

        工具把它落盘的位置写在 ``RawData.parquet_path`` 上（数据框）或
        ``RawData.paths``（产物文件）；两者都能让读者取回比上下文更完整的版本。
        """
        for source in result.sources or ():
            for candidate in (
                getattr(source, "parquet_path", None),
                *list(getattr(source, "paths", None) or ()),
            ):
                if candidate:
                    return str(candidate)
        return None

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
                action_name = next(
                    (action.name for action in actions if action.call_id == call_id),
                    "",
                )
                meta = ObservedCall(
                    call_id=call_id,
                    name=action_name,
                    ok=ok,
                    error=error,
                    preview=preview,
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
        self,
        tool_use: ToolUse,
        message: str,
        *,
        duration_ms: int | None = None,
        verdict: str | None = None,
    ) -> tuple[str, str]:
        """拒绝一次工具调用：发送失败状态、记录观测并返回编码后的结果。

        ``verdict`` 是拒绝的结构性原因（denied/blocked/loop_guard/timeout/
        unknown/lazy），监控侧据此把「安全拦截」与「工具失败」分开统计；
        None 表示普通工具执行失败。
        """
        status: dict[str, Any] = {
            "call_id": tool_use.call_id,
            "name": tool_use.name,
            "status": "failed",
            "ok": False,
            "error": message,
        }
        if verdict is not None:
            status["verdict"] = verdict
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
            return await self._reject(
                tool_use, f"unknown tool: {tool_use.name}", verdict="unknown"
            )

        # 按需工具的激活闸门：其 schema 此前从未注入，因此模型现在调用它意味着它凭
        # 名称猜到了工具。这不再是错误——注册层知道全部工具，按需注入只是省 token 的
        # 手段而非权限——因此就地激活并继续执行，而不是拒绝一次本可完成的调用。
        # 只在 registry 确实对工具分层时才应用（收窄后的 sub-agent registry 不携带分层）。
        lazy = getattr(self.registry, "lazy_names", None)
        is_active = getattr(self.registry, "is_active", None)
        if (
            callable(lazy)
            and callable(is_active)
            and tool_use.name in set(lazy())
            and not is_active(tool_use.name)
        ):
            self.registry.activate(tool_use.name)
            self.ctx.activate_tool(tool_use.name)
            # 单独记一个属性而不是覆盖 status：本次调用随后仍会跑完并得到自己的
            # 终态（ok/error/timeout），而"这是一次按需激活"是额外事实，不是终态。
            span.set_attribute("lazy_activated", True)
            await self._emit(
                "tool_activated",
                {"call_id": tool_use.call_id, "name": tool_use.name},
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
                    "turn": self.turn,
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
        # 已经过 FSM 确认的调用跳过闸门，避免 plain "y" 再次变成 CONFIRM。
        if tool_use.call_id in self._skip_permission_ids:
            self._skip_permission_ids.discard(tool_use.call_id)
            decision = GateDecision(Verdict.ALLOW, "用户已确认")
        else:
            decide = getattr(self.gate, "decide", None)
            if callable(decide):
                decision = decide(tool, tool_use.args)
                if decision.verdict is Verdict.CONFIRM:
                    # 不应在 execute 路径上再 await 用户；当作拒绝以免卡死。
                    decision = GateDecision(
                        Verdict.DENY,
                        decision.reason
                        or f"写类工具需确认，但当前无交互通道：{tool.name}",
                        confirmation=None,
                    )
            else:
                decision = await self.gate.check(tool, tool_use.args)
        if decision.verdict is Verdict.DENY:
            await self._audit(tool, tool_use.args, action="denied", verdict="deny")
            span.set_attribute("status", "denied")
            return await self._reject(
                tool_use,
                decision.reason or f"tool denied: {tool_use.name}",
                verdict="denied",
            )
        if not await self.hooks.pre(tool, tool_use.args, turn=self.turn):
            await self._audit(tool, tool_use.args, action="denied", verdict="blocked")
            span.set_attribute("status", "blocked")
            return await self._reject(
                tool_use, f"tool blocked by hook: {tool_use.name}", verdict="blocked"
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
        if getattr(tool, "needs_interactive", False):
            captured = self._interaction_answer
            self._interaction_answer = None

            async def _oneshot(
                kind, prompt, options, *, multi_select: bool = False
            ):
                del kind, prompt, options, multi_select
                return captured

            tool.interactive = _oneshot
        # 对派发 sub-agent 的工具采用同样的模式（docs 03.10）：工具自身无法
        # 构建协调器，因此由循环把会话的协调器交给它。
        if getattr(tool, "needs_coordinator", False) and self.coordinator is not None:
            tool.coordinator = self.coordinator
        # 隔离计算通道同样是"工具自己够不到、由循环交给它"（docs 03.15）：
        # 会话身份（user/conversation）只在此处可知，工具不该自己拼。
        # 未配置远程 worker 时为 None，工具按声明回退进程内执行。
        if getattr(tool, "needs_compute", False) and self.compute is not None:
            tool.compute = self.compute
        # 长时间工具上报中间进展：call_id 与工具名由循环补齐，工具只描述进展内容，
        # 无需知道自己在对话里的调用标识。这不是装饰——服务端心跳是 SSE 注释帧，
        # 客户端看门狗只在真实事件到达时才重置，静默数分钟的工具会被判为卡死而掐断。
        if getattr(tool, "needs_progress", False):
            tool.progress = self._progress_emitter(tool_use)
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
                verdict="timeout",
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
                tool_use, f"tool failed: {exc}", duration_ms=duration_ms, verdict="error"
            )

        duration_ms = self.stats.record_tool_duration(tool_use.name, started_at)
        span.set_attribute("status", "ok" if result.ok else "error")
        result.citations = self._register_citations(tool_use, result)
        self._note_review_outcome(result)
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
        return tool_use.call_id, self._encode(
            result, self._result_budget(tool, tool_use.args)
        )

    def _result_budget(self, tool: Any, args: dict[str, Any]) -> int:
        """按 运维覆盖 → 工具声明 → 全局 解析本次结果的 token 预算。

        这里与工具渲染时（``BaseTool._result_token_budget``）走的是同一个解析器，
        因此渲染放行的内容不会被这里再砍回去（docs 03.3.3）。
        """
        detail = str((args or {}).get("detail") or "summary")
        return resolve_result_budget(
            settings=self.settings,
            tool_name=getattr(tool, "name", ""),
            tool=tool,
            detail=detail,
        )

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
        except Exception:  # noqa: BLE001 - 审计失败绝不能中断一轮
            self._log_audit_failure(action, getattr(tool, "name", "?"))

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
