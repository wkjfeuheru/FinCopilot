"""子代理协调器：在隔离上下文中执行聚焦范围的工作（docs 03.10）。

只有当隔离能换来某些东西时，子代理才配得上它的开销。上下文压缩已经能回收窗口
空间，并行工具调用已经能覆盖速度，所以这两者都不是派生一个子代理的理由。真正
的理由有两个：

* **一次独立阅读。** 风险复核者从未撰写该研报，并且可以重新获取底层数据，因此
  它能看见作者在自己草稿中看不到的东西（``focus="risk"``）。
* **上下文隔离。** 一个中间材料会挤占主窗口的任务——例如消化三份长文档——可以
  在它自己的上下文中运行至完成，只回传其结论（``focus="general"``）。

隔离是通过结构而非指令来强制的：

* 每个子代理都获得全新的 ``ResearchContext``、transcript 与 stats，因此它看不到
  主对话的推理，且它所做的任何事都不会进入对话记忆（``store=None``）；
* 它的工具目录在构造时就被收窄（``only=<focus tool names>``），因此子集之外的
  名称根本无法解析——又因为 ``spawn_agent`` 位于 META 组，子代理在结构上无法再
  派生另一个子代理；
* 引用登记表（citation registry）是共享的，使 cid 在整个会话中保持连续，但每个
  子代理通过 ``ScopedCitationRegistry`` 写入，因此它铸造的 cid 可以被精确知晓，
  无需对共享存储做差集比较（两个并发的子代理会把它算错）。

子代理不编辑任何东西：它返回发现，由主代理决定如何处理它们（docs 03.10）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from finharness.config.settings import Settings
from finharness.data.access import DataAccess
from finharness.data.citation import CitationRegistry, ScopedCitationRegistry

# 聚焦名与派发上限是 tools.spawn_agent 与协调器之间的**协议**，因此定义在
# shared（tools 也要读它），这里只消费。
from finharness.shared.agents import GENERAL_FOCUS, MAX_SPAWN_TASKS, RISK_FOCUS

# 读取研报，重新获取并交叉核对其中若干数字，然后写出复核意见。三轮曾经过紧：
# 一份有几个数字需要核验的研报会在写出任何意见之前就耗尽轮次（表现为
# ``max_turns_exhausted``），从而静默地交付一份未经复核的研报。六轮为读取加上
# 若干独立核查留出了余量，又不会允许出现死循环。
REVIEW_MAX_TURNS = 6
# worker 读取其材料并写出结论；它不获取数据，因此需要的轮次远少于复核者。
WORKER_MAX_TURNS = 4


@dataclass(frozen=True, slots=True)
class SubAgentResult:
    """子代理返回的内容：它的结论以及所产生的开销。"""

    focus: str
    summary: str
    citations: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    ok: bool = True
    error: str | None = None
    task: str = ""


@dataclass(frozen=True, slots=True)
class _Focus:
    """一个子代理角色：被告知什么、可以调用什么、可以运行多久。"""

    name: str
    system: Callable[[], str]
    tool_names: Callable[[], tuple[str, ...]]
    max_turns: int


def _risk_system() -> str:
    """角色 prompt 加上风险核查清单，使判据无需重复陈述。

    该清单与情景研报模板所描述的资产相同，因此复核者的判据与作者的判据保持
    一致。清单缺失时降级为仅使用角色 prompt，而不是让复核失败。
    """
    from finharness.engine.prompt import risk_checklist_prompt, risk_review_prompt

    sections = [risk_review_prompt()]
    try:
        body = risk_checklist_prompt()
    except Exception:  # noqa: BLE001 - 没有它复核依然有用
        body = ""
    if body:
        sections.append("## 风险核查清单（判据来源）\n\n" + body)
    return "\n\n".join(sections)


def _risk_tools() -> tuple[str, ...]:
    from finharness.tools.registry import review_tool_names

    return review_tool_names()


def _general_system() -> str:
    from finharness.engine.prompt import worker_prompt

    return worker_prompt()


def _worker_tools() -> tuple[str, ...]:
    from finharness.tools.registry import worker_tool_names

    return worker_tool_names()


_FOCUSES: dict[str, _Focus] = {
    RISK_FOCUS: _Focus(
        name=RISK_FOCUS,
        system=_risk_system,
        tool_names=_risk_tools,
        max_turns=REVIEW_MAX_TURNS,
    ),
    GENERAL_FOCUS: _Focus(
        name=GENERAL_FOCUS,
        system=_general_system,
        tool_names=_worker_tools,
        max_turns=WORKER_MAX_TURNS,
    ),
}


def focus_names() -> tuple[str, ...]:
    """调用方可以请求的子代理角色。"""
    return tuple(sorted(_FOCUSES))


class Coordinator:
    """在隔离上下文中运行聚焦范围的子代理（docs 03.10）。"""

    def __init__(
        self,
        *,
        provider: Any,
        data: DataAccess,
        settings: Settings,
        cite: CitationRegistry,
        counter: Any | None = None,
        on_usage: Callable[[str, int, int], None] | None = None,
        stop_signal_provider: Callable[[], Any | None] | None = None,
        audit_hook_factory: Callable[[str], Any] | None = None,
        user_id: str = "",
    ) -> None:
        self.provider = provider
        self.data = data
        self.settings = settings
        self.cite = cite
        self.counter = counter
        self._on_usage = on_usage
        # 由持有它的循环接线，使子代理的开销计入会话总量。即使没有接线，子代理
        # 仍会运行，只是其开销不会被计费。
        self._usage: Any | None = None
        self._stats: Any | None = None
        # 子代理的 LLM Span 需要与主循环区分（call_type=subagent）；未接线时
        # 子代理仍会运行，只是不产生自己的 Span。
        self._observer: Any | None = None
        # 主循环停止信号的取值器（docs 03.3）：子代理在启动前取一次，因此用户
        # 在主循环等待子代理扇出时按下的停止也能被它们看到。传取值器而非信号
        # 本身，因为信号是在 loop 构造之后才装上的。
        self._stop_signal_provider = stop_signal_provider
        # 子代理审计（docs 03.7.3）：审计不可移除是全局不变量，子代理不应是
        # 盲区。工厂按子代理 session_id 产出 hook，使复核者与 worker 的行为
        # 各自落行；未接线时子代理仍运行，只是不留痕（测试替身路径）。
        self._audit_hook_factory = audit_hook_factory
        self._user_id = user_id

    def bind_accounting(self, *, usage: Any, stats: Any, observer: Any | None = None) -> None:
        """挂接主循环的计数器与观测器，使子代理的开销被记到那里。"""
        self._usage = usage
        self._stats = stats
        if observer is not None:
            self._observer = observer

    def _current_stop_signal(self) -> Any | None:
        """当前主循环的停止信号；未接线时返回 None（子代理永不因此停下）。"""
        if self._stop_signal_provider is None:
            return None
        try:
            return self._stop_signal_provider()
        except Exception:  # noqa: BLE001 - 取值失败只是"子代理不响应停止"
            return None

    # -- 风险复核 ----------------------------------------------------------
    async def review_risk(self, *, topic: str, markdown: str) -> SubAgentResult:
        """复核一份已渲染的研报；绝不抛出异常，因此研报始终成立。"""
        try:
            return await self._run(
                focus=RISK_FOCUS,
                task=_review_request(topic=topic, markdown=markdown),
                context=None,
            )
        except Exception as exc:  # noqa: BLE001 - 复核失败绝不能是致命的
            return SubAgentResult(
                focus=RISK_FOCUS,
                summary="",
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    # -- 通用扇出 ------------------------------------------------------
    async def spawn(
        self,
        *,
        tasks: list[str],
        focus: str = GENERAL_FOCUS,
        context: str | None = None,
    ) -> list[SubAgentResult]:
        """把独立任务作为并发子代理运行并收集结论。

        结果按任务给定的顺序返回，每个结果都如实说明自身的结果：一个任务失败
        不会中止它的兄弟任务。此方法绝不抛出异常——调用方会为每个任务得到一个
        结果，其中失败者会被标记出来。
        """
        if focus not in _FOCUSES:
            raise ValueError(f"未知的子代理类型：{focus}（可选 {'、'.join(sorted(_FOCUSES))}）")
        cleaned = [task.strip() for task in tasks if task and task.strip()]
        if not cleaned:
            raise ValueError("子任务列表不能为空")
        if len(cleaned) > MAX_SPAWN_TASKS:
            raise ValueError(f"单次最多派发 {MAX_SPAWN_TASKS} 个子任务（收到 {len(cleaned)}）")

        # 并发是刻意设计的：这些任务彼此独立，这正是它们能够被拆分的前提。每个
        # _run 已经会吞掉自身的失败，但 gather 也被包裹起来，这样 harness 中意外
        # 的抛出就不会丢失兄弟任务的结果。
        gathered = await asyncio.gather(
            *(self._run(focus=focus, task=task, context=context) for task in cleaned),
            return_exceptions=True,
        )
        results: list[SubAgentResult] = []
        for task, item in zip(cleaned, gathered):
            if isinstance(item, SubAgentResult):
                results.append(item)
            else:
                results.append(
                    SubAgentResult(
                        focus=focus,
                        summary="",
                        task=task,
                        ok=False,
                        error=f"{type(item).__name__}: {item}",
                    )
                )
        return results

    # -- 执行 ------------------------------------------------------------
    async def _run(self, *, focus: str, task: str, context: str | None) -> SubAgentResult:
        """构建并运行一个子代理；绝不抛出异常。

        一旦子循环存在，它记录的用量就是这次运行开销的真相——包括中途失败但
        已经消耗了 token 的运行——因此两条路径都据此计账。
        """
        from finharness.context.session import ResearchContext
        from finharness.engine.loop import AgentLoop
        from finharness.permissions.gate import ReadOnlyGate
        from finharness.tools.registry import ToolRegistry

        spec = _FOCUSES[focus]
        try:
            system = spec.system()
            tool_names = spec.tool_names()
        except Exception as exc:  # noqa: BLE001 - 资源损坏只是本任务自己的失败
            return SubAgentResult(
                focus=focus, summary="", task=task, ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        # 派生的 settings 对象：收紧子代理的轮次预算不得影响主循环的，且它共享
        # 主缓存目录。
        sub_settings = self.settings.model_copy(
            update={
                "context": self.settings.context.model_copy(
                    update={"max_turns": spec.max_turns}
                )
            }
        )

        # 共享登记表以保证 cid 连续，使用带作用域的写入器，使归属即使在多个子
        # 代理同时运行时也保持精确。
        scoped_cite = ScopedCitationRegistry(self.cite)
        sub_ctx = ResearchContext(cite=scoped_cite, settings=sub_settings)
        sub_session_id = f"{focus}-subagent"
        # 子代理审计行与主会话可区分（session_id 带 focus 前缀），但归属于同一
        # 用户与写入器：检索 audit.jsonl 时不必先知道会话结构也能回答"谁做的"。
        from finharness.hooks.base import HookChain

        sub_hooks = HookChain(
            [self._audit_hook_factory(sub_session_id)]
            if self._audit_hook_factory is not None
            else []
        )

        sub_loop = AgentLoop(
            provider=self.provider,
            registry=ToolRegistry(
                self.data, ctx=sub_ctx, settings=sub_settings, only=set(tool_names)
            ),
            settings=sub_settings,
            system=system,
            cite=scoped_cite,
            ctx=sub_ctx,
            gate=ReadOnlyGate(),
            counter=self.counter,
            # 不设 store：这次运行是它自己的一个片段，而非对话记忆。它的
            # transcript 随本次调用一起消亡。
            store=None,
            session_id=sub_session_id,
            hooks=sub_hooks,
            user_id=self._user_id,
            # 子代理的模型调用以 call_type=subagent 标记，使其 token 与耗时
            # 在指标里与主循环分开；它的 ``run()`` 也因此不发射请求级指标，
            # 避免把子代理算作一次用户请求。
            observer=self._observer,
            call_type="subagent",
            # 场景路由属于主循环：子代理的输入只有任务文本，据此推断意图只会往核查者
            # 的上下文里塞进与本次核查无关的方法论，并平白多花 token。
            route_skills=False,
        )
        # 子代理与主循环共享同一个停止信号：否则用户在主循环等子代理扇出时按
        # 停止，要等这个子代理自己跑完才生效。共享而非复制，使置位即刻可见。
        sub_loop.stop_signal = self._current_stop_signal()

        try:
            outcome = await sub_loop.run(_sub_request(task=task, context=context))
        except Exception as exc:  # noqa: BLE001 - 先为已消耗的部分计费，再报告
            result = self._result_from(
                sub_loop, scoped_cite, focus=focus, task=task, ok=False, error=str(exc)
            )
            self._account(result)
            return result

        result = self._result_from(
            sub_loop,
            scoped_cite,
            focus=focus,
            task=task,
            ok=bool(outcome.succeeded),
            error=None if outcome.succeeded else (outcome.error or outcome.reason),
            summary=(outcome.answer or "").strip(),
        )
        self._account(result)
        return result

    def _result_from(
        self,
        sub_loop: Any,
        scoped_cite: ScopedCitationRegistry,
        *,
        focus: str,
        task: str,
        ok: bool,
        error: str | None,
        summary: str = "",
    ) -> SubAgentResult:
        """从子循环的真实计数器构建结果，无论成功与否。"""
        return SubAgentResult(
            focus=focus,
            summary=summary,
            # 精确归属：本作用域铸造的 id，而非共享状态的差集。
            citations=scoped_cite.created,
            input_tokens=int(getattr(sub_loop.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(sub_loop.usage, "output_tokens", 0) or 0),
            turns=int(getattr(sub_loop, "turn", 0) or 0),
            ok=ok,
            error=error,
            task=task,
        )

    def _account(self, result: SubAgentResult) -> None:
        """把本次运行的 token 汇入会话总量与按代理的明细。"""
        if self._usage is not None:
            self._usage.input_tokens += result.input_tokens
            self._usage.output_tokens += result.output_tokens
        if self._stats is not None:
            # add_usage 会推进总量；record_agent_usage 只是给它打标签。
            self._stats.add_usage(result.input_tokens, result.output_tokens)
            self._stats.record_agent_usage(
                result.focus, result.input_tokens, result.output_tokens
            )
        if self._on_usage is not None:
            self._on_usage(result.focus, result.input_tokens, result.output_tokens)


def _sub_request(*, task: str, context: str | None) -> str:
    """子代理看到的单条用户消息：任务，加上共享背景。"""
    parts: list[str] = []
    if context:
        parts.append(f"共享背景：\n{context}\n")
    parts.append(f"你的任务：\n{task}\n")
    return "\n".join(parts)


def _review_request(*, topic: str, markdown: str) -> str:
    """构造风险复核者的用户请求：主题、去掉附录的正文与逐项核查指令。"""
    return (
        f"请对以下研报做风险终审，主题：{topic}。\n"
        "报告正文如下（附录已略去）：\n\n"
        "<report>\n"
        f"{markdown}\n"
        "</report>\n\n"
        "按你的职责逐项核查并给出结论。"
    )
