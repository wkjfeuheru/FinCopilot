"""针对组装好的 AgentLoop 运行评测用例（docs 03.13）。

每个用例拥有各自的设置路径（cache/output/memory/audit）与独立的循环，因此各次
运行不会相互污染，也不会污染用户的真实数据。多轮用例复用同一个循环——这正是
“接上题”用例有意义的原因——同时会话以用例 id 为键，使记忆表现得如同在同一次
对话中。
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finharness.config.settings import Settings
from finharness.context.memory.store import MemoryStore
from finharness.context.session import ResearchContext
from finharness.data.access import DataAccess
from finharness.data.adapters.akshare_adapter import AkShareAdapter
from finharness.data.cache import LocalCache
from finharness.data.citation import CitationRegistry
from finharness.engine.loop import AgentLoop
from finharness.engine.prompt import system_prompt
from finharness.eval.capture import CapturedTurn, InteractionChannel, RecordingSink
from finharness.eval.schema import EvalCase
from finharness.hooks.audit import AuditHook, AuditLogWriter
from finharness.hooks.base import HookChain
from finharness.permissions.gate import PermissionGate
from finharness.tools.registry import ToolRegistry
from finharness.utils.jsonx import stable_dumps


@dataclass(slots=True)
class CaseRun:
    """单个用例的完整记录：逐轮结果、产物与审计。"""

    case: EvalCase
    turns: list[CapturedTurn] = field(default_factory=list)
    error: str | None = None
    duration_ms: int = 0
    exports: list[str] = field(default_factory=list)
    report_text: str = ""
    audit_denials: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    run_dir: str = ""

    @property
    def case_id(self) -> str:
        return self.case.id

    def called_tools(self) -> list[str]:
        """实际执行成功的工具名（observation ok=True）。"""
        names: list[str] = []
        for turn in self.turns:
            for round_trace in turn.trace:
                for observation in round_trace.observations:
                    if observation.ok:
                        names.append(observation.name)
        return names

    def attempted_tools(self) -> list[str]:
        """模型尝试调用的工具名，无论已执行还是被拒。"""
        names: list[str] = []
        for turn in self.turns:
            for round_trace in turn.trace:
                for action in round_trace.actions:
                    names.append(action.name)
        return names

    def refused_tools(self, reason: str | None = None) -> list[str]:
        """调用被拒（denied、lazy、blocked）即未执行的工具。"""
        names: list[str] = []
        for turn in self.turns:
            for round_trace in turn.trace:
                for observation in round_trace.observations:
                    if observation.ok:
                        continue
                    error = observation.error or ""
                    if reason is None or reason in error:
                        names.append(observation.name)
        return names

    # -- 轨迹分析辅助方法 ---------------------------------------------------
    def executed_sequence(self) -> list[str]:
        """按调用顺序排列的已执行工具名（不含被拒）。"""
        return self.called_tools()

    def executed_fingerprints(self) -> list[tuple[str, str]]:
        """按顺序给出每次已执行调用的 (工具, 参数指纹)。"""
        pairs: list[tuple[str, str]] = []
        for turn in self.turns:
            for round_trace in turn.trace:
                by_id = {obs.call_id: obs for obs in round_trace.observations}
                for action in round_trace.actions:
                    observation = by_id.get(action.call_id)
                    if observation is None or not observation.ok:
                        continue
                    pairs.append((action.name, _fingerprint(action.args)))
        return pairs

    def max_repeat_count(self) -> int:
        """同一 (工具, 参数) 调用执行次数的最大值。"""
        counts: dict[tuple[str, str], int] = {}
        for pair in self.executed_fingerprints():
            counts[pair] = counts.get(pair, 0) + 1
        return max(counts.values(), default=0)

    def plan_present(self) -> bool:
        """该用例执行期间是否形成了研究计划。"""
        for turn in self.turns:
            for data in [event.data for event in turn.events if event.kind == "plan_progress"]:
                if data:
                    return True
            done = [event.data for event in turn.events if event.kind == "done"]
            if done and done[-1].get("plan"):
                return True
        return False

    def loaded_skills(self) -> list[str]:
        """本次运行注入的方法论，从 ``context_routed`` 事件中读取。

        加载不再是工具调用，而是引擎的路由动作，因此观测点随之从工具轨迹移到事件流。
        方法名保持不变，使按"这次拿到了哪些方法"编写的用例断言无需改写。
        """
        skills: list[str] = []
        for turn in self.turns:
            for event in turn.events:
                if event.kind != "context_routed":
                    continue
                for name in event.data.get("skills") or []:
                    skills.append(str(name))
        return skills

    def degraded(self) -> bool:
        """本次运行是否以降级而非干净成功结束。"""
        return (not self.succeeded()) or self.reason() is not None

    def final_answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""

    def reason(self) -> str | None:
        for turn in reversed(self.turns):
            reason = getattr(turn.outcome, "reason", None)
            if reason:
                return reason
        return None

    def succeeded(self) -> bool:
        return all(bool(getattr(turn.outcome, "succeeded", False)) for turn in self.turns)

    def totals(self) -> dict[str, int]:
        input_tokens = sum(
            int(getattr(turn.outcome.usage, "input_tokens", 0)) for turn in self.turns
        )
        output_tokens = sum(
            int(getattr(turn.outcome.usage, "output_tokens", 0)) for turn in self.turns
        )
        tool_calls = sum(int(getattr(turn.outcome, "tool_calls", 0)) for turn in self.turns)
        rounds = sum(int(getattr(turn.outcome, "rounds", 0)) for turn in self.turns)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "tool_calls": tool_calls,
            "rounds": rounds,
        }


def isolate_settings(base: Settings, case_dir: Path) -> Settings:
    """返回 ``base`` 的副本，其可变路径均指向 ``case_dir`` 内部。

    状态文件（密钥、用户库、配置库）也一并落到用例目录：评测会反复重建应用，
    复用工作目录下的真实 ``state/`` 会让用例读到上一次运行留下的密钥与配置。
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    data = base.data.model_copy(update={"cache_dir": case_dir / "cache"})
    state_dir = case_dir / "state"
    paths = base.paths.model_copy(
        update={
            "output_dir": case_dir / "output",
            "state_dir": state_dir,
            "memory_db": state_dir / "memory.db",
            "auth_db": state_dir / "users.db",
            "config_db": state_dir / "config.db",
            "secret_key": state_dir / "secret.key",
        }
    )
    audit = base.audit.model_copy(update={"log_path": case_dir / "audit.jsonl"})
    isolated = base.model_copy(update={"data": data, "paths": paths, "audit": audit})
    (case_dir / "cache").mkdir(parents=True, exist_ok=True)
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return isolated


def build_data_access(settings: Settings, *, offline: bool) -> DataAccess:
    """组装一次运行可能用到的适配器，与生产环境的接线保持一致。

    离线模式使用提供合成数据帧的确定性适配器，使整条管线（工具、引用、报告）
    无需网络或凭证即可运行。
    """
    adapters: list[Any] = []
    if offline:
        from finharness.eval.offline_adapter import OfflineAdapter

        adapters.append(OfflineAdapter())
    else:
        # 与生产接线同一顺序：同花顺优先，akshare 兜底。
        from finharness.data.adapters.fuyao_adapter import FuyaoMcpAdapter

        adapters.append(
            FuyaoMcpAdapter(
                api_key=settings.fuyao.resolved_api_key(),
                base_url=settings.fuyao.base_url,
                timeout_s=settings.fuyao.timeout_s,
                proxy=settings.fuyao.proxy,
                throttle_seconds=settings.data.throttle_seconds,
            )
        )
        adapters.append(AkShareAdapter(throttle_seconds=settings.data.throttle_seconds))
        api_key = os.getenv(settings.search.env_key) or settings.search.api_key
        if api_key:
            from finharness.data.adapters.tavily_adapter import TavilyAdapter

            adapters.append(
                TavilyAdapter(
                    api_key=api_key,
                    base_url=settings.search.base_url,
                    timeout_s=settings.search.timeout_s,
                    proxy=settings.search.proxy,
                )
            )
        from finharness.data.adapters.eastmoney_report_adapter import (
            EastmoneyReportAdapter,
        )

        adapters.append(
            EastmoneyReportAdapter(
                timeout_s=settings.search.timeout_s,
                with_text_allowed=settings.search.local_pdf_fallback,
                pdf_dir=Path(settings.data.cache_dir) / "pdf",
            )
        )
    return DataAccess(
        adapters, cache=LocalCache(settings.data.cache_dir), settings=settings
    )


def _fingerprint(args: Any) -> str:
    """对调用参数做稳定的 JSON 渲染，用于检测重复调用。"""
    try:
        return stable_dumps(args or {})
    except (TypeError, ValueError):
        return str(args)


def _read_audit_denials(path: Path) -> list[dict[str, Any]]:
    """从审计日志中读出 verdict 为 deny/blocked 的记录。"""
    if not path.exists():
        return []
    denials: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("verdict")) in {"deny", "blocked"}:
            denials.append(record)
    return denials


def _collect_exports(output_dir: Path) -> list[str]:
    """收集输出目录下导出的 .md/.docx 文件路径。"""
    if not output_dir.exists():
        return []
    return sorted(
        str(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".docx"}
    )


class EvalRunner:
    """针对共享的 provider 逐个驱动用例运行。"""

    def __init__(
        self,
        *,
        base_settings: Settings,
        provider: Any,
        run_dir: Path,
        offline: bool = False,
        turn_timeout_s: float = 300.0,
        verbose: bool = False,
        trace_store: Any | None = None,
    ) -> None:
        self.base_settings = base_settings
        self.provider = provider
        self.run_dir = Path(run_dir)
        self.offline = offline
        self.turn_timeout_s = turn_timeout_s
        self.verbose = verbose
        # 可选 trace 落库（监控平台）：开启后每个用例轮次都以 source=eval
        # 落一行 run，使线上监控页与评测运行共享同一观测面。TraceStore 吞
        # 错，不影响评测本身。
        self.trace_store = trace_store

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    async def run_case(self, case: EvalCase) -> CaseRun:
        """运行单个用例：隔离设置、组装引擎、逐轮执行并采集结果。

        多对话用例（``chats``）依次运行，共享同一 store 与 user_id，因此
        前一个对话写入的跨对话情节（LTM）对后续对话可见——这正是
        "跨对话记忆"用例的观测方式。

        返回记录该用例全程的 CaseRun。
        """
        case_dir = self.run_dir / "cases" / case.id
        # 每个用例都从干净状态开始：同一用例重跑时不得续用上次运行持久化的
        # 会话记录，否则模型对该问题的第二次观察会与第一次不同。
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
        settings = isolate_settings(self.base_settings, case_dir)
        record = CaseRun(
            case=case,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            run_dir=str(case_dir),
        )

        sink = RecordingSink()
        channel = InteractionChannel()
        data = build_data_access(settings, offline=self.offline)
        store = MemoryStore(settings.paths.memory_db)

        started = time.monotonic()
        turn_index = 0
        try:
            for group_index, (fixed_id, turns) in enumerate(case.conversation_groups()):
                # 单对话形式沿用既有行为（以用例 id 为键）；多对话形式下
                # 没写 id 的对话按序号生成，避免与其它对话冲突。
                conversation_id = fixed_id or (
                    case.id if not case.chats else f"{case.id}#{group_index + 1}"
                )
                cite = CitationRegistry()
                ctx = ResearchContext(cite=cite, settings=settings)
                registry = ToolRegistry(data, ctx=ctx, settings=settings)
                audit = AuditHook(
                    AuditLogWriter(settings.audit.log_path),
                    session_id=conversation_id,
                )
                gate = PermissionGate(settings=settings, confirm=channel.confirm)
                loop = AgentLoop(
                    provider=self.provider,
                    registry=registry,
                    settings=settings,
                    system=system_prompt(),
                    cite=cite,
                    session_id=conversation_id,
                    ctx=ctx,
                    gate=gate,
                    hooks=HookChain([audit]),
                    conversation_id=conversation_id,
                    store=store,
                    output=sink,
                )
                # InteractivePort: FSM confirmation uses channel.prompt(spec).
                # PermissionGate still receives confirm=channel.confirm above.
                loop.interactive = channel
                for turn in turns:
                    channel.reset(turn.interactive, turn.interactive_answer)
                    sink.events = []
                    turn_started = time.monotonic()
                    outcome = await asyncio.wait_for(
                        loop.run(turn.user), self.turn_timeout_s
                    )
                    captured = CapturedTurn(
                        index=turn_index,
                        user=turn.user,
                        outcome=outcome,
                        events=list(sink.events),
                        interactions=list(channel.log),
                        duration_ms=round((time.monotonic() - turn_started) * 1000),
                    )
                    record.turns.append(captured)
                    self._record_trace(case, conversation_id, turn, captured)
                    turn_index += 1
                    self._log(
                        f"  [{case.id}] turn {turn_index} ({conversation_id}): "
                        f"succeeded={outcome.succeeded} "
                        f"tools={outcome.tool_calls} tokens="
                        f"{outcome.usage.input_tokens + outcome.usage.output_tokens}"
                    )
                # 跨对话记忆用例：对话结束后蒸馏一次，使 decision/excerpt
                # 情节对后续对话可见。task_result 情节每轮已写入，不依赖这里；
                # 蒸馏失败（离线自检 provider 不产出 JSON 即属此列）也不影响
                # 用例成立，因此吞掉异常。
                if case.distill_between_chats and case.chats:
                    try:
                        from finharness.context.memory.distill import EpisodeDistiller

                        distiller = EpisodeDistiller(
                            provider=self.provider, store=store, settings=settings
                        )
                        await distiller.distill_conversation(
                            conversation_id, user_id=""
                        )
                    except Exception:  # noqa: BLE001 - 蒸馏失败不影响用例
                        pass
        except Exception as exc:  # noqa: BLE001 - 失败的用例是数据，而非崩溃
            record.error = f"{type(exc).__name__}: {exc}"
            self._log(f"  [{case.id}] error: {record.error}")
        finally:
            record.duration_ms = round((time.monotonic() - started) * 1000)
            record.exports = _collect_exports(settings.paths.output_dir)
            record.report_text = self._read_reports(record.exports)
            record.audit_denials = _read_audit_denials(settings.audit.log_path)
        return record

    def _record_trace(self, case: EvalCase, conversation_id: str, turn: Any, captured: CapturedTurn) -> None:
        """把一个评测轮次落进监控 trace 库（未启用时为 no-op）。

        run_id 必须可重入稳定：同一用例重跑覆盖旧行，监控页看到的是最新
        一次评测的轨迹，而不是历史叠加。
        """
        store = self.trace_store
        if store is None:
            return
        try:
            run_id = f"tr_eval_{case.id}_{captured.index}"
            store.start_run(
                run_id=run_id,
                source="eval",
                user_id="",
                session_id=None,
                conversation_id=conversation_id,
                eval_case_id=case.id,
                input=turn.user,
            )
            for event in captured.events:
                if event.kind != "text_delta":
                    store.record_event(run_id, event.kind, dict(event.data))
            outcome = captured.outcome
            usage = getattr(outcome, "usage", None)
            store.finish_run(
                run_id,
                status="done" if getattr(outcome, "succeeded", False) else "error",
                answer=str(getattr(outcome, "answer", "") or ""),
                reason=getattr(outcome, "reason", None),
                succeeded=bool(getattr(outcome, "succeeded", False)),
                rounds=getattr(outcome, "rounds", None),
                tool_calls=getattr(outcome, "tool_calls", None),
                retry_count=getattr(outcome, "retry_count", None),
                usage={
                    "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                },
                citations=list(getattr(outcome, "citations", []) or []),
                trace_rounds=list(getattr(outcome, "trace", []) or []),
            )
        except Exception:  # noqa: BLE001 - trace 落库绝不影响评测
            self._log(f"  [{case.id}] trace record failed")

    @staticmethod
    def _read_reports(exports: list[str]) -> str:
        """拼接导出的 markdown，供产物检查项检视。"""
        chunks: list[str] = []
        for path_str in exports:
            path = Path(path_str)
            if path.suffix.lower() != ".md":
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8"))
            except OSError:
                continue
        return "\n".join(chunks)


def build_manifest(
    *, settings: Settings, provider: Any, offline: bool, case_count: int, set_name: str
) -> dict[str, Any]:
    """记录生成本次运行的要素：模型、provider、环境、代码版本。"""
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "set": set_name,
        "cases": case_count,
        "offline": offline,
        "provider": type(provider).__name__,
        "model": settings.model.model_name,
        "provider_name": settings.model.provider,
        "temperature": settings.model.temperature,
        "max_tokens": settings.model.max_tokens,
        "max_turns": settings.context.max_turns,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_version": _code_version(),
    }


def _code_version() -> str:
    """读取 git 短版本号；失败时返回 unknown。"""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 版本号属于锦上添花，绝不致命
        return "unknown"


__all__ = [
    "CaseRun",
    "EvalRunner",
    "build_data_access",
    "build_manifest",
    "isolate_settings",
]
