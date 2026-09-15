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
from datetime import datetime, timezone
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

    def all_traces(self) -> list[Any]:
        return [round_trace for turn in self.turns for round_trace in turn.trace]

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

    def all_interactions(self) -> list[Any]:
        return [item for turn in self.turns for item in turn.interactions]

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

    def plan_max_revision(self) -> int:
        revisions = [
            int(event.data.get("revision", 1))
            for turn in self.turns
            for event in turn.events
            if event.kind == "plan_progress"
        ]
        return max(revisions, default=0)

    def loaded_skills(self) -> list[str]:
        """模型加载的技能名，从 load_skill 调用中解析得到。"""
        skills: list[str] = []
        for turn in self.turns:
            for round_trace in turn.trace:
                for action in round_trace.actions:
                    if action.name != "load_skill":
                        continue
                    name = str((action.args or {}).get("name") or "")
                    if name:
                        skills.append(name)
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
    """返回 ``base`` 的副本，其可变路径均指向 ``case_dir`` 内部。"""
    case_dir.mkdir(parents=True, exist_ok=True)
    data = base.data.model_copy(update={"cache_dir": case_dir / "cache"})
    paths = base.paths.model_copy(
        update={
            "output_dir": case_dir / "output",
            "memory_db": case_dir / "memory.db",
        }
    )
    audit = base.audit.model_copy(update={"log_path": case_dir / "audit.jsonl"})
    isolated = base.model_copy(update={"data": data, "paths": paths, "audit": audit})
    (case_dir / "cache").mkdir(parents=True, exist_ok=True)
    (case_dir / "output").mkdir(parents=True, exist_ok=True)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
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
            )
        )
    return DataAccess(
        adapters, cache=LocalCache(settings.data.cache_dir), settings=settings
    )


def _fingerprint(args: Any) -> str:
    """对调用参数做稳定的 JSON 渲染，用于检测重复调用。"""
    try:
        return json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
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
    ) -> None:
        self.base_settings = base_settings
        self.provider = provider
        self.run_dir = Path(run_dir)
        self.offline = offline
        self.turn_timeout_s = turn_timeout_s
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    async def run_case(self, case: EvalCase) -> CaseRun:
        """运行单个用例：隔离设置、组装引擎、逐轮执行并采集结果。

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
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            run_dir=str(case_dir),
        )

        sink = RecordingSink()
        channel = InteractionChannel()
        data = build_data_access(settings, offline=self.offline)
        cite = CitationRegistry()
        ctx = ResearchContext(cite=cite, settings=settings)
        registry = ToolRegistry(data, ctx=ctx, settings=settings)
        store = MemoryStore(settings.paths.memory_db)
        audit = AuditHook(AuditLogWriter(settings.audit.log_path), session_id=case.id)
        gate = PermissionGate(settings=settings, confirm=channel.confirm)
        loop = AgentLoop(
            provider=self.provider,
            registry=registry,
            settings=settings,
            system=system_prompt(),
            cite=cite,
            session_id=case.id,
            ctx=ctx,
            gate=gate,
            hooks=HookChain([audit]),
            conversation_id=case.id,
            store=store,
        )
        loop.interactive = channel.ask

        started = time.monotonic()
        try:
            for index, turn in enumerate(case.turns):
                channel.reset(turn.interactive, turn.interactive_answer)
                sink.events = []
                turn_started = time.monotonic()
                outcome = await asyncio.wait_for(
                    loop.run(turn.user), self.turn_timeout_s
                )
                captured = CapturedTurn(
                    index=index,
                    user=turn.user,
                    outcome=outcome,
                    events=list(sink.events),
                    interactions=list(channel.log),
                    duration_ms=round((time.monotonic() - turn_started) * 1000),
                )
                record.turns.append(captured)
                self._log(
                    f"  [{case.id}] turn {index + 1}: succeeded={outcome.succeeded} "
                    f"tools={outcome.tool_calls} tokens="
                    f"{outcome.usage.input_tokens + outcome.usage.output_tokens}"
                )
        except Exception as exc:  # noqa: BLE001 - 失败的用例是数据，而非崩溃
            record.error = f"{type(exc).__name__}: {exc}"
            self._log(f"  [{case.id}] error: {record.error}")
        finally:
            record.duration_ms = round((time.monotonic() - started) * 1000)
            record.exports = _collect_exports(settings.paths.output_dir)
            record.report_text = self._read_reports(record.exports)
            record.audit_denials = _read_audit_denials(settings.audit.log_path)
        return record

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
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
