"""Permission gate: decide allow / confirm / deny for one tool call (docs 03.7.1).

Rule priority is deny > mode fallback. Read tools are always allowed; write
tools confirm under DEFAULT/PLAN and pass under AUTO. Writes whose ``path``
target resolves inside the output or cache directory count as artefact writes
and are allowed outright.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from finharness.permissions.modes import PermissionMode, Verdict
from finharness.permissions.rules import scan_args, scan_text
from finharness.tools.base import PermissionLevel

# Path-bearing tools: a write whose target sits in these directories is an
# artefact write, not a mutation of user data.
_WHITELIST_DIR_KEYS = ("output", "cache")
_PATH_ARG_KEYS = ("path", "file", "filename", "target")


@dataclass(frozen=True, slots=True)
class GateDecision:
    verdict: Verdict
    reason: str = ""


class ReadOnlyGate:
    """Default gate: permits read tools, refuses everything else.

    This preserves the pre-governance behaviour for callers that do not inject a
    real gate, and keeps the refusal message the loop already produced.
    """

    async def check(self, tool, args: dict) -> GateDecision:  # noqa: ARG002 - interface parity
        if tool.permission is PermissionLevel.READ:
            return GateDecision(Verdict.ALLOW)
        return GateDecision(Verdict.DENY, f"tool is not read-only: {tool.name}")


class PermissionGate:
    """Mode-aware gate with deny rules and a path whitelist."""

    def __init__(
        self,
        *,
        settings,
        confirm: "Callable[[str, dict], Awaitable[bool]] | None" = None,
        deny_patterns: tuple[str, ...] | None = None,
        mode: PermissionMode | None = None,
    ) -> None:
        self.settings = settings
        self.mode = mode or PermissionMode(settings.permission.default_mode)
        self.confirm = confirm
        self._output_dir = Path(settings.paths.output_dir).resolve()
        self._cache_dir = Path(settings.data.cache_dir).resolve()
        self._deny_patterns = deny_patterns

    async def check(self, tool, args: dict) -> GateDecision:
        """Decide the verdict. CONFIRM is answered here when a callback exists."""
        # 1. Trading-intent rules win over everything, including read tools:
        #    a data parameter carrying an order instruction is still an attack.
        hit = self._scan_deny(tool, args)
        if hit is not None:
            return GateDecision(Verdict.DENY, hit.reason())

        # 2. Read tools are always allowed.
        if tool.permission is PermissionLevel.READ:
            return GateDecision(Verdict.ALLOW)

        # 3. Artefact writes inside the whitelisted directories bypass confirmation.
        if self._path_in_whitelist(args):
            return GateDecision(Verdict.ALLOW, "写入白名单目录（output/data_cache）")

        # 4. Mode fallback for write tools.
        if self.mode is PermissionMode.AUTO:
            return GateDecision(Verdict.ALLOW, "auto 模式放行写类工具")

        if self.confirm is None:
            # Non-interactive caller: refuse rather than silently allowing.
            return GateDecision(Verdict.DENY, f"写类工具需确认，但当前无交互通道：{tool.name}")
        approved = await self.confirm(tool.name, args)
        if approved:
            return GateDecision(Verdict.ALLOW, "用户已确认")
        return GateDecision(Verdict.DENY, "用户已拒绝该工具调用")

    # -- helpers --------------------------------------------------------------
    def _scan_deny(self, tool, args: dict):
        code = args.get("code") if isinstance(args.get("code"), str) else None
        if code is not None:
            hit = scan_text(code, where="code")
            if hit is not None:
                return hit
        return scan_args(args)

    def _path_in_whitelist(self, args: dict) -> bool:
        for key in _PATH_ARG_KEYS:
            raw = args.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                target = Path(raw).resolve()
            except (OSError, ValueError):
                continue
            if self._under(target, self._output_dir) or self._under(target, self._cache_dir):
                return True
        return False

    @staticmethod
    def _under(target: Path, root: Path) -> bool:
        """Containment check that resists ``..`` escapes (both already resolved)."""
        try:
            return target == root or target.is_relative_to(root)
        except (OSError, ValueError):
            return False
