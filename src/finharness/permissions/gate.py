"""权限闸门：为一次工具调用决定 allow / confirm / deny（docs 03.7.1）。

规则优先级为 deny > 模式回退。读类工具始终放行；写类工具在 DEFAULT/PLAN 下需
确认，在 AUTO 下直接通过。``path`` 目标解析后位于 output 或 cache 目录内的写入
被视为产物（artefact）写入，直接放行。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from finharness.permissions.modes import PermissionMode, Verdict
from finharness.permissions.rules import scan_args, scan_text
from finharness.tools.base import PermissionLevel

# 带路径的工具：目标位于这些目录中的写入属于产物写入，而非对用户数据的改动。
_WHITELIST_DIR_KEYS = ("output", "cache")
_PATH_ARG_KEYS = ("path", "file", "filename", "target")


@dataclass(frozen=True, slots=True)
class GateDecision:
    verdict: Verdict
    reason: str = ""


class ReadOnlyGate:
    """默认闸门：放行读类工具，拒绝其他一切。

    这为未注入真实闸门的调用方保留了治理之前的行为，并沿用循环已经产生的
    拒绝消息。
    """

    async def check(self, tool, args: dict) -> GateDecision:  # noqa: ARG002 - 保持接口一致
        if tool.permission is PermissionLevel.READ:
            return GateDecision(Verdict.ALLOW)
        return GateDecision(Verdict.DENY, f"tool is not read-only: {tool.name}")


class PermissionGate:
    """感知模式、带拒绝规则与路径白名单的闸门。"""

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
        """决定裁决。当存在回调时，CONFIRM 在此处被应答。"""
        # 1. 交易意图规则优先于一切，包括读类工具：
        #    一个携带下单指令的数据参数仍然是一次攻击。
        hit = self._scan_deny(tool, args)
        if hit is not None:
            return GateDecision(Verdict.DENY, hit.reason())

        # 2. 读类工具始终放行。
        if tool.permission is PermissionLevel.READ:
            return GateDecision(Verdict.ALLOW)

        # 3. 白名单目录内的产物写入绕过确认。
        if self._path_in_whitelist(args):
            return GateDecision(Verdict.ALLOW, "写入白名单目录（output/data_cache）")

        # 4. 写类工具的模式回退。
        if self.mode is PermissionMode.AUTO:
            return GateDecision(Verdict.ALLOW, "auto 模式放行写类工具")

        if self.confirm is None:
            # 非交互式调用方：拒绝，而不是静默放行。
            return GateDecision(Verdict.DENY, f"写类工具需确认，但当前无交互通道：{tool.name}")
        approved = await self.confirm(tool.name, args)
        if approved:
            return GateDecision(Verdict.ALLOW, "用户已确认")
        return GateDecision(Verdict.DENY, "用户已拒绝该工具调用")

    # -- 辅助方法 --------------------------------------------------------------
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
        """抵抗 ``..`` 越权逃逸的包含性检查（两者均已解析）。"""
        try:
            return target == root or target.is_relative_to(root)
        except (OSError, ValueError):
            return False
