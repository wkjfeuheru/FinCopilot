"""权限闸门：为一次工具调用决定 allow / confirm / deny（docs 03.7.1）。

规则优先级为 deny > 缓存拒写 > 模式回退。读类工具默认放行，但网络外发
（联网检索、逐篇抓取研报全文）首次须经确认，之后在本对话内免问；写类
工具在 DEFAULT/PLAN 下需确认，在 AUTO 下直接通过。``path`` 目标解析后
位于 output 目录内的写入被视为产物（artefact）写入，直接放行；写入
data_cache 则被结构性拒绝——缓存的 lookup 键跨用户共享，模型侧的一次
写入就能替换他人稍后命中的载荷。

**deny 规则先于读/写分流，且两个闸门都执行它。** 只读闸门只描述"这个工具不会
改状态"，不描述"它的入参可信"：一段携带下单指令的文本参数在只读工具上同样是
攻击，因此 ``ReadOnlyGate`` 也必须扫描（docs 03.7.2）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from finharness.permissions.modes import PermissionMode, Verdict
from finharness.permissions.rules import RuleHit, scan_args, scan_text
from finharness.tools.base import PermissionLevel

# 带路径的工具：目标位于这些目录中的写入属于产物写入，而非对用户数据的改动。
# cache 不在其中——写缓存在 check 中被显式拒绝，而非默默绕过确认。
_WHITELIST_DIR_KEYS = ("output",)
_PATH_ARG_KEYS = ("path", "file", "filename", "target")

# 网络外发确认类别：同一对话内首次确认后免问。键是"风险类别"而非工具名，
# 使 web_search 与研报全文抓取共享一次授权——它们把不可信第三方文本拉入
# 上下文的性质相同，分开问只会打断用户两次。
EGRESS_CATEGORY = "egress"


def _egress_requested(tool, args: dict) -> bool:
    """判定一次读类调用是否构成网络外发。

    表驱动而非给 @tool 加声明属性：目前只有两个工具、且其中一个按参数
    （``with_text``）而非按工具判定，声明轴上还没有第二个消费者；等出现
    "整类工具都外发"的工具组时再提升为声明属性。
    """
    if tool.name == "web_search":
        return True
    if tool.name == "get_research_reports" and args.get("with_text") is True:
        return True
    return False


@dataclass(frozen=True, slots=True)
class GateDecision:
    verdict: Verdict
    reason: str = ""


def _scan_tool(tool, args: dict, patterns: tuple[str, ...] | None) -> RuleHit | None:
    """对一次工具调用的参数快照施加拒绝规则。

    ``code`` 单独扫一次并标为 ``where="code"``，使拒绝理由能指出命中位置；
    其余参数走递归扫描，因此嵌套结构不会成为藏身处。
    """
    code = args.get("code")
    if isinstance(code, str):
        hit = scan_text(code, where="code", patterns=patterns)
        if hit is not None:
            return hit
    return scan_args(args, patterns=patterns)


class ReadOnlyGate:
    """默认闸门：放行读类工具，拒绝其他一切。

    这为未注入真实闸门的调用方保留了治理之前的行为，并沿用循环已经产生的
    拒绝消息。拒绝规则仍然生效——只读不等于入参可信。
    """

    def __init__(self, deny_patterns: tuple[str, ...] | None = None) -> None:
        self._deny_patterns = deny_patterns

    async def check(self, tool, args: dict) -> GateDecision:  # noqa: ARG002 - 保持接口一致
        hit = _scan_tool(tool, args, self._deny_patterns)
        if hit is not None:
            return GateDecision(Verdict.DENY, hit.reason())
        if tool.permission is PermissionLevel.READ:
            return GateDecision(Verdict.ALLOW)
        return GateDecision(Verdict.DENY, f"tool is not read-only: {tool.name}")


class PermissionGate:
    """感知模式、带拒绝规则与路径白名单的闸门。

    ``confirmed_categories`` 是对话级"已确认类别"集合：网络外发在同一
    对话首次确认后免问。由调用方传入并共享（同一对话的多个 gate 实例
    看到同一份），进程内存活；对话结束即随 registry 消亡，重启后重新
    询问——免问授权的生存期不应长于它所授权的那个对话。
    """

    def __init__(
        self,
        *,
        settings,
        confirm: "Callable[[str, dict], Awaitable[bool]] | None" = None,
        deny_patterns: tuple[str, ...] | None = None,
        mode: PermissionMode | None = None,
        conversation_id: str = "local",
        confirmed_categories: set[str] | None = None,
        confirm_egress: "Callable[[str, dict], Awaitable[bool]] | None" = None,
    ) -> None:
        self.settings = settings
        self.mode = mode or PermissionMode(settings.permission.default_mode)
        self.confirm = confirm
        # 网络外发确认可以携带"本对话不再询问"的第三选项；缺省回落到与写
        # 工具相同的两选项回调。
        self.confirm_egress = confirm_egress or confirm
        self.conversation_id = conversation_id
        self.confirmed_categories = (
            confirmed_categories if confirmed_categories is not None else set()
        )
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

        # 2. 写入共享缓存结构性拒绝：lookup 键跨用户共享，模型侧写入
        #    即缓存投毒。即使 AUTO 模式、即使参数带路径白名单键。
        if self._path_in_cache(args):
            return GateDecision(Verdict.DENY, "缓存目录不可写：data_cache 载荷为共享只读")

        # 3. 读类工具：默认放行；网络外发首次须确认。
        if tool.permission is PermissionLevel.READ:
            if not _egress_requested(tool, args):
                return GateDecision(Verdict.ALLOW)
            return await self._check_egress(tool, args)

        # 4. 白名单目录内的产物写入绕过确认。
        if self._path_in_whitelist(args):
            return GateDecision(Verdict.ALLOW, "写入白名单目录（output）")

        # 5. 写类工具的模式回退。
        if self.mode is PermissionMode.AUTO:
            return GateDecision(Verdict.ALLOW, "auto 模式放行写类工具")

        if self.confirm is None:
            # 非交互式调用方：拒绝，而不是静默放行。
            return GateDecision(Verdict.DENY, f"写类工具需确认，但当前无交互通道：{tool.name}")
        approved = await self.confirm(tool.name, args)
        if approved:
            return GateDecision(Verdict.ALLOW, "用户已确认")
        return GateDecision(Verdict.DENY, "用户已拒绝该工具调用")

    async def _check_egress(self, tool, args: dict) -> GateDecision:
        """网络外发的确认链；与写工具共用模式语义但独立于它计免问。

        是否记入"本对话不再询问"由 ``confirm_egress`` 的实现决定（它是
        唯一知道用户选了"允许一次"还是"允许并记住"的一方）；本方法只
        消费 ``confirmed_categories`` 与放行结果。
        """
        if EGRESS_CATEGORY in self.confirmed_categories:
            return GateDecision(Verdict.ALLOW, "本对话已确认网络访问")

        if self.mode is PermissionMode.AUTO:
            return GateDecision(Verdict.ALLOW, "auto 模式放行网络访问")

        if self.confirm_egress is None:
            return GateDecision(Verdict.DENY, f"网络访问需确认，但当前无交互通道：{tool.name}")
        approved = await self.confirm_egress(tool.name, args)
        if approved:
            return GateDecision(Verdict.ALLOW, "用户已确认网络访问")
        return GateDecision(Verdict.DENY, "用户已拒绝该网络访问")

    # -- 辅助方法 --------------------------------------------------------------
    def _scan_deny(self, tool, args: dict):
        return _scan_tool(tool, args, self._deny_patterns)

    def _path_in_cache(self, args: dict) -> bool:
        for key in _PATH_ARG_KEYS:
            raw = args.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                target = Path(raw).resolve()
            except (OSError, ValueError):
                continue
            if self._under(target, self._cache_dir):
                return True
        return False

    def _path_in_whitelist(self, args: dict) -> bool:
        for key in _PATH_ARG_KEYS:
            raw = args.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                target = Path(raw).resolve()
            except (OSError, ValueError):
                continue
            if self._under(target, self._output_dir):
                return True
        return False

    @staticmethod
    def _under(target: Path, root: Path) -> bool:
        """抵抗 ``..`` 越权逃逸的包含性检查（两者均已解析）。"""
        try:
            return target == root or target.is_relative_to(root)
        except (OSError, ValueError):
            return False
