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

决策与交互分离：``decide()`` 同步返回是否需要确认；``resolve()`` 把用户答案
映射为 allow/deny 并更新免问集合。``check()`` 仍是兼容适配器，供仍注入
confirm 回调的直接调用方使用。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from finharness.hooks.audit import summarize_args
from finharness.permissions.modes import PermissionMode, Verdict
from finharness.permissions.rules import RuleHit, scan_args, scan_text
from finharness.tools.base import PermissionLevel
from finharness.workspace import Workspace

# 带路径的工具：目标位于可写区内的写入属于产物写入，而非对用户数据的改动。
# cache 不在其中——写缓存在 check 中被显式拒绝，而非默默绕过确认。
_PATH_ARG_KEYS = ("path", "file", "filename", "target")

# 网络外发确认类别：同一对话内首次确认后免问。键是"风险类别"而非工具名，
# 使 web_search 与研报全文抓取共享一次授权——它们把不可信第三方文本拉入
# 上下文的性质相同，分开问只会打断用户两次。
EGRESS_CATEGORY = "egress"
WRITE_CATEGORY = "write"

_APPROVE_ONCE = frozenset({"y", "y_remember", "y_session"})


def _egress_requested(tool, args: dict) -> bool:
    """判定一次读类调用是否构成网络外发。

    读的是**声明**（``@tool(egress=True)``），因为这正是当初注释里预告的时点：判定
    一度是闸门内的名称表，只在"恰好两个工具"时成立。同花顺把整组数据工具都变成
    外发工具后，逐名字维护必然漏改，而漏改的后果是一个本该询问用户的外发调用静默
    放行——所以规则搬到了它本来该在的地方。

    ``get_research_reports`` 是唯一按**参数**而非按工具判定的情形（``with_text``），
    因此这里保留一个显式特例：它关掉全文时只是一次普通的读书调用，把它整类标成外发
    会让元数据查询也弹框。
    """
    if tool.name == "get_research_reports":
        return args.get("with_text") is True
    return bool(getattr(tool, "egress", False))


@dataclass(frozen=True, slots=True)
class ConfirmationSpec:
    kind: str
    prompt: str
    options: tuple[str, ...]
    category: str
    call_ids: tuple[str, ...] = ()
    multi_select: bool = False


@dataclass(frozen=True, slots=True)
class GateDecision:
    verdict: Verdict
    reason: str = ""
    confirmation: ConfirmationSpec | None = None


class InteractivePort(Protocol):
    async def prompt(self, spec: ConfirmationSpec) -> str | None: ...


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

    def decide(self, tool, args: dict) -> GateDecision:  # noqa: ARG002 - 保持接口一致
        hit = _scan_tool(tool, args, self._deny_patterns)
        if hit is not None:
            return GateDecision(Verdict.DENY, hit.reason())
        if tool.permission is PermissionLevel.READ:
            return GateDecision(Verdict.ALLOW)
        return GateDecision(Verdict.DENY, f"tool is not read-only: {tool.name}")

    async def check(self, tool, args: dict) -> GateDecision:
        return self.decide(tool, args)


class PermissionGate:
    """感知模式、带拒绝规则与路径白名单的闸门。

    ``confirmed_categories`` 是对话级"已确认类别"集合：网络外发在同一
    对话首次确认后免问。由调用方传入并共享（同一对话的多个 gate 实例
    看到同一份），进程内存活；对话结束即随 registry 消亡，重启后重新
    询问——免问授权的生存期不应长于它所授权的那个对话。

    ``session_approved`` 是会话级写类免问集合（``y_session``）：与对话级
    ``confirmed_categories`` 并列，生命周期同样由调用方持有。
    """

    def __init__(
        self,
        *,
        settings,
        confirm: Callable[[str, dict], Awaitable[bool]] | None = None,
        deny_patterns: tuple[str, ...] | None = None,
        mode: PermissionMode | None = None,
        conversation_id: str = "local",
        confirmed_categories: set[str] | None = None,
        session_approved: set[str] | None = None,
        confirm_egress: Callable[[str, dict], Awaitable[bool]] | None = None,
    ) -> None:
        self.settings = settings
        # ``mode`` 规范化后再存：下方对 AUTO 的判定用 ``is`` 比较枚举成员，而签名虽然
        # 声明 ``PermissionMode``，Python 不会替我们挡住一个等价的字符串。此前落库的
        # ``"auto"`` 会让两个 AUTO 分支永不命中——AUTO 模式静默退化成"需确认"，无交互
        # 通道时外发工具被直接拒绝。``PermissionMode`` 是 ``str`` 枚举，转换是幂等的，
        # 因此对已是成员的入参没有额外代价。
        self.mode = (
            PermissionMode(mode)
            if mode is not None
            else PermissionMode(settings.permission.default_mode)
        )
        self.confirm = confirm
        # 网络外发确认可以携带"本对话不再询问"的第三选项；缺省回落到与写
        # 工具相同的两选项回调。
        self.confirm_egress = confirm_egress or confirm
        self.conversation_id = conversation_id
        self.confirmed_categories = (
            confirmed_categories if confirmed_categories is not None else set()
        )
        self.session_approved = session_approved if session_approved is not None else set()
        # 可达性来自 workspace（唯一来源），与工具侧同源判定：门与工具因此
        # 不可能对"这是不是产物写入"得出不同结论。此前两者各自拼根，
        # 白名单根一度宽于工具实际根。
        self.workspace = Workspace(settings)
        self._deny_patterns = deny_patterns

    def decide(self, tool, args: dict) -> GateDecision:
        """同步裁决：需要确认时返回 ``CONFIRM`` + ``ConfirmationSpec``，不 await。"""
        hit = self._scan_deny(tool, args)
        if hit is not None:
            return GateDecision(Verdict.DENY, hit.reason())

        if tool.permission is not PermissionLevel.READ and self._path_in_cache(args):
            return GateDecision(Verdict.DENY, "缓存目录不可写：data_cache 载荷为共享只读")

        if tool.permission is PermissionLevel.READ:
            if not _egress_requested(tool, args):
                return GateDecision(Verdict.ALLOW)
            return self._decide_egress(tool, args)

        if self._path_in_whitelist(args):
            return GateDecision(Verdict.ALLOW, "写入白名单目录（output）")

        if self.mode is PermissionMode.AUTO:
            return GateDecision(Verdict.ALLOW, "auto 模式放行写类工具")

        if WRITE_CATEGORY in self.session_approved:
            return GateDecision(Verdict.ALLOW, "本会话已确认写类工具")

        prompt = f"工具 {tool.name} 将执行，入参：{summarize_args(args)}"
        return GateDecision(
            Verdict.CONFIRM,
            "写类工具需确认",
            confirmation=ConfirmationSpec(
                kind="permission",
                prompt=prompt,
                options=("y", "y_session", "n"),
                category=WRITE_CATEGORY,
            ),
        )

    def resolve(self, spec: ConfirmationSpec, answer: str | None) -> GateDecision:
        """把用户答案映射为 allow/deny，并更新免问集合。"""
        if answer is None or answer == "n" or answer not in _APPROVE_ONCE:
            return GateDecision(Verdict.DENY, "用户已拒绝该工具调用")
        if answer == "y_remember":
            self.confirmed_categories.add(spec.category)
        elif answer == "y_session":
            self.session_approved.add(spec.category)
        if spec.category == EGRESS_CATEGORY:
            return GateDecision(Verdict.ALLOW, "用户已确认网络访问")
        return GateDecision(Verdict.ALLOW, "用户已确认")

    async def check(self, tool, args: dict) -> GateDecision:
        """兼容适配器：``decide`` 后若需确认则 await 回调，再 ``resolve``。"""
        decision = self.decide(tool, args)
        if decision.verdict is not Verdict.CONFIRM:
            return decision
        spec = decision.confirmation
        if spec is None:
            return GateDecision(Verdict.DENY, f"写类工具需确认，但当前无交互通道：{tool.name}")

        callback = (
            self.confirm_egress if spec.category == EGRESS_CATEGORY else self.confirm
        )
        if callback is None:
            if spec.category == EGRESS_CATEGORY:
                return GateDecision(
                    Verdict.DENY, f"网络访问需确认，但当前无交互通道：{tool.name}"
                )
            return GateDecision(
                Verdict.DENY, f"写类工具需确认，但当前无交互通道：{tool.name}"
            )
        approved = await callback(tool.name, args)
        return self.resolve(spec, "y" if approved else "n")

    def _decide_egress(self, tool, args: dict) -> GateDecision:
        if EGRESS_CATEGORY in self.confirmed_categories:
            return GateDecision(Verdict.ALLOW, "本对话已确认网络访问")

        if self.mode is PermissionMode.AUTO:
            return GateDecision(Verdict.ALLOW, "auto 模式放行网络访问")

        prompt = (
            f"工具 {tool.name} 将访问外部网络并引入第三方内容，"
            f"入参：{summarize_args(args)}"
        )
        return GateDecision(
            Verdict.CONFIRM,
            "网络访问需确认",
            confirmation=ConfirmationSpec(
                kind="permission",
                prompt=prompt,
                options=("y", "y_remember", "n"),
                category=EGRESS_CATEGORY,
            ),
        )

    # -- 辅助方法 --------------------------------------------------------------
    def _scan_deny(self, tool, args: dict):
        return _scan_tool(tool, args, self._deny_patterns)

    def _path_in_cache(self, args: dict) -> bool:
        for key in _PATH_ARG_KEYS:
            if self.workspace.is_in_cache(args.get(key)):
                return True
        return False

    def _path_in_whitelist(self, args: dict) -> bool:
        for key in _PATH_ARG_KEYS:
            if self.workspace.is_artifact_write(args.get(key)):
                return True
        return False
