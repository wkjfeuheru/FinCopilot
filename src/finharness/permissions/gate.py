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
        # 可达性来自 workspace（唯一来源），与工具侧同源判定：门与工具因此
        # 不可能对"这是不是产物写入"得出不同结论。此前两者各自拼根，
        # 白名单根一度宽于工具实际根。
        self.workspace = Workspace(settings)
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
        #    **仅限写入**：缓存的两棵子树（parquet/、pdf/）对读取是开放的，
        #    读类工具本就以它们为工作对象——研报正文落盘后由 read_pdf 精读、
        #    summarize_document 摘要，复核子代理也据此重读载荷。此处若不分
        #    读写一律拒绝，这条设计内的主流程会整条不可用，也与 workspace
        #    的可读契约（output/ + cache 的 parquet/pdf）相矛盾。读类调用
        #    的可达性由工具侧 resolve_read 把关，它只放行那两棵子树。
        if tool.permission is not PermissionLevel.READ and self._path_in_cache(args):
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
            if self.workspace.is_in_cache(args.get(key)):
                return True
        return False

    def _path_in_whitelist(self, args: dict) -> bool:
        for key in _PATH_ARG_KEYS:
            if self.workspace.is_artifact_write(args.get(key)):
                return True
        return False
