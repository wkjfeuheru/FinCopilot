"""两层工具注册表：常驻 schema 加上可检索的懒加载工具。

“哪些工具常驻、哪些按需”不再由本模块维护：它是 ``@tool(tier=...)`` 声明的一部分
（docs 03.4.3）。此前那份 ``DEFAULT_LAZY_TOOLS`` 必须与工具清单手工同步，漏改会让一个
工具"列得出来却调不到"；现在层级随声明走，``settings.tools`` 的 ``resident``/``lazy``
降级为运维覆盖——不配置时完全听声明的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from finharness.context.tokens import CHARS_PER_TOKEN
from finharness.context.trim import DEFAULT_MAX_DESC_LEN, trim_schema
from finharness.data.access import DataAccess
from finharness.tools.base import BaseTool
from finharness.tools.declare import PermissionLevel, Tier, ToolGroup

# 工具类经由 loader 注册，使目录扩张时该模块仍可导入。
from finharness.tools.fin.announcements import GetAnnouncementsTool
from finharness.tools.fin.backtest import RunBacktestTool
from finharness.tools.fin.chart import MakeChartTool
from finharness.tools.fin.dataset import (
    ListFuyaoDatasetsTool,
    QueryAShareDataTool,
    QueryFundDataTool,
    QueryFuturesDataTool,
    QueryOptionsDataTool,
)
from finharness.tools.fin.financials import GetFinancialsTool
from finharness.tools.fin.indicators import GetIndicatorsTool
from finharness.tools.fin.industry import GetIndustryConstituentsTool, GetIndustryPerfTool
from finharness.tools.fin.kline import GetKlineTool
from finharness.tools.fin.macro import GetMacroIndicatorsTool
from finharness.tools.fin.metrics import CalcMetricsTool
from finharness.tools.fin.news import GetMarketNewsTool
from finharness.tools.fin.peers import GetPeersTool
from finharness.tools.fin.quote import GetQuoteTool
from finharness.tools.fin.research_reports import GetResearchReportsTool
from finharness.tools.fin.valuation import GetValuationTool
from finharness.tools.fin.valuation_calc import CalcValuationTool
from finharness.tools.fin.writer import WriteReportTool
from finharness.tools.generic.files import ReadFileTool, WriteFileTool
from finharness.tools.generic.pdf import ReadPdfTool
from finharness.tools.generic.web import WebSearchTool
from finharness.tools.meta.ask import AskUserTool
from finharness.tools.meta.discovery import SearchToolsTool
from finharness.tools.meta.memory import (
    ForgetMemoryTool,
    SearchMemoryTool,
    UpdateMemoryTool,
)
from finharness.tools.meta.plan import (
    RecordConclusionTool,
    ResearchPlanTool,
    UpdatePlanStepTool,
)
from finharness.tools.meta.preference import RememberPreferenceTool
from finharness.tools.meta.spawn import SpawnAgentTool
from finharness.tools.meta.summarize import SummarizeDocumentTool

ALL_TOOL_CLASSES: tuple[type[BaseTool], ...] = (
    # 金融-数据
    GetQuoteTool,
    GetKlineTool,
    GetIndicatorsTool,
    GetFinancialsTool,
    GetValuationTool,
    GetPeersTool,
    GetMarketNewsTool,
    GetAnnouncementsTool,
    GetResearchReportsTool,
    GetMacroIndicatorsTool,
    GetIndustryPerfTool,
    GetIndustryConstituentsTool,
    # 金融-数据（同花顺长尾数据集派发器）
    ListFuyaoDatasetsTool,
    QueryAShareDataTool,
    QueryFundDataTool,
    QueryFuturesDataTool,
    QueryOptionsDataTool,
    # 金融-计算
    CalcMetricsTool,
    CalcValuationTool,
    RunBacktestTool,
    # 金融-输出
    MakeChartTool,
    WriteReportTool,
    # 通用
    ReadFileTool,
    ReadPdfTool,
    WriteFileTool,
    WebSearchTool,
    # 元
    ResearchPlanTool,
    UpdatePlanStepTool,
    RecordConclusionTool,
    SearchToolsTool,
    SpawnAgentTool,
    SummarizeDocumentTool,
    AskUserTool,
    RememberPreferenceTool,
    SearchMemoryTool,
    UpdateMemoryTool,
    ForgetMemoryTool,
)

FINANCIAL_DATA_TOOLS = ALL_TOOL_CLASSES

# 其工具可交给只读复核子代理的分组。数据工具让它能重新取数并核对报告中的数字；
# GENERIC 增加 read_file 以读取渲染产物与缓存 parquet。FIN_CALC 被刻意排除：
# 这些工具只在调用方提供的输入上做计算，因此无法增加独立核验某个数字的能力。
REVIEW_TOOL_GROUPS: tuple[ToolGroup, ...] = (ToolGroup.FIN_DATA, ToolGroup.GENERIC)


def review_tool_names() -> tuple[str, ...]:
    """复核子代理被允许调用的只读子集。

    由契约推导而来而非手工维护的清单，因此新的只读数据工具会自动对复核者可用。
    规则隐含了四项排除，且每一项都有其意义：

    * ``PermissionLevel.WRITE`` —— 复核者绝不能有写权限。
    * META 分组 —— ``research_plan`` 与 ``remember_preference`` 会对会话/全局状态
      产生副作用。
    * ``needs_interactive`` —— ``ask_user`` 在子代理中无法工作，因为没有接上交互通道。
    * ``review_eligible=False`` —— 联网工具。一旦复核者开始搜索互联网，就会消耗 token
      把不可信文本拉入一个只负责拿报告与会话自身数据核对的上下文。
    """
    return tuple(
        tool_cls.name
        for tool_cls in ALL_TOOL_CLASSES
        if tool_cls.permission is PermissionLevel.READ
        and tool_cls.group in REVIEW_TOOL_GROUPS
        and not tool_cls.needs_interactive
        and tool_cls.review_eligible
    )


def worker_tool_names() -> tuple[str, ...]:
    """通用子代理可调用的子集：仅限本地材料（docs 03.10）。

    为隔离上下文而派发的子代理只消费交给它的材料；它不会自行去取数。因此这里刻意
    只含*通用*只读工具——``read_file``，用于以路径传入的材料——而**不含**
    ``review_tool_names()``，后者还额外携带数据层。一个唯一输入只是任务文本的
    worker 本就无法正确选择股票代码或行业，给它数据工具只会招致猜测，而非隔离。

    与 ``review_tool_names`` 一样按谓词推导：新的通用只读工具无需修改本函数即可可用。
    """
    return tuple(
        tool_cls.name
        for tool_cls in ALL_TOOL_CLASSES
        if tool_cls.permission is PermissionLevel.READ
        and tool_cls.group is ToolGroup.GENERIC
        and not tool_cls.needs_interactive
        and tool_cls.review_eligible
    )


@dataclass(frozen=True, slots=True)
class ToolBrief:
    """检索结果：足以让模型判断是否要用、以及怎么调用它。

    ``params`` 是命中工具的**参数清单**，不再是完整 schema。带它是为了去掉一次往返：
    懒加载工具的参数说明过去只能靠激活后下一轮的 schema 才可见，于是"检索到"与"能用"
    之间隔着一次模型调用。检索结果里直接给出参数，模型一次就能正确发起调用。
    """

    name: str
    description: str
    group: str
    tier: str = Tier.RESIDENT.value
    active: bool = True
    score: int = 0
    inputs: tuple[str, ...] = ()
    required: frozenset[str] = field(default_factory=frozenset)
    output_note: str = ""


def build_parameters(model: type[Any]) -> dict[str, Any]:
    """模型的 JSON Schema，已剥除 title 噪声字段。"""
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


class ToolRegistry:
    """实例化所有工具，但默认只暴露常驻 schema。

    ``only`` 收窄的是目录本身，而不只是其可见性：集合之外的名称无法解析，因此循环会
    将其报告为未知。子代理正是以这种方式被限域——该限制是结构性的，而不是一条子代理
    可能被说服去忽略的策略。
    """

    def __init__(
        self,
        data: DataAccess,
        *,
        ctx: Any | None = None,
        settings: Any | None = None,
        only: set[str] | None = None,
    ) -> None:
        self.settings = settings
        # 保留它，使循环无需调用方两次传入同一个 DataAccess，即可铸造一个协调器
        # （或为协调器铸造一个限域注册表）。
        self.data = data
        classes = (
            tuple(tool_cls for tool_cls in ALL_TOOL_CLASSES if tool_cls.name in only)
            if only is not None
            else ALL_TOOL_CLASSES
        )
        self.tools: dict[str, BaseTool] = {
            tool_cls.name: tool_cls(data, ctx=ctx) for tool_cls in classes
        }
        # 元工具需要它们所栖身的目录；构造后接线。
        for tool in self.tools.values():
            tool.registry = self
        configured_lazy = tuple(getattr(settings.tools, "lazy", ()) or ()) if settings else ()
        configured_resident = tuple(getattr(settings.tools, "resident", ()) or ()) if settings else ()
        if only is not None:
            # 限域目录是一个工作集，因此其中没有任何工具是懒加载的：它是一份被交给
            # 子代理的固定清单，按需激活在那里只会使目录随运行而漂移。
            self._lazy: set[str] = set()
        elif configured_resident:
            # 显式的常驻列表优先；其余一切都变为懒加载。
            self._lazy = {name for name in self.tools if name not in configured_resident}
        elif configured_lazy:
            self._lazy = {name for name in configured_lazy if name in self.tools}
        else:
            # 默认口径来自声明：``@tool(tier=Tier.LAZY)`` 的工具按需注入。
            self._lazy = {
                name
                for name, tool in self.tools.items()
                if getattr(type(tool), "tier", None) is Tier.LAZY
            }
        # 懒加载工具只有在激活后才进入请求（docs 03.4.3）。
        self._active: set[str] = {name for name in self.tools if name not in self._lazy}
        # 激活顺序：用于在超限时按最久未用降级（``_max_active``）。
        self._order: list[str] = list(self._active)

    # -- 目录 ------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self.tools)

    def resident_names(self) -> list[str]:
        return [name for name in self.tools if name not in self._lazy]

    def lazy_names(self) -> list[str]:
        return [name for name in self.tools if name in self._lazy]

    def resolve(self, name: str) -> BaseTool | None:
        return self.tools.get(name)

    def is_read_only(self, name: str) -> bool:
        tool = self.tools.get(name)
        return tool is not None and tool.permission is PermissionLevel.READ

    def is_active(self, name: str) -> bool:
        return name in self._active

    def is_lazy(self, name: str) -> bool:
        return name in self._lazy

    def activate(self, name: str) -> bool:
        """将一个工具加入已注入集合；对未知或已激活的工具返回 False。"""
        if name not in self.tools or name in self._active:
            return False
        self._active.add(name)
        self._order = [existing for existing in self._order if existing != name]
        self._order.append(name)
        self._demote_beyond_cap()
        return True

    def activate_many(self, names: list[str]) -> list[str]:
        """批量激活，返回真正新激活的名称（已激活或未知的项被忽略）。"""
        return [name for name in names if self.activate(name)]

    def _demote_beyond_cap(self) -> None:
        """常驻集合超限时，把最久未激活的懒加载工具退回按需状态。

        ``tools`` 数组属于 provider 的缓存前缀，而激活会让它变化。此前激活是稀有的一次性
        事件，接受其一次失效；改为"检索即激活"后，长会话里的常驻集合会单调增长，于是这里
        设一个上限：超出部分按最久未用退场，让上限成为可推理的常量而不是会话长度的函数。
        """
        cap = int(getattr(getattr(self.settings, "tools", None), "max_active", 0) or 0)
        if cap <= 0 or len(self._active) <= cap:
            return
        resident = {name for name in self.tools if name not in self._lazy}
        # 常驻工具永不退场；退场候选按激活顺序从最旧的开始。
        for name in list(self._order):
            if len(self._active) <= cap:
                break
            if name in resident or name not in self._lazy:
                continue
            if name in self._active:
                self._active.discard(name)
                self._order = [existing for existing in self._order if existing != name]

    # -- schema --------------------------------------------------------------
    def schemas(self, names: set[str] | None = None) -> list[dict]:
        """已注入集合的 OpenAI 风格 schema（按注册表顺序）。

        ``names`` 默认为活动集合，因此懒加载工具会一直隐藏，直到被发现层检索命中或
        被路由层推断需要。描述会按配置的预算裁剪，因为每个常驻 schema 都会在每次请求时
        重新发送（docs 3.6.1）。
        """
        selected = self._active if names is None else names
        budget = self._desc_budget()
        return [
            trim_schema(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": self._describe(tool),
                        "parameters": build_parameters(tool.input_model),
                    },
                },
                max_desc_len=budget,
            )
            for name, tool in self.tools.items()
            if name in selected
        ]

    def _desc_budget(self) -> int:
        """由 token 设置推导出的每条描述字符预算。"""
        if self.settings is None:
            return DEFAULT_MAX_DESC_LEN
        # 用文档给出的中文近似值做 token -> 字符换算，使该设置仍以文档所用单位表达。
        return max(int(self.settings.context.max_tool_schema_tokens * CHARS_PER_TOKEN), 20)

    # -- 检索 ---------------------------------------------------------------
    def search(self, query: str, *, limit: int = 5) -> list[ToolBrief]:
        """在名称/描述上做关键词加权检索，并把结果整理成可据以调用的形式。

        v1 刻意采用词匹配加与时效无关的权重（docs 3.4.3）：不用向量库。条目里带上
        参数清单，使一次检索就足以发起调用（见 ``ToolBrief``）。
        """
        terms = [term for term in query.lower().replace("，", " ").split() if term]
        scored: list[ToolBrief] = []
        for name, tool in self.tools.items():
            haystack = f"{name} {tool.description}".lower()
            score = 0
            for term in terms:
                if term in name.lower():
                    score += 3
                if term in haystack:
                    score += 1
            if score:
                scored.append(self.brief(name, tool, score=score))
        scored.sort(key=lambda brief: (-brief.score, brief.name))
        return scored[:limit]

    def brief(self, name: str, tool: BaseTool | None = None, *, score: int = 0) -> ToolBrief:
        """一个工具的可调用摘要（名称、用途、参数清单、层级与激活状态）。"""
        resolved = tool if tool is not None else self.tools[name]
        spec = getattr(type(resolved), "__tool_spec__", None)
        inputs: tuple[str, ...] = ()
        required: frozenset[str] = frozenset()
        if spec is not None:
            inputs = tuple(
                spec.required_names + spec.optional_names
            )
            required = frozenset(spec.required_names)
        return ToolBrief(
            name=name,
            description=resolved.description,
            group=resolved.group.value,
            tier=getattr(getattr(type(resolved), "tier", None), "value", Tier.RESIDENT.value),
            active=self.is_active(name),
            score=score,
            inputs=inputs,
            required=required,
            output_note=resolved.output_schema_note,
        )

    @staticmethod
    def _describe(tool: BaseTool) -> str:
        if tool.output_schema_note:
            return tool.description + " " + tool.output_schema_note
        return tool.description
