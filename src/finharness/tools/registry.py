"""两层工具注册表：常驻 schema 加上可检索的懒加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from finharness.context.trim import DEFAULT_MAX_DESC_LEN, trim_schema
from finharness.data.access import DataAccess
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup

# 工具类经由 loader 注册，使目录扩张时该模块仍可导入。
from finharness.tools.fin.announcements import GetAnnouncementsTool
from finharness.tools.fin.backtest import RunBacktestTool
from finharness.tools.fin.chart import MakeChartTool
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
from finharness.tools.generic.web import WebSearchTool
from finharness.tools.meta.ask import AskUserTool
from finharness.tools.meta.discovery import LoadToolTool, SearchToolsTool
from finharness.tools.meta.plan import (
    RecordConclusionTool,
    ResearchPlanTool,
    UpdatePlanStepTool,
)
from finharness.tools.meta.preference import RememberPreferenceTool
from finharness.tools.meta.skills import ListSkillsTool, LoadSkillTool
from finharness.tools.meta.spawn import SpawnAgentTool

# 这里的每个名称都必须解析到一个已注册的类——没有实现的懒加载条目会被
# lazy_names() 列出却无法触达，因此在对应工具存在之前，run_backtest / read_pdf
# 不予列入。
#
# 联网访问与研报设为懒加载，是因为大多数金融问题从不需要它们，而常驻 schema 会在
# 每次请求时重新发送；系统提示会点名它们，使发现过程不依赖于模型主动先去检索。
DEFAULT_LAZY_TOOLS: tuple[str, ...] = (
    "get_announcements",
    "calc_valuation",
    "web_search",
    "get_research_reports",
    "spawn_agent",
    # 重量级计算（面板取数、IC/分组回测）：由量化因子场景按需激活，而非随每次请求携带。
    "run_backtest",
)

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
    # 金融-计算
    CalcMetricsTool,
    CalcValuationTool,
    RunBacktestTool,
    # 金融-输出
    MakeChartTool,
    WriteReportTool,
    # 通用
    ReadFileTool,
    WriteFileTool,
    WebSearchTool,
    # 元
    ResearchPlanTool,
    UpdatePlanStepTool,
    RecordConclusionTool,
    SearchToolsTool,
    ListSkillsTool,
    LoadSkillTool,
    LoadToolTool,
    SpawnAgentTool,
    AskUserTool,
    RememberPreferenceTool,
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
      产生副作用，而 ``load_tool`` 会让复核者自行扩大其目录。
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
    """检索结果：足以描述一个懒加载工具，而无需其完整 schema。"""

    name: str
    description: str
    group: str
    score: int = 0


def build_parameters(model: type[BaseModel]) -> dict[str, Any]:
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
            # 限域目录是一个工作集，因此其中没有任何工具是懒加载的：激活路径需要
            # ``load_tool``，而受限于限域的注册表没有它，所以这里的懒加载工具会被
            # 列出却无法触达。
            self._lazy: set[str] = set()
        elif configured_resident:
            # 显式的常驻列表优先；其余一切都变为懒加载。
            self._lazy = {name for name in self.tools if name not in configured_resident}
        elif configured_lazy:
            self._lazy = {name for name in configured_lazy if name in self.tools}
        else:
            self._lazy = {name for name in DEFAULT_LAZY_TOOLS if name in self.tools}
        # 懒加载工具只有在激活后才进入请求（docs 03.4.3）。
        self._active: set[str] = {name for name in self.tools if name not in self._lazy}

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

    def activate(self, name: str) -> bool:
        """将一个工具加入已注入集合；对未知或已激活的工具返回 False。"""
        if name not in self.tools or name in self._active:
            return False
        self._active.add(name)
        return True

    # -- schema --------------------------------------------------------------
    def schemas(self, names: set[str] | None = None) -> list[dict]:
        """已注入集合的 OpenAI 风格 schema（按注册表顺序）。

        ``names`` 默认为活动集合，因此懒加载工具会一直隐藏，直到激活后的下一轮。
        描述会按配置的预算裁剪，因为每个常驻 schema 都会在每次请求时重新发送
        （docs 3.6.1）。
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
        return max(int(self.settings.context.max_tool_schema_tokens * 1.7), 20)

    # -- 检索 ---------------------------------------------------------------
    def search(self, query: str, *, limit: int = 5) -> list[ToolBrief]:
        """在名称/描述上做关键词加权检索，并将懒加载工具编入目录。

        v1 刻意采用词匹配加与时效无关的权重（docs 3.4.3）：不用向量库。
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
                scored.append(
                    ToolBrief(
                        name=name,
                        description=tool.description,
                        group=tool.group.value,
                        score=score,
                    )
                )
        scored.sort(key=lambda brief: (-brief.score, brief.name))
        return scored[:limit]

    @staticmethod
    def _describe(tool: BaseTool) -> str:
        if tool.output_schema_note:
            return tool.description + " " + tool.output_schema_note
        return tool.description
