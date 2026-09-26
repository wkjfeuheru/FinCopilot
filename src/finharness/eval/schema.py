"""评测工具的用例 schema 与 YAML 加载（docs 03.13）。

一个用例是一道问题（或一段简短的多轮对话），外加从问题集“通过标准/典型失败
信号”两列翻译而来的结构化验收标准。该 schema 刻意基于行为：它断言的是调用了
哪些工具、是否形成计划、拒答与引用，而非答案的字面文本，因为模型的措辞每次
运行都会变化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class SchemaError(ValueError):
    """当用例文件格式错误时抛出；携带出错位置。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerExpect(_Model):
    """对面向读者的答案文本的断言。"""

    # True = 必须拒答/说明边界；False = 不得拒答。
    refusal: bool | None = None
    # 覆盖本用例默认的拒答标记。当边界措辞已知且默认列表过于粗糙时使用
    # （例如某用例的合规答案中正当地包含针对某个数据点的“无法判断”）。
    refusal_markers: list[str] = Field(default_factory=list)
    contains_any: list[str] = Field(default_factory=list)
    contains_all: list[str] = Field(default_factory=list)
    contains_none: list[str] = Field(default_factory=list)
    # 这些正则中任意一条命中即可（Python re 语法）。
    matches_any: list[str] = Field(default_factory=list)
    # 答案必须点名"排行取数结果"里名列前 N 的实体——用来把「有没有真的告诉用户是哪几个」
    # 变成确定性断言（不依赖当场数据排名，因而是日期无关的）。观测对象取自本用例中
    # 返回排行表的工具观测（如 get_industry_perf 的 view="ranking"）。
    top_names_from_observations: int | None = None
    min_chars: int | None = None
    max_chars: int | None = None


class TrajectoryExpect(_Model):
    """对记录的 Thought/Action/Observation 路径的断言。"""

    # 每个列出的工具在整个用例过程中必须至少被调用一次。
    tools_must: list[str] = Field(default_factory=list)
    # 这些工具中至少有一个必须被调用。
    tools_any: list[str] = Field(default_factory=list)
    # 这些工具都不得被调用（此处指已执行的调用，而非被拒）。
    tools_must_not: list[str] = Field(default_factory=list)
    skills_must: list[str] = Field(default_factory=list)
    skills_must_not: list[str] = Field(default_factory=list)
    plan_required: bool | None = None
    # 相对顺序：每个内层列表是一个必须按序出现的子序列。
    order: list[list[str]] = Field(default_factory=list)
    # 同一 (工具, 参数) 调用最多可执行的次数。
    max_repeats: int | None = None
    max_rounds: int | None = None
    max_tool_calls: int | None = None
    # 这些工具不得作为成功执行出现；它们仍可被尝试并拒绝（这正是安全用例）。
    must_not_succeed: list[str] = Field(default_factory=list)
    # 扇出编排契约：**若**本次调用了 spawn_agent，则每个列出的实体标识（股票代码、
    # 行业名等）必须至少出现在一条派出的子任务里——即主 Agent 确实做了逐单元拆解，
    # 而不是把整个多实体请求原样当成一条任务。**未扇出时该项自动跳过**，因为扇出是
    # 自由裁量；该断言只在"已触发扇出"时验证拆解质量。
    spawn_tasks_cover: list[str] = Field(default_factory=list)


class CitationExpect(_Model):
    min: int = 0
    # True = 至少需要一条引用。
    required: bool = False


class SafetyExpect(_Model):
    """针对拒答与高风险拦截的专项断言。"""

    # 必须被拒绝（绝不执行）的工具——即“明确的拒绝机制”。
    blocked_tools: list[str] = Field(default_factory=list)
    # True = 运行必须干净地停止而非降级；None = 不检查。
    no_degradation: bool | None = None


class ArtifactExpect(_Model):
    """针对导出报告文件的断言。"""

    report_exported: bool = False
    # 导出报告中不得残留任何 ``[!无来源:n]`` 标记。
    no_unsourced_numbers: bool = True
    # 面向读者的报告正文不得出现内部工具名。
    no_tool_names: bool = True


class Expect(_Model):
    answer: AnswerExpect = Field(default_factory=AnswerExpect)
    trajectory: TrajectoryExpect = Field(default_factory=TrajectoryExpect)
    citations: CitationExpect = Field(default_factory=CitationExpect)
    safety: SafetyExpect = Field(default_factory=SafetyExpect)
    artifacts: ArtifactExpect = Field(default_factory=ArtifactExpect)


class Budget(_Model):
    max_tokens: int | None = None
    max_seconds: float | None = None
    max_rounds: int | None = None
    max_steps: int | None = None


InteractivePolicy = Literal["none", "confirm", "deny", "answer"]


class TurnCase(_Model):
    user: str
    # 评测工具如何应答交互提示（ask_user / 写操作确认）：
    #   none    — 无通道；ask_user 报告为未作答，写操作被拒
    #   confirm — 批准写操作确认
    #   deny    — 拒绝写操作确认
    #   answer  — ask_user 以 ``interactive_answer`` 作答
    interactive: InteractivePolicy = "answer"
    interactive_answer: str = "综合"
    expect: Expect = Field(default_factory=Expect)


class ChatCase(_Model):
    """多对话用例中的一个对话：一组轮次 + 可选的固定 conversation_id。

    ``conversation_id`` 省略时由 runner 按用例 id 与序号生成。多个对话共享
    同一 store 与 user_id，因此它们之间的长期记忆是可见的——这正是
    "跨对话记忆"用例得以表达的方式。
    """

    conversation_id: str | None = None
    turns: list[TurnCase] = Field(min_length=1)


class EvalCase(_Model):
    id: str
    title: str = ""
    category: str = ""
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    # 标记规则无法完全判定的标准（后续交由 LLM 裁判处理）。
    judge: Literal["none", "todo"] = "none"
    notes: str = ""
    # 单对话用例的简写形式：等价于只有一个 chat。
    turns: list[TurnCase] = Field(default_factory=list)
    # 多对话用例（跨对话记忆）：各对话依次运行，共享 store 与 user_id。
    chats: list[ChatCase] = Field(default_factory=list)
    # 多对话用例中，每个对话结束后是否同步跑一次蒸馏，使后续对话能召回
    # decision/excerpt 情节。默认写入的 task_result 情节已随门控关闭
    # （ltm.auto_task_episodes=false），故跨对话用例依赖本蒸馏或显式写入
    # （remember_preference 等）；蒸馏失败（如离线自检 provider 不产出 JSON）
    # 时若用例本身走的是显式写入，仍成立。
    distill_between_chats: bool = True
    budget: Budget = Field(default_factory=Budget)

    @model_validator(mode="after")
    def _exactly_one_form(self) -> EvalCase:
        if self.turns and self.chats:
            raise ValueError("turns 与 chats 只能提供其一")
        if not self.turns and not self.chats:
            raise ValueError("用例必须提供 turns 或 chats 之一")
        return self

    def conversation_groups(self) -> list[tuple[str | None, list[TurnCase]]]:
        """规整为 ``[(conversation_id | None, turns), ...]``。

        单对话形式返回 ``[(None, turns)]``：由 runner 决定用哪个 id（沿用
        既有的"以用例 id 为键"行为，使旧用例的记忆表现完全不变）。
        """
        if self.chats:
            return [(chat.conversation_id, list(chat.turns)) for chat in self.chats]
        return [(None, list(self.turns))]

    @property
    def all_turns(self) -> list[TurnCase]:
        """按执行顺序扁平化的全部轮次（断言与评分读它，两种形式通吃）。

        多对话用例的断言写在 ``chats[*].turns`` 上，而评分器逐轮比对
        ``run.turns``；扁平化后两者的索引一一对应。
        """
        return [turn for _cid, turns in self.conversation_groups() for turn in turns]


def _as_list(payload: Any, *, where: str) -> list[dict]:
    """把 YAML 载荷规整为用例列表，必要时解包 ``cases`` 键。"""
    if payload is None:
        return []
    if isinstance(payload, dict) and "cases" in payload:
        payload = payload["cases"]
    if not isinstance(payload, list):
        raise SchemaError(f"{where}: 顶层必须是用例列表（- id: ...）")
    return payload


def load_cases(path: str | Path) -> list[EvalCase]:
    """加载并校验一个含用例列表的 YAML 文件。"""
    file_path = Path(path)
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SchemaError(f"{file_path}: YAML 解析失败：{exc}") from exc
    except OSError as exc:
        raise SchemaError(f"{file_path}: 无法读取：{exc}") from exc

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for entry in _as_list(raw, where=str(file_path)):
        try:
            case = EvalCase.model_validate(entry)
        except ValidationError as exc:
            raise SchemaError(f"{file_path}: 用例校验失败：{exc}") from exc
        if case.id in seen:
            raise SchemaError(f"{file_path}: 用例 id 重复：{case.id}")
        seen.add(case.id)
        cases.append(case)
    return cases


def load_cases_dir(directory: str | Path) -> list[EvalCase]:
    """加载目录下所有 ``*.yaml``，按文件名排序以保证稳定。"""
    root = Path(directory)
    if not root.exists():
        raise SchemaError(f"用例目录不存在：{root}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for file_path in sorted(root.glob("*.yaml")):
        for case in load_cases(file_path):
            if case.id in seen:
                raise SchemaError(f"{file_path}: 用例 id 与其它文件重复：{case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise SchemaError(f"用例目录为空：{root}")
    return cases


__all__ = [
    "AnswerExpect",
    "ArtifactExpect",
    "Budget",
    "ChatCase",
    "CitationExpect",
    "EvalCase",
    "Expect",
    "InteractivePolicy",
    "SchemaError",
    "SafetyExpect",
    "TrajectoryExpect",
    "TurnCase",
    "load_cases",
    "load_cases_dir",
]
