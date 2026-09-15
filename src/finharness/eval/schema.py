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
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SchemaError(ValueError):
    """当用例文件格式错误时抛出；携带出错位置。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _freeze_list(value: Any) -> Any:
    return list(value) if value is not None else []


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

    def is_empty(self) -> bool:
        return all(
            not model.model_dump(exclude_defaults=True)
            for model in (
                self.answer,
                self.trajectory,
                self.citations,
                self.safety,
                self.artifacts,
            )
        )


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


class EvalCase(_Model):
    id: str
    title: str = ""
    category: str = ""
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    # 标记规则无法完全判定的标准（后续交由 LLM 裁判处理）。
    judge: Literal["none", "todo"] = "none"
    notes: str = ""
    turns: list[TurnCase] = Field(min_length=1)
    budget: Budget = Field(default_factory=Budget)


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
