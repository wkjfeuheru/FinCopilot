"""评测配置：权重、红线、用例集与预算。

从 ``evals/config.yaml`` 加载。所有内容均可覆盖，因此加权综合分与
通行/拦截门禁都能在不改代码的情况下调整。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigError(ValueError):
    """当配置文件格式错误时抛出。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Weights(_Model):
    """综合分各维度权重；其和必须为正数。"""

    task: float = 0.40
    trajectory: float = 0.25
    efficiency: float = 0.15
    safety: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "task": self.task,
            "trajectory": self.trajectory,
            "efficiency": self.efficiency,
            "safety": self.safety,
        }

    @field_validator("task", "trajectory", "efficiency", "safety")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("权重不能为负")
        return value


class EfficiencyWeights(_Model):
    tokens: float = 0.40
    steps: float = 0.30
    latency: float = 0.30

    def as_dict(self) -> dict[str, float]:
        return {"tokens": self.tokens, "steps": self.steps, "latency": self.latency}


class EfficiencyConfig(_Model):
    sub_weights: EfficiencyWeights = Field(default_factory=EfficiencyWeights)
    # 当用例未声明自身预算时应用的兜底预算。
    default_max_tokens: int = 200_000
    default_max_rounds: int = 12
    default_max_seconds: float = 240.0


class GateConfig(_Model):
    """无论综合分多少都会使运行失败的红线。"""

    # 任何带有这些标签之一的失败用例都会使门禁不通过。
    red_line_tags: list[str] = Field(default_factory=lambda: ["red-line"])
    # 声明了 safety.blocked_tools 的用例，只要有拦截未命中，门禁即不通过。
    require_blocked: bool = True
    # 未触发红线时通过所需的最低综合分。
    min_composite: float = 0.0


class SetSpec(_Model):
    """具名用例集：对所提供的选择器取或；全部为空 = 所有用例。"""

    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    ids: list[str] = Field(default_factory=list)

    def matches(self, case) -> bool:
        if not (self.tags or self.categories or self.ids):
            return True
        return (
            bool(set(case.tags) & set(self.tags))
            or case.category in set(self.categories)
            or case.id in set(self.ids)
        )


class EvalConfig(_Model):
    weights: Weights = Field(default_factory=Weights)
    efficiency: EfficiencyConfig = Field(default_factory=EfficiencyConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    sets: dict[str, SetSpec] = Field(
        default_factory=lambda: {
            "smoke": SetSpec(tags=["smoke"]),
            "core": SetSpec(tags=["smoke", "core", "red-line"]),
            "full": SetSpec(),
        }
    )
    # 目录名相对于配置文件所在目录解析。
    cases_dir: str = "cases"

    def resolve_set(self, name: str) -> SetSpec:
        if name not in self.sets:
            raise ConfigError(
                f"未知题集：{name}；可选：{', '.join(sorted(self.sets))}"
            )
        return self.sets[name]


def load_config(path: str | Path) -> EvalConfig:
    """加载并校验评测配置；文件不存在时返回默认配置。"""
    file_path = Path(path)
    if not file_path.exists():
        return EvalConfig()
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file_path}: YAML 解析失败：{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{file_path}: 顶层必须是映射")
    try:
        return EvalConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - pydantic 的消息本身就是载荷
        raise ConfigError(f"{file_path}: 配置校验失败：{exc}") from exc


__all__ = [
    "ConfigError",
    "EfficiencyConfig",
    "EfficiencyWeights",
    "EvalConfig",
    "GateConfig",
    "SetSpec",
    "Weights",
    "load_config",
]
