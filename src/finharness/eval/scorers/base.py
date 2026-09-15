"""四个评测维度的评分器（docs 03.13）。

每个评分器把已完成的 ``CaseRun`` 转换为一项部分得分以及一组具名检查项。
检查项基于行为且由代码校验；规则无法判定的内容会记为未检查项，而不是
默默判为通过。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from finharness.eval.config import EvalConfig
from finharness.eval.runner import CaseRun
from finharness.eval.schema import EvalCase


@dataclass(slots=True)
class Check:
    """一条由代码校验的断言及其结果。"""

    name: str
    passed: bool
    detail: str = ""
    dimension: str = ""


@dataclass(slots=True)
class DimensionScore:
    """某维度在 [0, 1] 区间内的得分，以及产生该得分的检查项。"""

    dimension: str
    score: float = 0.0
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> Check:
        """新增一条检查项并返回它。"""
        check = Check(name=name, passed=passed, detail=detail, dimension=self.dimension)
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


__all__ = ["Check", "DimensionScore"]
