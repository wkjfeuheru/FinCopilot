"""FinHarness agent 的评测资产（docs 03.13）。

测试集是 ``evals/cases/d*.yaml``（v2.0：200 条 × 5 维度 normal / missing /
failure / risk / noise，权威说明见 ``evals/README.md``；旧人工题库
``docs/测试问题集-功能与幻觉.md`` 已归档至 ``docs/archive/``）。本包加载
用例、驱动真实或仿真的 ``AgentLoop``，并对四个维度打分：

* 任务完成率 —— 结构化、由代码校验的验收标准；
* 轨迹正确性 —— 记录的 Thought/Action/Observation 路径与期望路径的对比；
* 效率 —— token 消耗、步数与时延相对于预算的表现；
* 安全 —— 拒答正确性与高风险调用拦截。

本包不改变引擎行为；它只读取引擎记录的轨迹。
"""

from finharness.eval.config import EvalConfig, SetSpec
from finharness.eval.schema import (
    EvalCase,
    Expect,
    TurnCase,
    load_cases,
    load_cases_dir,
)

__all__ = [
    "EvalCase",
    "EvalConfig",
    "Expect",
    "SetSpec",
    "TurnCase",
    "load_cases",
    "load_cases_dir",
]
