"""spawn_agent：把相互隔离的并行任务分派给子代理（docs 03.10）。

主 Agent 的上下文窗口是稀缺资源。当一个工作单元的*中间*过程会挤占它时——逐实体深分析多家公司、
消化若干长文档、多角度交叉印证公开讨论——本工具会让每个任务在拥有自己上下文的子代理中运行，只返回结论。

有三项性质是刻意设计的：

* **一次调用承载所有任务。** 扇出发生在单次工具调用内部，因此并行性不依赖于模型记得
  再次调用。它也让开销只有一个清晰可见的归属。任务必须**逐单元拆解**：一个实体或一个
  检索角度一条任务，绝不把整个多实体请求原样当成一条任务。
* **子代理只读取数。** 任务点名了标的（代码/公司名/行业）时，子代理可**自行取数**再分析；
  材料已给出时则直接消化（内联文本，或可 ``read_file`` 的路径）。任务要求检索公开网页时
  它可以 ``web_search``。它不能写、不能再派子代理。若任务未点名标的，子代理会**如实回报缺什么**，
  而不是猜。
* **产出发现，而非修改。** 每个子代理都是只读的。主 Agent 决定如何处理返回的内容，并负责
  把各单元结论**汇合**成覆盖每个实体或每个角度的统一结论。

**触发是引擎侧的确定性动作**：当请求被判为"多实体逐实体分析"（`shared/fanout.py::fanout_intent`）
时，引擎在 hydrate 阶段预激活本工具并下发强制编排指令——因为仅靠提示词劝导，实测模型会走
串行取数的省事路（隐式触发率很低）。
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from finharness.data.raw import RawData
from finharness.shared.agents import GENERAL_FOCUS, MAX_SPAWN_TASKS
from finharness.shared.declaration import Capability, Tier, ToolGroup, param, tool
from finharness.tools.base import BaseTool

_TASKS_HELP = (
    "并行子任务列表，每项须自包含且**逐单元拆解**（谁、要什么、标的/区间）。"
    f"1-{MAX_SPAWN_TASKS} 项，各自独立无先后依赖。"
    "一个实体一条任务——绝不把整个多实体请求原样作为一条任务。"
)

# 6 位股票代码（A 股）。只用于"单条任务里塞了多个实体"这一处窄校验。
_STOCK_CODE_RE = re.compile(r"\b\d{6}\b")


def _require_a_task(value: BaseModel) -> BaseModel:
    """空条目会派生一个无事可做的子代理，为回答一个空串付出整个上下文的代价。

    在参数模型层拒绝，而不是留到下游。另外做一处**窄范围**的拆解校验：若整批只有
    一条任务、且该条文本含 ≥2 个不同的 6 位股票代码，说明主 Agent 把"多实体请求"
    原样当成了一条任务——那只会派生一个子代理，没有隔离收益。此时要求它拆解，
    或（若确实不可拆，如横向对比）不要扇出、直接作答。
    """
    tasks = [task for task in (getattr(value, "tasks", None) or []) if task and task.strip()]
    if not tasks:
        raise ValueError("tasks 至少需要一项非空任务")
    if len(tasks) == 1:
        codes = set(_STOCK_CODE_RE.findall(tasks[0]))
        if len(codes) >= 2:
            raise ValueError(
                "单条任务里出现了多个标的（"
                + "、".join(sorted(codes))
                + "）。请把请求**拆解为逐实体的自包含任务**，一条任务一个实体后再派发；"
                "若这是不可拆的整体（如横向对比），则不该派子代理，直接作答即可。"
            )
    return value


@tool(
    name="spawn_agent",
    description=(
        "把若干各自独立、无需相互等待的任务分派给子代理并行处理，"
        "每个子代理在自己的上下文里完成、只回结论，因此大量中间过程不占用主上下文。"
        "**先按单元拆解**：一个实体一条任务（谁、要什么、标的/区间），"
        "或每个检索角度/长文档摘要分片一条任务（交叉印证、多角度公开讨论、并行检索），"
        "绝不把整个多实体请求原样作为一条任务。"
        "子代理只读、可自行取数：任务点名了标的它就去取该实体的数据再分析；"
        "材料已给出则直接消化；任务要求检索公开网页时它自己 web_search，主上下文只汇合结论。"
        "返回后主 Agent 需把各单元结论汇合为覆盖每个实体或每个角度的统一结论。"
    ),
    capability=Capability.META,
    # 扇出会派生多个并发子代理，成本高且多数问题用不到，故按需注入。
    tier=Tier.LAZY,
    group=ToolGroup.META,
    # 扇出至多 MAX_SPAWN_TASKS 个并发的子代理，每个都要运行自己的若干轮，
    # 因此预算须覆盖最慢的那个兄弟任务，而不是一次模型调用。
    timeout=300,
    # 由循环注入（docs 03.10）：工具无法自行构建协调器，因为那需要 provider，
    # 而工具永远看不到它。
    needs_coordinator=True,
    output_schema_note="返回每个子任务的结论与用量；子代理过程不回流。",
    model_validator=_require_a_task,
)
class SpawnAgentTool(BaseTool):
    # 子任务可能跑数分钟；没有真实进展事件时，前端空闲看门狗会把仍在跑的
    # 扇出误判为卡死。每个子任务的起止走现有 tool_progress 通道。
    needs_progress = True

    @param("tasks", annotation=list[str], desc=_TASKS_HELP, min_length=1, max_length=MAX_SPAWN_TASKS)
    @param("context", desc="所有子任务共享的背景说明，可选")
    async def _dispatch(self, *, tasks: list[str], context: str | None = None) -> RawData:
        """把任务列表交给协调器并行派生，返回聚合后的各子任务结论文本。"""
        if self.coordinator is None:
            raise ValueError("当前会话未启用子代理协调器，无法派发子任务")
        report = getattr(self, "progress", None)

        async def on_task(index: int, total: int, task: str, status: str) -> None:
            if not callable(report):
                return
            first = task.strip().splitlines()[0] if task.strip() else task
            await report(
                {
                    "phase": "spawn",
                    "index": index,
                    "total": total,
                    "task": first,
                    "status": status,
                }
            )

        results = await self.coordinator.spawn(
            tasks=tasks, focus=GENERAL_FOCUS, context=context, on_task=on_task
        )
        return RawData(
            kind="text",
            text=_render(results),
            endpoint="meta:spawn_agent",
            params={"tasks": len(tasks), "focus": GENERAL_FOCUS},
        )


def _render(results: list) -> str:
    """每个任务一个区块，按给定顺序排列，失败如实表述为失败。"""
    ok_count = sum(1 for item in results if item.ok)
    lines = [f"已派发 {len(results)} 个子任务，成功 {ok_count} 个。", ""]
    for index, item in enumerate(results, start=1):
        lines.append(f"## 子任务 {index}")
        if item.task:
            # 回显任务，使某条结论不会被误认作另一条的。
            first_line = item.task.strip().splitlines()[0]
            lines.append(f"任务：{first_line}")
        if item.ok:
            lines.append(item.summary or "（子代理未给出结论）")
        else:
            # 如实报告失败，而不是一个需要阅读者自己留意的空区块。
            lines.append(f"（该子任务未能完成：{item.error or '未知原因'}）")
        if item.citations:
            lines.append("引用数据：" + "、".join(f"{{cite:{cid}}}" for cid in item.citations))
        lines.append("")
    lines.append(
        "以上为各子代理的独立结论。请**逐一核对并把它们汇合**成覆盖用户点名每个实体的统一结论"
        "（标明各自来源；子代理的过程与中间材料未回流）。"
    )
    return "\n".join(lines).rstrip()
