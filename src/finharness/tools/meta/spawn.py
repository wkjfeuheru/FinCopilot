"""spawn_agent：把相互隔离的并行任务分派给子代理（docs 03.10）。

主 Agent 的上下文窗口是稀缺资源。当一个任务的*中间*材料会挤占它时——消化若干长文档、
比较彼此独立的工作线——本工具会让每个任务在拥有自己上下文的子代理中运行，只返回结论。

有三项性质是刻意设计的：

* **一次调用承载所有任务。** 扇出发生在单次工具调用内部，因此并行性不依赖于模型记得
  再次调用。它也让开销只有一个清晰可见的归属。
* **子代理不取数。** 它们只消费交给自己的材料（内联给出，或作为可 ``read_file`` 的
  路径）。取数是主 Agent 的职责，因为那里知道股票代码与期间；只给一句话任务的 worker
  只能靠猜。
* **产出发现，而非修改。** 每个子代理都是只读的。主 Agent 决定如何处理返回的内容。
"""

from __future__ import annotations

from pydantic import BaseModel

from finharness.data.raw import RawData
from finharness.shared.agents import GENERAL_FOCUS, MAX_SPAWN_TASKS
from finharness.shared.declaration import Capability, Tier, ToolGroup, param, tool
from finharness.tools.base import BaseTool

_TASKS_HELP = (
    "并行子任务列表，每项须自包含（谁、要什么、材料在哪）。"
    f"1-{MAX_SPAWN_TASKS} 项，各自独立无先后依赖。"
)


def _require_a_task(value: BaseModel) -> BaseModel:
    """空条目会派生一个无事可做的子代理，为回答一个空串付出整个上下文的代价。

    在参数模型层拒绝，而不是留到下游。
    """
    tasks = getattr(value, "tasks", None) or []
    if not any(task and task.strip() for task in tasks):
        raise ValueError("tasks 至少需要一项非空任务")
    return value


@tool(
    name="spawn_agent",
    description=(
        "把若干各自独立、无需相互等待的任务分派给子代理并行处理，"
        "每个子代理在自己的上下文里完成、只回结论，因此大量中间材料不占用主上下文。"
        "子代理只读、不取数：材料需写在任务里或以本地文件路径给出。"
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
    @param("tasks", annotation=list[str], desc=_TASKS_HELP, min_length=1, max_length=MAX_SPAWN_TASKS)
    @param("context", desc="所有子任务共享的背景说明，可选")
    async def _dispatch(self, *, tasks: list[str], context: str | None = None) -> RawData:
        """把任务列表交给协调器并行派生，返回聚合后的各子任务结论文本。"""
        if self.coordinator is None:
            raise ValueError("当前会话未启用子代理协调器，无法派发子任务")
        results = await self.coordinator.spawn(
            tasks=tasks, focus=GENERAL_FOCUS, context=context
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
        "以上为各子代理的独立结论，主 Agent 需自行核对并汇总；子代理的过程与中间材料未回流。"
    )
    return "\n".join(lines).rstrip()
