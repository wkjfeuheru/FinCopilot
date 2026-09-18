"""write_report：把结构化大纲转成报告产物（文档 3.9）。

刻意不接收 ``path`` 参数。输出位置由主题推导，这带来两点后果：
命名约定无法被绕过；并且——由于权限闸门只对携带产物目录内
路径的写入跳过确认——生成报告时总会先征求用户同意。

撰写报告还会运行风险审查子代理（文档 03.10）。这是审查的唯一触发点：
把它放在这里，而不是放在一个模型可以自行选择的工具里，能让"每份报告
都经过审查"成为系统的一种属性，而非对模型行为的期望。编排逻辑本身位于
``report/review.py``；本工具只负责渲染，然后调用它一次。

复核意见随工具**文本**返回给模型，不构成产物：sidecar 文件（``.review.md``）
只用于内容哈希闩锁与审计取证，因此它不入 ``attachments``，也不在结果里给出
路径——终审是模型对模型的内部闸门，用户侧不该看到一份"第二报告"（文档 03.10.7）。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.coordinator.review import format_review_lines, review_report
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool
from finharness.tools.fin.report_pipeline import (
    ReportOutline,
    ReportPipeline,
    ReportSection,
    ReportValidationError,
)


class SectionInput(BaseModel):
    """章节结构。

    这是 ``params_model`` 逃生口的用途：``sections`` 是一个**嵌套结构**（每项含标题、
    正文、引用与图表），其形态由报告管线定义，而不是工具的调用参数。工具的调用参数
    仍由 ``@param`` 声明，只有这个嵌套元素沿用模型。
    """

    heading: str = Field(description="章节标题")
    body: str = Field(description="章节正文（markdown，结论性数字需带 {cite:cid}）")
    cids: list[str] = Field(default_factory=list, description="本章节引用的 citation id")
    charts: list[str] = Field(default_factory=list, description="本章节引用的图表路径（make_chart 产出）")


@tool(
    name="write_report",
    description=(
        "把结构化大纲渲染成研报（markdown + docx）。正文中的结论性数字必须带引用，"
        "未标注来源的数字会在报告中被标记。"
    ),
    capability=Capability.OUTPUT,
    group=ToolGroup.FIN_OUTPUT,
    permission="write",
    # 报告渲染，外加一轮针对真实服务提供方的审查调用。默认的
    # 30s 预算无法覆盖一次审查所需的提供方往返耗时。
    timeout=300,
    output_schema_note="产出 output/<topic>_<YYYYMMDD>.md 与 .docx，并附风险终审意见。",
    needs_coordinator=True,
)
class WriteReportTool(BaseTool):
    @param("topic", desc="报告主题，用于文件命名")
    @param("core_view", desc="核心观点，每条须含 {cite:cid}")
    @param("sections", annotation=list[SectionInput], desc="正文章节")
    @param("risks", desc="风险提示，建议 ≥5 条")
    @param("formats", desc="产出形态，默认同时产出 md 与 docx")
    async def _dispatch(
        self,
        *,
        topic: str,
        core_view: list[str],
        sections: list[dict],
        risks: list[str],
        formats: list[str] | None = None,
    ) -> RawData:
        """按结构化大纲渲染报告，运行风险审查，并返回带附件的文本结果。"""
        outline = ReportOutline(
            topic=topic,
            core_view=list(core_view),
            sections=[
                ReportSection(
                    heading=str(item.get("heading", "")),
                    body=str(item.get("body", "")),
                    cids=[str(c) for c in item.get("cids") or []],
                    charts=[str(c) for c in item.get("charts") or []],
                )
                for item in sections
            ],
            risks=list(risks),
        )
        pipeline = ReportPipeline(
            cite=self.ctx.cite,
            settings=self.data.settings,
            # 工具目录同时充当检测字典：正文只要点名了任何内部工具——
            # 无论本次会话是否调用过——就是在暴露系统内部管线，
            # 而不是在面向读者行文。
            tool_names=set(self.registry.names()) if self.registry is not None else set(),
        )
        try:
            # 落盘 markdown 与 docx 导出（可能起 pandoc 子进程）都是阻塞调用，
            # 放进线程后 300s 的工具超时才真正生效，且渲染期间不占住事件循环。
            artifact = await asyncio.to_thread(
                pipeline.export, outline, formats=tuple(formats or ("md", "docx"))
            )
        except ReportValidationError as exc:
            # 精确、可操作的反馈，便于模型修正大纲。
            raise ValueError("报告校验未通过：" + "；".join(exc.problems)) from exc

        attachments = [artifact.markdown_path]
        if artifact.docx_path:
            attachments.append(artifact.docx_path)

        lines = [f"已生成报告：{artifact.topic}"]
        lines.append(f"- markdown：{artifact.markdown_path}")
        if artifact.docx_path:
            lines.append(f"- docx：{artifact.docx_path}")
        if artifact.docx_error:
            lines.append(f"- docx 导出失败（已保留 markdown）：{artifact.docx_error}")
        if artifact.citations:
            lines.append(f"- 引用数据 {len(artifact.citations)} 条")
        if artifact.warnings:
            lines.append("- 提示：")
            lines.extend(f"  - {w}" for w in artifact.warnings)

        outcome = await review_report(
            self.coordinator, topic=artifact.topic, markdown_path=artifact.markdown_path
        )
        lines.extend(format_review_lines(outcome))

        return RawData(
            kind="text",
            text="\n".join(lines),
            paths=attachments,
            endpoint="report:pipeline",
            params={"topic": artifact.topic, "citations": len(artifact.citations)},
            # 审计钩子读取此项来写入审查自身的审计记录；该字段绝不会发送给
            # 模型服务提供方。其中的未结事项同时供循环写入会话状态（docs 03.10.7）。
            metadata={"review": outcome.audit_metadata()},
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """报告结果会携带文件附件，供传输层使用。"""
        return (raw.text or "（报告生成失败）"), [raw]
