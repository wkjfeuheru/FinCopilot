"""东方财富研报（文档 03.4）。

这是 A 股数据工具的"研究"对位物：它不展示公司自身的数字，而是呈现券商
发布的内容——某一时间窗口内的行业与公司研报，可按行业、机构或标题关键词
进一步收窄。

有三件事决定了接口的形态：

* **参数贴近用户的提问方式。** "爬 3 份"对应 ``top_n``；"某个行业 /
  某家机构 / 某段时间"对应 ``industry`` / ``institution`` / ``start_date`` +
  ``end_date``。默认值（行业研报、近 30 天、10 条）匹配常见请求，
  因此不带参数调用也有用。
* **元数据优先，全文按需开启。** 列表 API 给出标题、机构、评级与日期；
  研报正文是另一台主机上的 PDF。``with_text`` 会逐篇抓取并解析这些 PDF，
  这也是它默认关闭的原因。
* **内容不可信。** 研报正文是第三方文本：当设置 ``with_text`` 时，
  每一篇都会像网页结果那样被围栏包裹（见 ``fencing.py``），
  使其读起来是引用材料而非指令。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, DataInput, PermissionLevel, ToolGroup
from finharness.tools.generic.fencing import (
    EXTERNAL_NOTICE,
    RESULT_CLOSE,
    RESULT_OPEN,
)

# 全文的单篇与整体结果上限。一份研报动辄数千字符；若不设上限，
# 少数几篇研报就会挤占它们本应用于提供信息的对话空间。
_MAX_TEXT_PER_REPORT = 6000
_MAX_TEXT_TOTAL = 24000


def _default_start() -> str:
    return (date.today() - timedelta(days=30)).isoformat()


def _default_end() -> str:
    return date.today().isoformat()


class ResearchReportsInput(DataInput):
    report_type: Literal["行业", "个股"] = Field(
        default="行业", description="研报类型：行业（默认）或 个股"
    )
    industry: str | None = Field(
        default=None,
        description="行业筛选，行业名（如 '证券Ⅱ'）或行业代码；仅行业研报适用",
    )
    institution: str | None = Field(
        default=None, description="机构筛选，如 '开源证券'（支持简称/全称）或机构代码"
    )
    keyword: str | None = Field(default=None, description="标题关键词，可选")
    start_date: str | None = Field(
        default_factory=_default_start, description="起始日期 YYYY-MM-DD，默认近 30 天"
    )
    end_date: str | None = Field(
        default_factory=_default_end, description="结束日期 YYYY-MM-DD，默认今天"
    )
    top_n: int = Field(default=10, description="返回份数（1-100）")
    with_text: bool = Field(
        default=False, description="是否逐篇抓取 PDF 全文（较慢，默认只返回元数据）"
    )


class GetResearchReportsTool(BaseTool):
    name = "get_research_reports"
    description = (
        "抓取东方财富研报（行业/个股），可按行业、机构、时间段与标题关键词筛选，"
        "返回标题、机构、评级、日期与链接，可选抓取 PDF 全文。"
    )
    input_model = ResearchReportsInput
    permission = PermissionLevel.READ
    group = ToolGroup.FIN_DATA
    # 全文模式意味着每篇研报都要下载一个 PDF，因此预算给得宽松。
    timeout = 120
    # 研报正文是从网络抓取的第三方文本，这与网页工具选择退出审查的原因相同：
    # 审查者的职责是把报告与会话自身的数据对照，而不是引入新的外部文本。
    review_eligible = False
    output_schema_note = "返回研报列表；with_text=true 时含正文（PDF 抽取）。"

    async def _dispatch(
        self,
        *,
        report_type: Literal["行业", "个股"] = "行业",
        industry: str | None = None,
        institution: str | None = None,
        keyword: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        with_text: bool = False,
    ) -> RawData:
        if not 1 <= int(top_n) <= 100:
            raise ValueError("top_n 需在 1-100 之间")
        return await self.data.research_reports(
            report_type=report_type,
            industry=industry,
            institution=institution,
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
            with_text=with_text,
        )

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """元数据以表格呈现；带全文时，每篇研报像网页那样被围栏包裹。"""
        df = raw.df
        if df is None or not len(df):
            return ("（未找到符合条件的研报）"), [raw]

        has_text = "content" in df.columns and df["content"].astype(bool).any()
        if not has_text:
            header, body = self._metadata_view(df, raw)
            return body + header, [raw]

        return self._full_text_view(df, raw), [raw]

    # -- 视图 ------------------------------------------------------------------
    def _metadata_view(self, df, raw: RawData) -> tuple[str, str]:
        """渲染只含元数据的表格视图（去掉正文列），并附带完整性说明。"""
        display = df.drop(columns=["content"], errors="ignore")
        note = self._completeness_note(df, raw)
        table = self.trim_dataframe(
            display, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        head = f"共 {len(df)} 篇研报。\n\n" if len(df) else ""
        return (head, table + ("\n" + note if note else ""))

    def _full_text_view(self, df, raw: RawData) -> str:
        """逐篇渲染研报全文（含围栏与来源），并受整体字数上限约束。"""
        lines = [EXTERNAL_NOTICE, ""]
        budget = _MAX_TEXT_TOTAL
        for index, row in enumerate(df.itertuples(index=False), start=1):
            lines.append(RESULT_OPEN.format(index=index, url=row.pdf_url or row.detail_url))
            lines.append(f"标题：{row.title}")
            lines.append(f"机构：{row.institution}｜日期：{row.publish_date}｜评级：{row.rating}")
            if row.industry:
                lines.append(f"行业：{row.industry}")
            content = str(row.content or "")
            if content:
                if budget <= 0:
                    lines.append("（正文已省略：本次返回总字数已达上限）")
                else:
                    clipped = content[: min(_MAX_TEXT_PER_REPORT, budget)]
                    if len(clipped) < len(content):
                        clipped += "…（正文已截断）"
                    budget -= len(clipped)
                    lines.append(clipped)
            else:
                lines.append("（无正文）")
            lines.append(RESULT_CLOSE)
            lines.append("")
        note = self._completeness_note(df, raw)
        if note:
            lines.append(note)
        return "\n".join(lines).rstrip()

    @staticmethod
    def _completeness_note(df, raw: RawData) -> str:
        """说明本次请求未能完整兑现之处，而不是暗示已经完整。

        这里能观察到两类缺口：返回行数少于请求数量；以及按*名称*给出的行业
        （API 无法据此过滤，只能在客户端于有限页数内匹配）。
        """
        parts: list[str] = []
        requested = raw.params.get("top_n")
        if requested and len(df) < int(requested):
            parts.append(f"仅匹配到 {len(df)} 篇（请求 {requested} 篇）")
        industry = raw.params.get("industry")
        if industry and not str(industry).isdigit() and raw.params.get("report_type") == "行业":
            parts.append(
                f"行业「{industry}」按名称客户端匹配，可能不全；如需精确请改用行业代码"
            )
        if not parts:
            return ""
        return "（" + "；".join(parts) + "。）"
