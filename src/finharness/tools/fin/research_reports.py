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
  研报正文是另一台主机上的 PDF。``with_text`` 会逐篇抓取这些 PDF 并**落盘**，
  这也是它默认关闭的原因。
* **句柄优先于正文。** 一份研报动辄数十页，远超单条工具结果的 token 预算，因此
  这里返回的是"每篇一行句柄（含本地路径）+ 首页预览"，而不是正文本身。  要摘要用
  ``summarize_document`` 拿分片索引再按片 ``spawn_agent``，要细节用 ``read_pdf`` 按页读取——两者都基于同一个
  落盘文件，无需重新抓取。
* **内容不可信。** 研报正文是第三方文本：预览与分页读取都会像网页结果那样被
  围栏包裹（见 ``fencing.py``），使其读起来是引用材料而非指令。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from finharness.data.raw import RawData
from finharness.shared.declaration import Capability, Tier, ToolGroup, param, tool
from finharness.shared.fencing import (
    EXTERNAL_NOTICE,
    RESULT_CLOSE,
    fence,
    neutralize,
)
from finharness.tools.base import BaseTool

# 单条结果的 token 预算。研报是"一次抓取、多篇句柄"，比常规数据结果需要更多空间
# 才能列出全部句柄与预览，因此显式声明高于全局默认值。
_RESULT_TOKENS = 4000


def _default_start() -> str:
    return (date.today() - timedelta(days=30)).isoformat()


def _default_end() -> str:
    return date.today().isoformat()


@tool(
    name="get_research_reports",
    description=(
        "抓取东方财富研报（行业/个股），可按行业、机构、时间段与标题关键词筛选，"
        "返回标题、机构、评级、日期与链接，可选抓取 PDF 全文。"
    ),
    capability=Capability.RESEARCH_REPORT,
    # 抓取与全文下载都较重，而多数问题用本地数据即可回答，故按需注入。
    tier=Tier.LAZY,
    group=ToolGroup.FIN_DATA,
    # 全文模式意味着每篇研报都要下载一个 PDF，因此预算给得宽松。
    timeout=120,
    result_tokens=_RESULT_TOKENS,
    # 研报正文是从网络抓取的第三方文本，这与网页工具选择退出审查的原因相同：
    # 审查者的职责是把报告与会话自身的数据对照，而不是引入新的外部文本。
    review_eligible=False,
    data_tool=True,
    output_schema_note=(
        "返回研报列表；with_text=true 时每篇另含本地 PDF 路径与首页预览"
        "（先 summarize_document 拿分片索引再 spawn_agent，精读用 read_pdf）。"
    ),
)
class GetResearchReportsTool(BaseTool):
    @param("report_type", desc="研报类型：行业（默认）或 个股")
    @param("industry", desc="行业筛选，行业名（如 '证券Ⅱ'）或行业代码；仅行业研报适用")
    @param("institution", desc="机构筛选，如 '开源证券'（支持简称/全称）或机构代码")
    @param("keyword", desc="标题关键词，可选")
    @param("start_date", desc="起始日期 YYYY-MM-DD，默认近 30 天", default_factory=_default_start)
    @param("end_date", desc="结束日期 YYYY-MM-DD，默认今天", default_factory=_default_end)
    @param("top_n", desc="返回份数（1-100）")
    @param("with_text", desc="是否逐篇抓取 PDF 全文（较慢，默认只返回元数据）")
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
        """元数据以表格呈现；带全文时，先列句柄再给围栏预览。"""
        df = raw.df
        if df is None or not len(df):
            return ("（未找到符合条件的研报）"), [raw]

        # 落地路径与预览任一存在即视为"已抓全文"：某篇抓取失败时 content 是说明性
        # 文案而非空串，此时仍应走全文视图，如实呈现该篇的缺口。
        has_path = "pdf_path" in df.columns and df["pdf_path"].astype(bool).any()
        has_text = "content" in df.columns and df["content"].astype(bool).any()
        if not (has_path or has_text):
            header, body = self._metadata_view(df, raw)
            return body + header, [raw]

        return self._full_text_view(df, raw), [raw]

    # -- 视图 ------------------------------------------------------------------
    def _metadata_view(self, df, raw: RawData) -> tuple[str, str]:
        """渲染只含元数据的表格视图（去掉正文与内部路径列），并附带完整性说明。"""
        display = df.drop(columns=["content", "pdf_path"], errors="ignore")
        note = self._completeness_note(df, raw)
        table = self.trim_dataframe(
            display, source_path=raw.parquet_path, detail=self._render_detail(raw)
        )
        head = f"共 {len(df)} 篇研报。\n\n" if len(df) else ""
        return (head, table + ("\n" + note if note else ""))

    def _full_text_view(self, df, raw: RawData) -> str:
        """渲染"句柄 + 预览"视图。

        行文顺序是刻意的：**先给出全部句柄，再给预览**。过去这里把正文逐篇内联，
        于是引擎的单条结果预算在几页之内就把内容砍断——第 2 篇起的元数据与路径
        一并消失（head-only 截断），而它们的 PDF 早已下载完毕。先列句柄意味着即使
        仍超出预算，读者也拿到全部路径，可以按需精读或摘要，而不会"只见到首页且
        不知道丢了什么"。
        """
        lines = [EXTERNAL_NOTICE, ""]
        lines.append(f"共 {len(df)} 篇研报（正文已落盘，先 summarize_document 拿分片索引再 spawn_agent，精读用 read_pdf）。")
        lines.append("")

        lines.append("### 研报清单")
        for index, row in enumerate(df.itertuples(index=False), start=1):
            lines.append(
                f"{index}. {neutralize(row.title)}｜{neutralize(row.institution)}｜{row.publish_date}"
                f"｜评级：{row.rating}｜共 {row.pdf_pages or '?'} 页"
            )
            lines.append(f"   路径：{row.pdf_path or '（未落盘）'}")
        lines.append("")

        # 预览放最后，因此它被预算裁掉时，上面的清单仍然完整。
        lines.append("### 首页预览")
        for index, row in enumerate(df.itertuples(index=False), start=1):
            content = str(row.content or "")
            lines.append("")
            lines.append(
                fence(index, row.pdf_path or row.pdf_url or row.detail_url)
            )
            lines.append(f"标题：{neutralize(row.title)}")
            if content:
                lines.append(neutralize(content))
            else:
                lines.append("（无预览）")
            lines.append(RESULT_CLOSE)

        note = self._completeness_note(df, raw)
        if note:
            lines.append("")
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
