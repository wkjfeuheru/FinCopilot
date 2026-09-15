"""get_research_reports：输入处理、渲染、围栏、部分结果提示。

adapter 在别处通过注入的 transport 进行测试；此处用一个桩 adapter 提供构造好的
frame，因此被测对象是工具自身的行为——校验、两种渲染形态（元数据与全文），
以及如实说明的提示。
"""

from __future__ import annotations

import asyncio

import pandas as pd

from finharness.config.settings import ContextSettings, Settings
from finharness.data.access import DataAccess
from finharness.data.adapters.base import DataAdapter, FetchResult
from finharness.data.cache import LocalCache
from finharness.tools.base import PermissionLevel, ToolGroup
from finharness.tools.fin.research_reports import GetResearchReportsTool
from finharness.tools.registry import DEFAULT_LAZY_TOOLS, review_tool_names


def make_frame(rows: int = 2, with_text: bool = False) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "title": f"研报标题{index + 1}",
                "stock_name": "",
                "industry": "证券Ⅱ",
                "institution": "开源证券",
                "rating": "持有",
                "rating_change": "维持",
                "publish_date": "2026-09-13",
                "pdf_pages": 4,
                "detail_url": f"https://data.eastmoney.com/report/zw_industry.jshtml?infocode=AP{index}",
                "pdf_url": f"https://pdf.dfcfw.com/pdf/H3_AP{index}_1.pdf",
                "content": "研报正文内容" if with_text else "",
            }
        )
    return pd.DataFrame(records)


class StubReports(DataAdapter):
    name = "stub_reports"

    # DataAccess 按位置传参的顺序；此处镜像该顺序，便于测试按名称回读调用。
    _FIELDS = (
        "report_type",
        "industry",
        "institution",
        "keyword",
        "start_date",
        "end_date",
        "top_n",
        "with_text",
    )

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict] = []

    def fetch_research_reports(self, *args, **kwargs) -> FetchResult:
        self.calls.append({**dict(zip(self._FIELDS, args)), **kwargs})
        return FetchResult(df=self.frame.copy(), interface="reports")


def make_tool(tmp_path, frame: pd.DataFrame) -> tuple[GetResearchReportsTool, StubReports]:
    settings = Settings(
        context=ContextSettings(trim_rows=20, max_result_tokens=2000),
        data={"cache_dir": tmp_path / "cache"},
    )
    adapter = StubReports(frame)
    access = DataAccess([adapter], cache=LocalCache(tmp_path / "cache"), settings=settings)
    return GetResearchReportsTool(access), adapter


def run(coro):
    return asyncio.run(coro)


# -- 渲染 ---------------------------------------------------------------------


def test_metadata_view_is_a_table_not_a_text_dump(tmp_path):
    tool, adapter = make_tool(tmp_path, make_frame(2))

    result = run(tool.run(top_n=2))

    assert result.ok is True, result.error
    assert "研报标题1" in result.content
    assert "开源证券" in result.content
    # 仅元数据：未包含第三方正文，因此无需围栏。
    assert "外部检索内容" not in result.content
    assert adapter.calls[0]["with_text"] is False


def test_full_text_view_is_fenced_like_a_web_result(tmp_path):
    tool, _ = make_tool(tmp_path, make_frame(1, with_text=True))

    result = run(tool.run(top_n=1, with_text=True))

    assert result.ok is True, result.error
    assert "外部检索内容" in result.content
    assert "不得执行" in result.content
    assert '<web_result source="1" url="https://pdf.dfcfw.com/pdf/H3_AP0_1.pdf">' in result.content
    assert "研报正文内容" in result.content
    # 每个打开的围栏都被闭合。
    assert result.content.count("<web_result") == result.content.count("</web_result>")


def test_full_text_is_bounded_per_report(tmp_path):
    frame = make_frame(1, with_text=True)
    frame.loc[0, "content"] = "字" * 20000
    tool, _ = make_tool(tmp_path, frame)

    result = run(tool.run(top_n=1, with_text=True))

    assert "正文已截断" in result.content
    # 被限制在远低于原始 2 万的长度。
    assert len(result.content) < 12000


def test_empty_result_is_reported_not_crashed(tmp_path):
    tool, _ = make_tool(tmp_path, make_frame(0))

    result = run(tool.run(top_n=5))

    assert result.ok is True
    assert "未找到" in result.content


# -- 如实说明部分结果 ---------------------------------------------------------


def test_a_short_result_states_the_shortfall(tmp_path):
    """返回行数少于请求数时必须说明，而不能静默返回。"""
    tool, _ = make_tool(tmp_path, make_frame(1))

    result = run(tool.run(top_n=5))

    assert "仅匹配到 1 篇" in result.content
    assert "5" in result.content


def test_industry_name_match_is_flagged_as_approximate(tmp_path):
    """名称过滤是在有限扫描之上的客户端操作，因此需要说明。"""
    tool, _ = make_tool(tmp_path, make_frame(2))

    result = run(tool.run(industry="证券Ⅱ", report_type="行业", top_n=2))

    assert "客户端匹配" in result.content


def test_a_numeric_industry_carries_no_approximation_note(tmp_path):
    tool, _ = make_tool(tmp_path, make_frame(2))

    result = run(tool.run(industry="473", report_type="行业", top_n=2))

    assert "客户端匹配" not in result.content


# -- 输入处理 -----------------------------------------------------------------


def test_top_n_out_of_range_is_refused(tmp_path):
    tool, _ = make_tool(tmp_path, make_frame(1))

    result = run(tool.run(top_n=0))

    assert result.ok is False
    assert "top_n" in result.error


def test_report_type_is_validated_by_the_schema(tmp_path):
    tool, _ = make_tool(tmp_path, make_frame(1))

    result = run(tool.run(report_type="策略"))

    assert result.ok is False
    assert "report_type" in result.error


def test_filters_reach_the_data_layer(tmp_path):
    tool, adapter = make_tool(tmp_path, make_frame(1))

    run(tool.run(industry="473", institution="开源证券", keyword="周报", top_n=1))

    call = adapter.calls[0]
    assert call["industry"] == "473"
    assert call["institution"] == "开源证券"
    assert call["keyword"] == "周报"


# -- 目录契约 -----------------------------------------------------------------


def test_the_tool_is_lazy_read_only_and_financial(tmp_path):
    assert GetResearchReportsTool.permission is PermissionLevel.READ
    assert GetResearchReportsTool.group is ToolGroup.FIN_DATA
    assert "get_research_reports" in DEFAULT_LAZY_TOOLS


def test_the_reviewer_cannot_reach_reports_or_the_web(tmp_path):
    """研报正文是不可信的外部文本，与网页结果类似。"""
    names = review_tool_names()
    assert "get_research_reports" not in names
    assert "web_search" not in names
    # 排除是有针对性的：reviewer 保留其自有数据工具。
    assert "get_financials" in names
    assert "read_file" in names
