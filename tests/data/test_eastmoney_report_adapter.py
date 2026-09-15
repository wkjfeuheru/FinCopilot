"""东方财富研报 adapter：请求构造、过滤器、JSONP 解析、全文。

离线运行：HTTP transport 和 PDF fetcher 都是注入的，因此这里不会触及网络。
关键属性是两个过滤机制（服务端的 code 与客户端的 name），以及
部分结果路径的如实性。
"""

from __future__ import annotations

import json

import pytest

from finharness.data.adapters.base import AdapterError
from finharness.data.adapters.eastmoney_report_adapter import (
    EastmoneyReportAdapter,
    _authors,
    _parse_jsonp,
    _row,
)


def jsonp(payload: dict) -> str:
    return "x(" + json.dumps(payload, ensure_ascii=False) + ")"


def item(**overrides) -> dict:
    base = {
        "title": "非银金融行业周报：看好非银机会",
        "stockName": "",
        "stockCode": "",
        "orgCode": "80000162",
        "orgName": "开源证券股份有限公司",
        "orgSName": "开源证券",
        "publishDate": "2026-09-13 00:00:00.000",
        "infoCode": "AP202609131829339640",
        "industryCode": "473",
        "industryName": "证券Ⅱ",
        "emRatingName": "持有",
        "ratingChange": 3,
        "attachPages": 4,
        "author": ["11000250533.高超"],
    }
    base.update(overrides)
    return base


def adapter_returning(pages: list[dict], *, orgs: dict | None = None, **kwargs):
    """一个 adapter，其 list endpoint 从 ``pages`` 给出应答（下标从 1 开始）。"""
    calls: list[str] = []

    def http_get(url: str) -> str:
        calls.append(url)
        if "/report/org" in url:
            return jsonp({"hits": len(orgs or {}), "data": orgs or []})
        page_no = int(url.split("pageNo=")[1].split("&")[0])
        payload = pages[page_no - 1] if page_no - 1 < len(pages) else {"hits": 0, "data": []}
        return jsonp(payload)

    adapter = EastmoneyReportAdapter(http_get=http_get, **kwargs)
    adapter._test_calls = calls  # type: ignore[attr-defined]
    return adapter


# -- 解析 ------------------------------------------------------------------


def test_jsonp_is_unwrapped():
    assert _parse_jsonp('x({"hits": 1, "data": []})') == {"hits": 1, "data": []}


def test_a_non_json_body_is_an_adapter_error():
    with pytest.raises(AdapterError):
        _parse_jsonp("<html>not json</html>")


def test_authors_keep_names_and_drop_ids():
    assert _authors({"author": ["11000250533.高超", "11000402431.卢崑"]}) == "高超、卢崑"
    assert _authors({"author": "not-a-list"}) == ""
    assert _authors({}) == ""


def test_row_builds_display_fields_and_links():
    row = _row(item(), "行业")

    assert row["institution"] == "开源证券"
    assert row["industry"] == "证券Ⅱ"
    assert row["rating"] == "持有"
    assert row["rating_change"] == "维持"
    # API 发送的是 datetime；对读者而言只有日期有用。
    assert row["publish_date"] == "2026-09-13"
    assert row["pdf_url"] == "https://pdf.dfcfw.com/pdf/H3_AP202609131829339640_1.pdf"
    assert "zw_industry" in row["detail_url"]


def test_rating_change_codes_map_to_words():
    assert _row(item(ratingChange=0), "行业")["rating_change"] == "调高"
    assert _row(item(ratingChange=1), "行业")["rating_change"] == "调低"
    assert _row(item(ratingChange=2), "行业")["rating_change"] == "首次"
    assert _row(item(ratingChange=99), "行业")["rating_change"] == ""


# -- 请求构造 ----------------------------------------------------------


def test_qtype_follows_the_report_type():
    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(report_type="行业", top_n=1)
    assert "qType=1" in adapter._test_calls[0]

    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(report_type="个股", top_n=1)
    assert "qType=0" in adapter._test_calls[0]


def test_an_unknown_report_type_is_refused():
    adapter = adapter_returning([{"hits": 1, "data": []}])
    with pytest.raises(ValueError):
        adapter.fetch_research_reports(report_type="策略")


def test_dates_are_always_sent_even_when_caller_omits_them():
    """缺少两个时间边界时 API 会返回 400，因此 adapter 绝不能省略它们。"""
    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(top_n=1)

    url = adapter._test_calls[0]
    assert "beginTime=" in url and "endTime=" in url


def test_explicit_dates_are_passed_through():
    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(start_date="2026-01-01", end_date="2026-02-01", top_n=1)

    url = adapter._test_calls[0]
    assert "beginTime=2026-01-01" in url
    assert "endTime=2026-02-01" in url


def test_page_size_is_capped_at_the_api_limit():
    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(top_n=100, industry="473")
    assert "pageSize=100" in adapter._test_calls[0]


# -- 机构过滤（服务端，通过名称查找） ------------------------


def test_institution_name_is_resolved_to_a_code():
    orgs = [{"orgName": "开源证券股份有限公司", "orgCode": "80000162", "orgSName": "开源证券"}]
    adapter = adapter_returning([{"hits": 1, "data": [item()]}], orgs=orgs)
    adapter.fetch_research_reports(institution="开源证券", top_n=1)

    assert any("/report/org" in url for url in adapter._test_calls)
    list_url = next(url for url in adapter._test_calls if "/report/list" in url)
    assert "orgCode=80000162" in list_url


def test_an_unknown_institution_is_reported():
    adapter = adapter_returning([{"hits": 1, "data": [item()]}], orgs=[])
    with pytest.raises(AdapterError) as exc:
        adapter.fetch_research_reports(institution="不存在的券商")
    assert "未找到机构" in str(exc.value)


def test_a_numeric_institution_is_used_directly():
    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(institution="80000162", top_n=1)

    # 当已提供 code 时无需进行 org 查找。
    assert not any("/report/org" in url for url in adapter._test_calls)
    assert "orgCode=80000162" in adapter._test_calls[0]


# -- 行业过滤（code 走服务端，name 走客户端） ---------------------


def test_a_numeric_industry_is_filtered_server_side():
    adapter = adapter_returning([{"hits": 1, "data": [item()]}])
    adapter.fetch_research_reports(industry="473", top_n=1)

    assert "industryCode=473" in adapter._test_calls[0]
    # 一页就够了：服务端已经过滤过了。
    assert len([u for u in adapter._test_calls if "/report/list" in u]) == 1


def test_an_industry_name_is_matched_client_side():
    page = {"hits": 2, "TotalPage": 1, "data": [item(industryName="证券Ⅱ"), item(industryName="养殖业")]}
    adapter = adapter_returning([page])
    result = adapter.fetch_research_reports(industry="养殖业", top_n=5)

    assert len(result.df) == 1
    assert result.df.iloc[0]["industry"] == "养殖业"


def test_industry_name_scan_is_paged_and_bounded():
    """一个始终不匹配的 name 必须在达到页数上限后停止，而不是无限翻页。"""
    pages = [{"hits": 100, "TotalPage": 100, "data": [item(industryName="别的行业")]} for _ in range(20)]
    adapter = adapter_returning(pages)
    result = adapter.fetch_research_reports(industry="找不到的行业", top_n=5)

    assert len(result.df) == 0
    list_calls = [u for u in adapter._test_calls if "/report/list" in u]
    assert len(list_calls) <= 5


def test_keyword_filters_titles_client_side():
    page = {"hits": 2, "TotalPage": 1, "data": [item(title="周报：白酒动销"), item(title="周报：啤酒")]}
    adapter = adapter_returning([page])
    result = adapter.fetch_research_reports(keyword="白酒", top_n=5)

    assert len(result.df) == 1
    assert "白酒" in result.df.iloc[0]["title"]


# -- 结果形态 -------------------------------------------------------------


def test_top_n_limits_the_rows_returned():
    page = {"hits": 5, "TotalPage": 1, "data": [item(title=f"报告{i}") for i in range(5)]}
    adapter = adapter_returning([page])
    result = adapter.fetch_research_reports(top_n=3)

    assert len(result.df) == 3
    assert result.interface == "reports"


def test_an_empty_result_is_an_empty_frame_not_an_error():
    adapter = adapter_returning([{"hits": 0, "data": []}])
    result = adapter.fetch_research_reports(top_n=5)

    assert len(result.df) == 0
    assert "title" in result.df.columns


# -- 全文 ----------------------------------------------------------------


def test_with_text_fills_the_content_column():
    page = {"hits": 1, "TotalPage": 1, "data": [item()]}
    adapter = adapter_returning(
        [page], text_fetcher=lambda url: "这是研报正文"
    )
    result = adapter.fetch_research_reports(top_n=1, with_text=True)

    assert result.df.iloc[0]["content"] == "这是研报正文"


def test_one_unreadable_pdf_does_not_lose_the_metadata():
    """正文抓取失败时只能降级该单元格，而不是整个列表。"""
    page = {"hits": 2, "TotalPage": 1, "data": [item(title="A"), item(title="B")]}

    def flaky(url: str) -> str:
        if "hits" in url:  # pragma: no cover - 防止误用
            return ""
        raise AdapterError("反爬校验未通过")

    adapter = adapter_returning([page], text_fetcher=flaky)
    result = adapter.fetch_research_reports(top_n=2, with_text=True)

    assert len(result.df) == 2
    assert "正文获取失败" in result.df.iloc[0]["content"]


def test_with_text_is_refused_when_the_operator_disabled_local_fetch():
    page = {"hits": 1, "TotalPage": 1, "data": [item()]}
    adapter = adapter_returning(
        [page], text_fetcher=lambda url: "正文", with_text_allowed=False
    )

    with pytest.raises(AdapterError) as exc:
        adapter.fetch_research_reports(top_n=1, with_text=True)

    assert "禁用" in str(exc.value)
