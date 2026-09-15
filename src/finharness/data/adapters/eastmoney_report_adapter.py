"""东方财富研报适配器：列表元数据，外加可选的全文（docs 03.4）。

研报列表是一个公开的 JSONP 接口，无需密钥；本适配器真正值得关注的部分，是
调用方所求与 API 所能过滤的两者之间的两处不匹配，因为掩盖其中任何一处都会
悄悄返回错误结果：

* **行业没有名称查询。** API 仅按 ``industryCode`` 过滤，且不提供代码表。
  因此给定名称时，我们会在行业数据流中翻页并在客户端匹配 ``industryName``，
  且设有页数上限——并会报告我们做了截断，而不是暗示我们搜索了全部内容。
* **机构确实有查询。** ``/report/org`` 列出了每家券商及其 ``orgCode``，因此
  名称会被解析为代码，然后在服务端过滤。

研报的 *正文* 是 PDF；摘要根本不在列表 API 中。因此全文是可选启用的
（``with_text``）并逐份报告抓取，这也是默认关闭它的原因。
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Callable

import httpx
import pandas as pd

from finharness.data.adapters.base import AdapterError, DataAdapter, FetchResult
from finharness.data.adapters.pdf_fetch import fetch_pdf_text
from finharness.data.adapters.tavily_adapter import system_proxy

REPORTS_INTERFACE = "reports"

LIST_URL = "https://reportapi.eastmoney.com/report/list"
ORG_URL = "https://reportapi.eastmoney.com/report/org"
PDF_URL_TEMPLATE = "https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"
DETAIL_URL_TEMPLATES = {
    "个股": "https://data.eastmoney.com/report/info/{info_code}.html",
    "行业": "https://data.eastmoney.com/report/zw_industry.jshtml?infocode={info_code}",
}

# ``qType`` 选择研报族别。0=个股、1=行业 是列表接口提供的两种；2/3/4
# （策略/宏观/晨会）完全位于另一个接口上。
_QTYPE = {"个股": 0, "行业": 1}

# 无论请求多少，该 API 的 pageSize 上限都是 100（已实测验证）。
_MAX_PAGE_SIZE = 100
# 客户端的行业/名称扫描在放弃前会遍历的列表页数。每页即一次请求；若没有上限，
# 一个匹配不到的名称将会把整个语料翻遍。
_MAX_SCAN_PAGES = 5

# ratingChange 以一个小整数形式返回；该映射来自站点自身。
_RATING_CHANGE = {"0": "调高", "1": "调低", "2": "首次", "3": "维持", "4": "无"}

_COLUMNS = (
    "title",
    "stock_name",
    "industry",
    "institution",
    "rating",
    "rating_change",
    "publish_date",
    "pdf_pages",
    "detail_url",
    "pdf_url",
    "content",
)

_JSONP_RE = re.compile(r"^\s*[A-Za-z_$][\w$]*\s*\((.*)\)\s*;?\s*$", re.DOTALL)

# 注入式传输层：给定完整 URL，返回响应体文本。
HttpGet = Callable[[str], str]


def _parse_jsonp(text: str) -> dict[str, Any]:
    """拆开 ``cb({...})`` 包装，并解码其中的 JSON 对象。"""
    match = _JSONP_RE.match(text)
    payload = match.group(1) if match else text
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise AdapterError("研报接口返回了非 JSON 响应") from exc
    if not isinstance(data, dict):
        raise AdapterError("研报接口返回了非预期的数据结构")
    return data


def _authors(item: dict[str, Any]) -> str:
    """API 返回的 ``author`` 形如 ``["<id>.<name>", ...]``；这里保留其中的名称。"""
    raw = item.get("author")
    if not isinstance(raw, list):
        return ""
    names = []
    for entry in raw:
        text = str(entry)
        names.append(text.split(".", 1)[1] if "." in text else text)
    return "、".join(name for name in names if name)


def _detail_url(report_type: str, info_code: str) -> str:
    template = DETAIL_URL_TEMPLATES.get(report_type) or DETAIL_URL_TEMPLATES["个股"]
    return template.format(info_code=info_code)


def _pdf_url(info_code: str) -> str:
    return PDF_URL_TEMPLATE.format(info_code=info_code)


def _row(item: dict[str, Any], report_type: str) -> dict[str, Any]:
    """将 API 返回的单条记录映射为契约列名所对应的行字典。"""
    publish = str(item.get("publishDate") or "")
    # publishDate 形如 "YYYY-MM-DD 00:00:00.000"；仅保留日期部分用于展示。
    if " " in publish:
        publish = publish.split(" ", 1)[0]
    info_code = str(item.get("infoCode") or "")
    industry = str(item.get("industryName") or "") or str(item.get("indvInduName") or "")
    return {
        "title": str(item.get("title") or ""),
        "stock_name": str(item.get("stockName") or ""),
        "industry": industry,
        "institution": str(item.get("orgSName") or ""),
        "rating": str(item.get("emRatingName") or ""),
        "rating_change": _RATING_CHANGE.get(str(item.get("ratingChange")), ""),
        "publish_date": publish,
        "pdf_pages": item.get("attachPages") or "",
        "detail_url": _detail_url(report_type, info_code) if info_code else "",
        "pdf_url": _pdf_url(info_code) if info_code else "",
        "content": "",
    }


class EastmoneyReportAdapter(DataAdapter):
    """东方财富的研报元数据（以及可选的全文）。"""

    name = "eastmoney_report"

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        http_get: HttpGet | None = None,
        text_fetcher: Callable[[str], str] | None = None,
        with_text_allowed: bool = True,
    ) -> None:
        self.timeout_s = timeout_s
        # 可注入，以便测试在不联网的情况下验证解析逻辑。
        self._http_get = http_get or self._network_get
        self._text_fetcher = text_fetcher or (lambda url: fetch_pdf_text(url))
        # 全文需要从本主机访问文档 CDN，因此运维方可以禁止；一旦禁止，适配器会
        # 显式报错，而不是悄悄省略。
        self.with_text_allowed = with_text_allowed

    # -- 传输层 ---------------------------------------------------------------
    def _network_get(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Referer": "https://data.eastmoney.com/report/",
        }
        try:
            with httpx.Client(
                timeout=self.timeout_s, proxy=system_proxy(), follow_redirects=True
            ) as client:
                response = client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise AdapterError(f"研报接口请求失败：{exc}", retryable=True) from exc
        if response.status_code >= 400:
            raise AdapterError(
                f"研报接口返回 {response.status_code}",
                retryable=response.status_code == 429 or response.status_code >= 500,
            )
        return response.text

    # -- 查询 -----------------------------------------------------------------
    @staticmethod
    def _date_range(start_date: str | None, end_date: str | None) -> tuple[str, str]:
        """解析时间窗口，默认取最近 30 天。

        两个边界在 API 上都是必填的（省略会返回 400），所以这里的默认值不是
        便利功能——它是让一次裸调用能够成立的关键。
        """
        end = end_date or date.today().isoformat()
        start = start_date or (date.fromisoformat(end) - timedelta(days=30)).isoformat()
        return start, end

    def _list_page(
        self,
        *,
        q_type: int,
        begin: str,
        end: str,
        page_no: int,
        page_size: int,
        org_code: str | None,
        industry_code: str | None,
    ) -> dict[str, Any]:
        params = [
            "cb=x",
            f"pageSize={min(max(page_size, 1), _MAX_PAGE_SIZE)}",
            f"beginTime={begin}",
            f"endTime={end}",
            f"pageNo={page_no}",
            f"qType={q_type}",
        ]
        if org_code:
            params.append(f"orgCode={org_code}")
        if industry_code:
            params.append(f"industryCode={industry_code}")
        return _parse_jsonp(self._http_get(LIST_URL + "?" + "&".join(params)))

    def resolve_org_code(self, institution: str) -> str | None:
        """将机构名称（或直接传入的代码）映射为 ``orgCode``。"""
        if institution.isdigit():
            return institution
        data = _parse_jsonp(self._http_get(ORG_URL + "?cb=x&pageSize=100&pageNo=1"))
        wanted = institution.strip().lower()
        entries = data.get("data") or []
        # 优先精确匹配简称，其次精确匹配全称，最后才是子串匹配——这样
        # "开源证券" 不会输给一个更长的无关名称。
        for field in ("orgSName", "orgName"):
            for entry in entries:
                if str(entry.get(field) or "").strip().lower() == wanted:
                    return str(entry.get("orgCode") or "") or None
        for entry in entries:
            haystack = f"{entry.get('orgSName','')}{entry.get('orgName','')}".lower()
            if wanted and wanted in haystack:
                return str(entry.get("orgCode") or "") or None
        return None

    def fetch_research_reports(
        self,
        report_type: str = "行业",
        industry: str | None = None,
        institution: str | None = None,
        keyword: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        with_text: bool = False,
    ) -> FetchResult:
        q_type = _QTYPE.get(report_type)
        if q_type is None:
            raise ValueError(f"不支持的研报类型：{report_type}（仅支持 行业／个股）")
        top_n = max(1, min(int(top_n), _MAX_PAGE_SIZE))
        begin, end = self._date_range(start_date, end_date)

        org_code = None
        if institution:
            org_code = self.resolve_org_code(institution)
            if org_code is None:
                raise AdapterError(f"未找到机构：{institution}")

        # 纯数字的行业已经是代码；名称则必须在客户端与行业数据流进行匹配，可能
        # 需要翻多页。
        industry_code = None
        industry_name = None
        scan_by_name = False
        if industry:
            if industry.isdigit():
                industry_code = industry
            else:
                industry_name = industry
                scan_by_name = True

        rows = self._collect(
            q_type=q_type,
            begin=begin,
            end=end,
            top_n=top_n,
            org_code=org_code,
            industry_code=industry_code,
            industry_name=industry_name,
            scan_by_name=scan_by_name,
            keyword=keyword,
        )

        frame = pd.DataFrame(rows, columns=list(_COLUMNS))
        if with_text and len(frame):
            if not self.with_text_allowed:
                raise AdapterError("研报全文抓取已被配置禁用（search.local_pdf_fallback=false）")
            frame["content"] = [self._report_text(row["pdf_url"]) for row in rows]
        return FetchResult(df=frame, interface=REPORTS_INTERFACE)

    def _collect(
        self,
        *,
        q_type: int,
        begin: str,
        end: str,
        top_n: int,
        org_code: str | None,
        industry_code: str | None,
        industry_name: str | None,
        scan_by_name: bool,
        keyword: str | None,
    ) -> list[dict[str, Any]]:
        """收集数据行，仅当需要客户端名称匹配时才翻页。"""
        collected: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._list_page(
                q_type=q_type,
                begin=begin,
                end=end,
                page_no=page,
                page_size=_MAX_PAGE_SIZE,
                org_code=org_code,
                # 按代码在服务端过滤既精确又廉价。
                industry_code=industry_code,
            )
            items = data.get("data") or []
            if not items:
                break

            for item in items:
                row = _row(item, "行业" if q_type == 1 else "个股")
                if industry_name and industry_name not in row["industry"]:
                    continue
                if keyword and keyword not in row["title"]:
                    continue
                collected.append(row)
                if len(collected) >= top_n:
                    return collected

            # 只有名称匹配这条路径需要更多页；其他所有过滤都在服务端完成，所以
            # 第一页就已经包含答案。
            if not scan_by_name or page >= _MAX_SCAN_PAGES:
                break
            total_page = data.get("TotalPage") or 0
            if page >= total_page:
                break
            page += 1
        return collected

    def _report_text(self, pdf_url: str) -> str:
        try:
            return self._text_fetcher(pdf_url)
        except AdapterError as exc:
            # 一份不可读的 PDF 不应让整个列表丢失；元数据仍然有用，缺口会直接
            # 在单元格中说明。
            return f"（正文获取失败：{exc}）"
