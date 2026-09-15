"""网页检索（docs 03.4）。

作为 A 股数据源的外部数据补充：当问题需要本地适配器未覆盖的内容——一份政策文件、
一则新闻事件、公司自己的网站——这里通过配置的搜索 provider 触达公开网络。

两个设计要点承担关键作用：

* **不可信内容。** 返回的一切都是第三方撰写的文本，会直接进入模型上下文。这里不做
  注入指令扫描；而是为每条结果加上围栏与标签（见 ``fencing.py``），并由系统提示声明
  被围栏的内容是参考资料而非命令。
* **契约上的懒加载。** 检索是懒加载工具，因此模型必须先 ``load_tool`` 才能调用它。
  系统提示会点名它，使发现过程不依赖于模型主动先去检索。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool, PermissionLevel, ToolGroup
from finharness.tools.generic.fencing import (
    EXTERNAL_NOTICE,
    RESULT_CLOSE,
    RESULT_OPEN,
)


class WebSearchInput(BaseModel):
    query: str = Field(description="检索词，如 '2026年 白酒消费税 政策'")
    top_n: int = Field(default=5, description="返回条数（1-20）")
    topic: Literal["general", "news"] | None = Field(
        default=None,
        description="检索领域：general（通用，默认）或 news（时效新闻），可选",
    )
    time_range: str | None = Field(
        default=None, description="时间范围：day / week / month / year，可选"
    )


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "联网搜索公开网页信息，返回标题、网址与摘要，用于补充本地数据源未覆盖的"
        "政策、新闻与行业信息。结果为第三方内容，仅作事实参考。"
    )
    input_model = WebSearchInput
    permission = PermissionLevel.READ
    group = ToolGroup.GENERIC
    timeout = 30
    # 复核是拿报告与会话自身的数据做核对；搜索网页会消耗 token 并把不可信文本
    # 拉入复核者。
    review_eligible = False
    output_schema_note = "返回每条结果的标题/网址/摘要。"

    async def _dispatch(
        self,
        *,
        query: str,
        top_n: int = 5,
        topic: Literal["general", "news"] | None = None,
        time_range: str | None = None,
    ) -> RawData:
        """调用数据层执行联网检索；无结果时返回带外部内容提示的文本载荷。"""
        raw = await self.data.web_search(
            query, top_n=top_n, topic=topic, time_range=time_range
        )
        if raw.df is None or not len(raw.df):
            return RawData(
                kind="text",
                text=f"{EXTERNAL_NOTICE}\n\n（未检索到与「{query}」相关的网页结果）",
                endpoint=raw.endpoint,
                params=dict(raw.params),
                data_date=raw.data_date,
                from_cache=raw.from_cache,
                cache_key=raw.cache_key,
                parquet_path=raw.parquet_path,
            )
        return raw

    def render(self, raw: RawData) -> tuple[str, list[RawData]]:
        """为每条结果加围栏，使引用的网页文本不会被读成指令。"""
        if raw.df is None or not len(raw.df):
            return (raw.text or "（无结果）"), [raw]

        lines = [EXTERNAL_NOTICE, ""]
        for index, row in enumerate(raw.df.itertuples(index=False), start=1):
            lines.append(RESULT_OPEN.format(index=index, url=row.url))
            lines.append(f"标题：{row.title}")
            lines.append(str(row.content))
            lines.append(RESULT_CLOSE)
            lines.append("")
        return "\n".join(lines).rstrip(), [raw]
