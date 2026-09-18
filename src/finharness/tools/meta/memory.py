"""跨对话长期记忆工具（docs 03.6.4 LTM）。

三个工具对应记忆的三种操作：检索（READ，免确认——读自己的记忆正是
回答问题的前提）、改写与遗忘（WRITE，需确认——删除/改写是破坏性操作，
必须由用户把关）。

记忆分两类，寻址 id 前缀区分：
* 情节（``ep_``）：做过什么——任务结果/关键决策/对话片段；
* 语义（``fa_``）：知道什么——事实/概念/偏好。

agent 先 ``search_memory`` 拿到 id，再 ``update_memory`` / ``forget_memory``。
"""

from __future__ import annotations

from typing import Any

from finharness.data.raw import RawData
from finharness.tools.base import BaseTool
from finharness.tools.declare import Capability, ToolGroup, param, tool

_EPISODE_KIND_LABELS = {
    "task_result": "任务结果",
    "decision": "关键决策",
    "excerpt": "对话片段",
}
_FACT_KIND_LABELS = {"fact": "事实", "concept": "概念", "preference": "偏好"}


def _memory_store(tool: BaseTool):
    """当前会话的记忆存储；无存储时返回 None（工具自行降级报错）。"""
    ctx = tool.ctx
    return getattr(ctx, "store", None) if ctx is not None else None


def _user_id(tool: BaseTool) -> str:
    ctx = tool.ctx
    return str(getattr(ctx, "user_id", "") or "") if ctx is not None else ""


def _semantic_index(tool: BaseTool) -> Any | None:
    ctx = tool.ctx
    index = getattr(ctx, "semantic_index", None) if ctx is not None else None
    if index is not None and getattr(index, "enabled", False):
        return index
    return None


def _no_store(endpoint: str, what: str, **params: Any) -> RawData:
    return RawData(
        kind="text",
        text=f"当前会话未接入持久记忆，无法{what}。",
        endpoint=endpoint,
        params=dict(params),
    )


@tool(
    name="search_memory",
    description=(
        "检索跨对话长期记忆：此前对话中的任务结果、关键决策、对话片段，"
        "以及积累的事实、概念与用户偏好。"
    ),
    capability=Capability.META,
    group=ToolGroup.META,
    tier="lazy",
    timeout=10,
)
class SearchMemoryTool(BaseTool):
    @param("query", desc="关键词/自然语言问题；配了嵌入模型时按语义相似度检索，否则做包含匹配")
    @param("subject", desc="标的代码（如 600519）；留空则不限")
    @param("kind", desc="类型过滤：episode（情节）/ fact / concept / preference；留空则全部")
    @param("conversation_id", desc="来源对话 id；留空则不限（仅对情节有效）")
    @param("limit", desc="最多返回条数（1-50）", default=10)
    async def _dispatch(
        self,
        *,
        query: str = "",
        subject: str = "",
        kind: str = "",
        conversation_id: str = "",
        limit: int = 10,
    ) -> RawData:
        store = _memory_store(self)
        user_id = _user_id(self)
        if store is None:
            return _no_store("memory:search", "检索", query=query)
        top = max(1, min(int(limit), 50))
        index = _semantic_index(self)
        lines: list[str] = []
        # 显式只要某类时全额度给它；不限类型时语义占一半、情节占一半，
        # 避免任何一类把预算吃光。
        wants_facts = kind in _FACT_KIND_LABELS
        wants_episodes = kind in {"", "episode"}
        if wants_facts:
            lines.extend(
                self._fact_lines(
                    store, index=index, user_id=user_id, query=query,
                    subject=subject, kind=kind, limit=top,
                )
            )
        elif wants_episodes:
            episode_quota = top if kind == "episode" else max(top - max(top // 2, 1), 1)
            fact_quota = top - episode_quota
            if fact_quota > 0:
                lines.extend(
                    self._fact_lines(
                        store, index=index, user_id=user_id, query=query,
                        subject=subject, kind="", limit=fact_quota,
                    )
                )
            lines.extend(
                self._episode_lines(
                    store, user_id=user_id, query=query, subject=subject,
                    conversation_id=conversation_id, limit=episode_quota,
                )
            )
        if not lines:
            return RawData(
                kind="text",
                text="没有命中任何跨对话记忆。",
                endpoint="memory:search",
                params={"query": query},
            )
        return RawData(
            kind="text",
            text="\n".join(lines),
            endpoint="memory:search",
            params={"query": query},
        )

    @staticmethod
    def _fact_lines(
        store, *, index: Any | None, user_id: str, query: str, subject: str,
        kind: str, limit: int,
    ) -> list[str]:
        fact_kind = kind if kind in _FACT_KIND_LABELS else None
        facts: list[Any] = []
        if index is not None and query.strip() and not subject:
            facts = list(
                index.recall(
                    user_id=user_id,
                    query=query,
                    limit=limit,
                    exclude_keys=set(),
                )
            )
            if fact_kind:
                facts = [item for item in facts if item.kind == fact_kind]
        if not facts:
            facts = store.list_ltm_facts(
                user_id=user_id,
                kind=fact_kind,
                subject=subject or None,
                limit=limit,
            )
            if query:
                needle = query.strip().lower()
                facts = [
                    item
                    for item in facts
                    if needle in item.statement.lower() or needle in item.key.lower()
                ]
        lines = []
        for item in facts:
            label = _FACT_KIND_LABELS.get(item.kind, item.kind)
            scope = f"{item.subject}：" if item.subject else ""
            lines.append(f"- {item.fa_uid} [{label}] {scope}{item.statement}")
        return lines[:limit]

    @staticmethod
    def _episode_lines(
        store, *, user_id: str, query: str, subject: str,
        conversation_id: str, limit: int,
    ) -> list[str]:
        episodes = store.list_ltm_episodes(
            user_id=user_id,
            subject=subject or None,
            source_conversation_id=conversation_id or None,
            limit=max(1, limit),
        )
        if query:
            needle = query.strip().lower()
            episodes = [item for item in episodes if needle in item.summary.lower()]
        lines = []
        for item in episodes:
            label = _EPISODE_KIND_LABELS.get(item.kind, item.kind)
            scope = f"{item.subject}：" if item.subject else ""
            formed = (item.source_ts or item.created_at)[:10]
            lines.append(
                f"- {item.ep_uid} [{label}] {scope}{item.summary}"
                f"（{item.source_title or '未命名对话'}，{formed}）"
            )
        return lines[:limit]


@tool(
    name="update_memory",
    description=(
        "编辑一条跨对话记忆（按 search_memory 返回的 ep_/fa_ id 寻址）。"
    ),
    capability=Capability.META,
    group=ToolGroup.META,
    permission="write",
    tier="lazy",
    timeout=10,
)
class UpdateMemoryTool(BaseTool):
    @param("memory_id", desc="要编辑的记忆 id（ep_xxxxxxxxxxxx 为情节，fa_xxxxxxxxxxxx 为事实/偏好）")
    @param("content", desc="新的记忆内容（情节为摘要，语义为表述）")
    @param("kind", desc="新的类型：情节 decision/excerpt，语义 fact/concept/preference；留空不变")
    @param("subject", desc="新的标的代码；留空不变")
    async def _dispatch(
        self,
        *,
        memory_id: str,
        content: str,
        kind: str = "",
        subject: str = "",
    ) -> RawData:
        store = _memory_store(self)
        user_id = _user_id(self)
        if store is None:
            return _no_store("memory:update", "编辑", memory_id=memory_id)
        try:
            if memory_id.startswith("fa_"):
                result = self._update_fact(
                    store, index=_semantic_index(self), user_id=user_id,
                    fa_uid=memory_id, statement=content, kind=kind, subject=subject,
                )
            else:
                result = self._update_episode(
                    store, user_id=user_id, ep_uid=memory_id,
                    summary=content, kind=kind, subject=subject,
                )
        except ValueError as exc:
            return RawData(
                kind="text",
                text=f"编辑失败：{exc}",
                endpoint="memory:update",
                params={"memory_id": memory_id},
            )
        if result is None:
            return RawData(
                kind="text",
                text=f"未找到记忆条目 {memory_id}。",
                endpoint="memory:update",
                params={"memory_id": memory_id},
            )
        return RawData(
            kind="text",
            text=f"已更新记忆 {memory_id}：{result}",
            endpoint="memory:update",
            params={"memory_id": memory_id},
        )

    @staticmethod
    def _update_fact(
        store, *, index: Any | None, user_id: str, fa_uid: str,
        statement: str, kind: str, subject: str,
    ) -> str | None:
        updated = store.update_ltm_fact(
            fa_uid, user_id=user_id, statement=statement,
            kind=kind or None, subject=subject or None,
        )
        if updated is None:
            return None
        # 表述变了，旧向量随之失效（存储层已清空）；有索引时立即重算，
        # 否则该条目要到下次蒸馏回填才重新可召回。
        if index is not None:
            try:
                index.index_fact(user_id=user_id, key=updated.key)
            except Exception:  # noqa: BLE001 - 向量重算是增强项
                pass
        return f"[{updated.kind}] {updated.statement}"

    @staticmethod
    def _update_episode(
        store, *, user_id: str, ep_uid: str, summary: str, kind: str, subject: str
    ) -> str | None:
        updated = store.update_ltm_episode(
            ep_uid, user_id=user_id, summary=summary,
            kind=kind or None, subject=subject or None,
        )
        if updated is None:
            return None
        return f"[{updated.kind}] {updated.summary}"


@tool(
    name="forget_memory",
    description="删除一条跨对话记忆（按 search_memory 返回的 ep_/fa_ id 寻址）。",
    capability=Capability.META,
    group=ToolGroup.META,
    permission="write",
    tier="lazy",
    timeout=10,
)
class ForgetMemoryTool(BaseTool):
    @param("memory_id", desc="要删除的记忆 id（ep_xxxxxxxxxxxx 为情节，fa_xxxxxxxxxxxx 为事实/偏好）")
    async def _dispatch(self, *, memory_id: str) -> RawData:
        store = _memory_store(self)
        user_id = _user_id(self)
        if store is None:
            return _no_store("memory:forget", "删除", memory_id=memory_id)
        if memory_id.startswith("fa_"):
            fact = store.get_ltm_fact(memory_id, user_id=user_id)
            removed = store.delete_ltm_fact(memory_id, user_id=user_id)
            # 向量库里的点同步清理：留着会让后续召回命中一个已不存在的条目。
            index = _semantic_index(self)
            if removed and index is not None and fact is not None:
                index.unindex_fact(fact, user_id=user_id)
        else:
            removed = store.delete_ltm_episode(memory_id, user_id=user_id)
        if not removed:
            return RawData(
                kind="text",
                text=f"未找到记忆条目 {memory_id}。",
                endpoint="memory:forget",
                params={"memory_id": memory_id},
            )
        return RawData(
            kind="text",
            text=f"已删除记忆 {memory_id}。",
            endpoint="memory:forget",
            params={"memory_id": memory_id},
        )
