"""跨对话语义记忆的向量索引与召回（docs 03.6.4 LTM 语义记忆）。

**记录本体永远在 SQLite（``ltm_facts``）**：治理（查看/编辑/删除/保留）走单一
事实源，向量库只是检索加速器。向量的存放按部署形态二选一：

* 配了 Qdrant（``ltm.vector_db.url``）——向量只存 Qdrant（支持 user 过滤），
  ``ltm_facts.indexed_at`` 标记已索引；历史版本写进 BLOB 的存量在启动时
  迁入 Qdrant 后清空（``migrate_local_vectors``）；
* 没配 Qdrant 但配了 embedding 端点 —— 向量写进 ``ltm_facts.embedding``
  BLOB，``SqliteVectorStore`` 做本地暴力余弦（个人工作台规模毫秒级）；
* 两者都不可用 —— ``SemanticIndex`` 的调用方退回键匹配（``list_ltm_facts``）。

任何一层失效都不会让记忆链路断掉：Qdrant 抖动只损失一次语义召回，条目本体
与键匹配始终可用。
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from finharness.config.settings import Settings
from finharness.context.memory.store import LtmFactRecord, MemoryStore
from finharness.observability import get_logger
from finharness.provider.embeddings import Embedder


class VectorStore(Protocol):
    """向量索引的最小契约：写入、删除、按用户 KNN 检索。"""

    def upsert(
        self, *, point_id: str, vector: list[float], user_id: str, payload: dict[str, Any]
    ) -> bool: ...

    def delete(self, *, point_id: str, user_id: str) -> bool: ...

    def search(
        self, *, vector: list[float], user_id: str, limit: int
    ) -> list[str]: ...


class QdrantVectorStore:
    """Qdrant 支持；不可用时每次调用都返回失败/空（调用方据此降级）。

    刻意**不**做连接重试或异常上抛：语义召回是增强项，向量库挂掉不该让一轮
    对话变慢或失败。惰性导入 ``qdrant_client``，因此核心依赖里没有它。

    点 ID 用 ``fa_uid`` 的 UUID5 派生值：Qdrant 1.14+ 只接受无符号整数或
    UUID 作点 ID，不再收任意字符串；派生是确定性的，同一 ``fa_uid`` 永远
    映射到同一点，治理（删除/覆盖）按 ``fa_uid`` 寻址的语义不变。
    """

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        api_key: str | None = None,
        timeout_s: float = 10.0,
        client: Any | None = None,
        dim: int = 0,
    ) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.dim = dim
        self._client = client
        self._available: bool | None = None
        self._ensured = False

    @staticmethod
    def _point_id(fa_uid: str) -> str:
        """fa_uid → 确定性 UUID（Qdrant 点 ID 合法形式）。"""
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"finharness:fa_uid:{fa_uid}"))

    def _connect(self) -> Any | None:
        if self._client is not None:
            return self._client
        if self._available is False:
            return None
        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self.url, api_key=self.api_key, timeout=self.timeout_s
            )
        except Exception:  # noqa: BLE001 - 未安装/不可达都只意味着降级
            self._available = False
            return None
        return self._client

    def _ensure_collection(self, dim: int) -> bool:
        """首次写入时按实际维度建集合；已存在则复用。"""
        if self._ensured:
            return True
        client = self._connect()
        if client is None:
            return False
        try:
            from qdrant_client import models as qmodels

            existing = {item.name for item in client.get_collections().collections}
            if self.collection not in existing:
                client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=dim, distance=qmodels.Distance.COSINE
                    ),
                )
            # user_id 载荷索引：每个查询都按用户过滤，没有它只能全表扫。
            try:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name="user_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 - 已存在或旧版不支持，均无妨
                pass
            self._ensured = True
            return True
        except Exception:  # noqa: BLE001
            self._available = False
            return False

    def upsert(
        self, *, point_id: str, vector: list[float], user_id: str, payload: dict[str, Any]
    ) -> bool:
        if not vector:
            return False
        if self.dim == 0:
            self.dim = len(vector)
        if not self._ensure_collection(len(vector)):
            return False
        client = self._connect()
        if client is None:
            return False
        try:
            from qdrant_client import models as qmodels

            client.upsert(
                collection_name=self.collection,
                points=[
                    qmodels.PointStruct(
                        id=self._point_id(point_id),
                        vector=vector,
                        payload={"user_id": user_id, **payload},
                    )
                ],
            )
            return True
        except Exception:  # noqa: BLE001 - 写入失败只降级，不影响记忆本体
            get_logger("finharness.memory.vector").warning(
                "qdrant_upsert_failed", extra={"point_id": point_id}
            )
            return False

    def delete(self, *, point_id: str, user_id: str) -> bool:
        client = self._connect()
        if client is None:
            return False
        try:
            from qdrant_client import models as qmodels

            client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(points=[self._point_id(point_id)]),
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    def search(self, *, vector: list[float], user_id: str, limit: int) -> list[str]:
        """按用户 KNN 检索，返回点 ID 的 ``fa_uid`` 原值（UUID5 逆向映射）。

        新版客户端已移除 ``search``，统一走 ``query_points``。为让召回回表
        仍按 ``fa_uid`` 寻址，把 UUID5 派生点 ID 映射回原值——存储侧在载荷
        里保留原 ``fa_uid``，这里直接读回，不做脆弱的字符串逆推。
        """
        client = self._connect()
        if client is None or not vector:
            return []
        try:
            from qdrant_client import models as qmodels

            response = client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=int(limit),
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="user_id",
                            match=qmodels.MatchValue(value=user_id),
                        )
                    ]
                ),
                with_payload=True,
            )
            results: list[str] = []
            for hit in response.points:
                payload = hit.payload or {}
                fa_uid = payload.get("fa_uid")
                if fa_uid:
                    results.append(str(fa_uid))
            return results
        except Exception:  # noqa: BLE001 - 检索失败即降级为"无向量命中"
            return []


class SqliteVectorStore:
    """无向量库时的本地暴力余弦；向量读自 ``ltm_facts.embedding``。

    个人工作台规模（数百到数千条）下，numpy/纯 python 的余弦扫描是毫秒级，
    因此这条降级路径是真正可用的检索，而不是安慰性的空实现。
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def upsert(
        self, *, point_id: str, vector: list[float], user_id: str, payload: dict[str, Any]
    ) -> bool:
        # 向量本体由 store.set_ltm_fact_embedding 写入；这里无需重复落库。
        return True

    def delete(self, *, point_id: str, user_id: str) -> bool:
        return True

    def search(self, *, vector: list[float], user_id: str, limit: int) -> list[str]:
        if not vector:
            return []
        scored: list[tuple[float, str]] = []
        for fa_uid, candidate in self.store.list_fact_vectors(user_id=user_id):
            score = _cosine(vector, candidate)
            if score > 0:
                scored.append((score, fa_uid))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fa_uid for _score, fa_uid in scored[: int(limit)]]


def _cosine(left: list[float], right: list[float]) -> float:
    """余弦相似度；维度不一致时返回 0（视为不可比，而不是抛错）。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def build_vector_store(settings: Settings, store: MemoryStore) -> VectorStore | None:
    """按配置选择向量后端；两者都不可用时返回 ``None``（调用方退回键匹配）。

    Qdrant 只在**显式配了 url** 时启用——一个有 url 但服务没起来的部署，
    其 upsert/search 会各自失败并让召回落到键匹配；这与"没配"在语义上一致。
    """
    config = settings.ltm.vector_db
    if config.url:
        import os

        api_key = os.getenv(config.api_key_env) if config.api_key_env else None
        return QdrantVectorStore(
            url=config.url,
            collection=config.collection,
            api_key=api_key,
            timeout_s=float(config.timeout_s),
            dim=int(config.dim),
        )
    return SqliteVectorStore(store)


class SemanticIndex:
    """语义记忆的向量索引门面：嵌入 → 索引 → 召回，全程可降级。

    三个能力各自独立降级，因此"配了 Qdrant 但没配 embedding"与"配了
    embedding 但 Qdrant 挂了"都不会报错，只是召回少一层：

    * ``embedder is None``        → ``enabled=False``，召回交回键匹配；
    * 向量写入失败                → 记忆本体仍然落库，只是检索弱一些；
    * 无 Qdrant 部署              → 向量写本地 BLOB（``index_fact``），
      ``SqliteVectorStore`` 用它做余弦检索；配了 Qdrant 则向量只存 Qdrant。
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        embedder: Embedder | None,
        vector_store: VectorStore | None,
        local_store: VectorStore | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_store = vector_store
        # 本地兜底：默认总是可用（读 ltm_facts.embedding），因此"向量库抖动"
        # 只降级为更慢的检索，而不是"这次没有语义召回"。
        self.local_store = local_store if local_store is not None else SqliteVectorStore(store)

    @property
    def enabled(self) -> bool:
        """语义召回是否可用（需要嵌入端点；向量后端二者之一）。"""
        return self.embedder is not None and (
            self.vector_store is not None or self.local_store is not None
        )

    def index_fact(self, *, user_id: str, key: str) -> bool:
        """为一条语义条目计算并写入向量；任何失败都返回 False 而不抛出。

        向量只存一份，真源按部署形态二选一：配了外部向量库（Qdrant）时
        向量只进 Qdrant、SQLite 不落 BLOB（``indexed_at`` 标记已索引）；
        没配时写本地 BLOB，即 ``SqliteVectorStore`` 的检索路径。
        """
        if not self.enabled:
            return False
        assert self.embedder is not None
        fact = self.store.get_ltm_fact_by_key(user_id=user_id, key=key)
        if fact is None:
            return False
        text = _fact_embedding_text(fact)
        vector = self.embedder.embed_one(text)
        if not vector:
            return False
        if isinstance(self.vector_store, SqliteVectorStore):
            # 无向量库部署：BLOB 就是真源，写入即完成索引。
            self.store.set_ltm_fact_embedding(
                user_id=user_id, key=key, vector=vector, model=self.embedder.model
            )
            return True
        if self.vector_store is None:
            return False
        indexed = self.vector_store.upsert(
            point_id=fact.fa_uid,
            vector=vector,
            user_id=user_id,
            payload={"fa_uid": fact.fa_uid, "key": key, "kind": fact.kind},
        )
        if indexed:
            self.store.mark_ltm_fact_indexed(user_id=user_id, key=key)
        return indexed

    def index_pending(self, *, user_id: str, limit: int = 50) -> int:
        """回填尚未向量化的条目（新写入的，或被改写后向量失效的）。"""
        if not self.enabled:
            return 0
        pending = self.store.facts_missing_embedding(user_id=user_id, limit=limit)
        return sum(
            1 for fact in pending if self.index_fact(user_id=user_id, key=fact.key)
        )

    def migrate_local_vectors(self, *, limit: int = 500) -> int:
        """把本地 BLOB 存量向量迁入外部向量库；返回迁移成功的条数。

        历史版本曾把向量双写进 SQLite BLOB，改为"配了 Qdrant 只存 Qdrant"
        后，这批存量需要在启动时搬家。向量直接取自 BLOB（不重调 embedding
        API），外部库确认写入成功才清空 BLOB；外部库不可用时不删，下次启动
        重试。没有外部向量库时无事可做。
        """
        if not self.enabled or not isinstance(self.vector_store, QdrantVectorStore):
            return 0
        migrated = 0
        for user_id, key, fa_uid, vector in self.store.local_vector_facts(limit=limit):
            if not vector:
                continue
            fact = self.store.get_ltm_fact_by_key(user_id=user_id, key=key)
            if fact is None:
                continue
            ok = self.vector_store.upsert(
                point_id=fa_uid,
                vector=vector,
                user_id=user_id,
                payload={"fa_uid": fa_uid, "key": key, "kind": fact.kind},
            )
            if not ok:
                continue
            self.store.clear_ltm_fact_embedding(user_id=user_id, key=key)
            self.store.mark_ltm_fact_indexed(user_id=user_id, key=key)
            migrated += 1
        return migrated

    def unindex_fact(self, fact: LtmFactRecord, *, user_id: str) -> None:
        """删除条目时同步清理向量库中的点（失败不影响删除本身）。"""
        if self.vector_store is None:
            return
        try:
            self.vector_store.delete(point_id=fact.fa_uid, user_id=user_id)
        except Exception:  # noqa: BLE001
            pass

    def recall(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        exclude_keys: set[str] | None = None,
    ) -> list[LtmFactRecord]:
        """按语义相似度召回条目；返回按相似度降序的完整记录。

        命中的 id 回到 SQLite 取正文——向量库只存向量与 id，因此这里
        不存在"两份事实源"的同步问题。

        无 Qdrant 部署走本地 BLOB 余弦；Qdrant 部署检索失败时本次召回为空，
        条目本体与键匹配不受影响。
        """
        if not self.enabled or not query.strip():
            return []
        assert self.embedder is not None
        vector = self.embedder.embed_one(query)
        if not vector:
            return []
        excluded = exclude_keys or set()
        candidates: list[str] = []
        primary = self.vector_store
        if primary is not None and not isinstance(primary, SqliteVectorStore):
            candidates = primary.search(
                vector=vector, user_id=user_id, limit=limit * 2
            )
        if not candidates and self.local_store is not None:
            candidates = self.local_store.search(
                vector=vector, user_id=user_id, limit=limit * 2
            )
        found: list[LtmFactRecord] = []
        for fa_uid in candidates:
            fact = self.store.get_ltm_fact(fa_uid, user_id=user_id)
            if fact is None or fact.key in excluded:
                continue
            found.append(fact)
            if len(found) >= limit:
                break
        return found


def _fact_embedding_text(fact: LtmFactRecord) -> str:
    """条目用于嵌入的文本：带 subject 上下文能显著改善标的相关的召回。"""
    if fact.subject:
        return f"{fact.subject}：{fact.statement}"
    return fact.statement


__all__ = [
    "QdrantVectorStore",
    "SemanticIndex",
    "SqliteVectorStore",
    "VectorStore",
    "build_vector_store",
]
