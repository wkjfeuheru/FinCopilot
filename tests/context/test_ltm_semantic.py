"""二期：跨对话语义记忆（facts）与向量检索（docs 03.6.4）。"""


import httpx
import pytest

from finharness.config.settings import Settings
from finharness.context.memory.store import LtmFactRecord, MemoryStore
from finharness.context.memory.vector import (
    QdrantVectorStore,
    SemanticIndex,
    SqliteVectorStore,
    _cosine,
    build_vector_store,
)
from finharness.provider.embeddings import Embedder, _parse_embeddings, build_embedder
from tests.conftest import settings_with_cache


def make_settings(tmp_path, **ltm) -> Settings:
    values = {**ltm}
    return settings_with_cache(tmp_path, ltm=values)


# --- 存储层：语义条目 -------------------------------------------------------

def test_upsert_fact_overwrites_by_key(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    first = store.upsert_ltm_fact(
        user_id="u1", key="report_style", statement="用表格", kind="preference"
    )
    again = store.upsert_ltm_fact(
        user_id="u1", key="report_style", statement="简洁，少用表格", kind="preference"
    )

    assert first is not None and again is not None
    # UPSERT：同键覆盖，不是追加——否则注入了两条互相矛盾的偏好。
    assert first.fa_uid == again.fa_uid
    facts = store.list_ltm_facts(user_id="u1")
    assert len(facts) == 1
    assert facts[0].statement == "简洁，少用表格"


def test_fact_uid_is_derived_from_user_and_key(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    fact = store.upsert_ltm_fact(user_id="u1", key="k", statement="v")
    assert fact is not None

    assert fact.fa_uid.startswith("fa_")
    assert store.get_ltm_fact(fact.fa_uid, user_id="u1").key == "k"
    # 归属不符视同不存在；其他用户同键条目 id 不同。
    assert store.get_ltm_fact(fact.fa_uid, user_id="u2") is None
    other = store.upsert_ltm_fact(user_id="u2", key="k", statement="v")
    assert other.fa_uid != fact.fa_uid


def test_fact_statement_change_invalidates_its_vector(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="k", statement="旧表述")
    store.set_ltm_fact_embedding(user_id="u1", key="k", vector=[1.0, 0.0], model="m")
    assert store.get_ltm_fact_by_key(user_id="u1", key="k").has_embedding is True

    store.upsert_ltm_fact(user_id="u1", key="k", statement="新表述")

    # 表述变了旧向量就不再代表这条记忆，必须重算。
    assert store.get_ltm_fact_by_key(user_id="u1", key="k").has_embedding is False
    assert [f.key for f in store.facts_missing_embedding(user_id="u1")] == ["k"]


def test_unchanged_statement_keeps_its_vector(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="k", statement="同样的话")
    store.set_ltm_fact_embedding(user_id="u1", key="k", vector=[1.0, 0.0], model="m")

    store.upsert_ltm_fact(user_id="u1", key="k", statement="同样的话")

    assert store.get_ltm_fact_by_key(user_id="u1", key="k").has_embedding is True


def test_fact_roundtrips_vector_through_blob(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="k", statement="v")
    vector = [0.25, -0.5, 0.75]
    store.set_ltm_fact_embedding(user_id="u1", key="k", vector=vector, model="m")

    [(fa_uid, restored)] = store.list_fact_vectors(user_id="u1")

    assert len(restored) == 3
    assert restored == pytest.approx(vector, abs=1e-6)


def test_delete_fact_by_uid_is_user_scoped(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    fact = store.upsert_ltm_fact(user_id="u1", key="k", statement="v")
    assert fact is not None

    assert store.delete_ltm_fact(fact.fa_uid, user_id="u2") is False
    assert store.get_ltm_fact(fact.fa_uid, user_id="u1") is not None
    assert store.delete_ltm_fact(fact.fa_uid, user_id="u1") is True
    assert store.get_ltm_fact(fact.fa_uid, user_id="u1") is None


def test_preferences_never_pruned_by_count(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.set_note("report_style", "简洁", user_id="u1")
    for index in range(3):
        store.upsert_ltm_fact(user_id="u1", key=f"fact_{index}", statement=str(index))

    store.prune_ltm_facts(user_id="u1", max_facts=0, max_age_days=3650)

    # 事实被清空，偏好必须留着——它被自动删除会让"记住我的偏好"失效。
    assert store.get_notes(user_id="u1") == {"report_style": "简洁"}


def test_notes_are_absorbed_into_facts(tmp_path):
    """旧库的 notes 内容在初始化时并入 ltm_facts（兼容既有部署）。"""
    import sqlite3

    path = tmp_path / "memory.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE notes (user_id TEXT NOT NULL DEFAULT '', key TEXT NOT NULL,"
        " value TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'preference',"
        " updated_at TEXT NOT NULL, PRIMARY KEY (user_id, key))"
    )
    connection.execute(
        "INSERT INTO notes VALUES ('u1', 'report_style', '简洁', 'preference',"
        " '2026-01-01T00:00:00+00:00')"
    )
    connection.commit()
    connection.close()

    store = MemoryStore(path)

    assert store.get_notes(user_id="u1") == {"report_style": "简洁"}
    # 迁移是幂等的：重新打开不会产生重复。
    MemoryStore(path)
    assert len(store.list_ltm_facts(user_id="u1")) == 1


# --- 嵌入客户端 -------------------------------------------------------------

def test_parse_embeddings_restores_input_order():
    body = {
        "data": [
            {"index": 1, "embedding": [0.4, 0.5]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }
    assert _parse_embeddings(body) == [[0.1, 0.2], [0.4, 0.5]]


def test_parse_embeddings_rejects_junk():
    assert _parse_embeddings(None) == []
    assert _parse_embeddings({"data": "nope"}) == []
    assert _parse_embeddings({"data": [{"embedding": ["x"]}]}) == []


def test_embedder_returns_none_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = Embedder(base_url="https://x/v1", api_key="k", model="m", client=client)

    # 嵌入失败必须返回 None（调用方降级），而不是抛出——语义召回是增强项。
    assert embedder.embed(["hello"]) is None


def test_embedder_detects_dimension_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]},
                               {"index": 1, "embedding": [1.0]}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = Embedder(base_url="https://x/v1", api_key="k", model="m", client=client)

    assert embedder.embed(["a", "b"]) is None


def test_embedder_success_sets_dim():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = Embedder(base_url="https://x/v1", api_key="k", model="m", client=client)

    assert embedder.embed(["a"]) == [[1.0, 2.0, 3.0]]
    assert embedder.dim == 3


def test_build_embedder_returns_none_without_endpoint(tmp_path):
    assert build_embedder(make_settings(tmp_path)) is None

# --- 向量检索 ---------------------------------------------------------------

def test_cosine_is_zero_for_mismatched_dimensions():
    assert _cosine([1.0], [1.0, 2.0]) == 0.0
    assert _cosine([], []) == 0.0
    assert _cosine([0.0], [0.0]) == 0.0


def test_sqlite_vector_store_ranks_by_similarity(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="a", statement="苹果")
    store.upsert_ltm_fact(user_id="u1", key="b", statement="香蕉")
    store.set_ltm_fact_embedding(user_id="u1", key="a", vector=[1.0, 0.0])
    store.set_ltm_fact_embedding(user_id="u1", key="b", vector=[0.0, 1.0])
    backend = SqliteVectorStore(store)

    hits = backend.search(vector=[1.0, 0.0], user_id="u1", limit=2)

    assert hits[0] == store.get_ltm_fact_by_key(user_id="u1", key="a").fa_uid


def test_build_vector_store_defaults_to_sqlite(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    assert isinstance(build_vector_store(make_settings(tmp_path), store), SqliteVectorStore)


def test_semantic_index_is_disabled_without_embedder(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    index = SemanticIndex(store=store, embedder=None, vector_store=SqliteVectorStore(store))

    assert index.enabled is False
    assert index.recall(user_id="u1", query="茅台") == []
    assert index.index_fact(user_id="u1", key="k") is False
    assert index.index_pending(user_id="u1") == 0


def test_semantic_index_recalls_similar_facts(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="maotai", statement="茅台是白酒龙头")
    store.upsert_ltm_fact(user_id="u1", key="bank", statement="银行股看息差")
    embedder = _StubEmbedder()
    index = SemanticIndex(
        store=store, embedder=embedder, vector_store=SqliteVectorStore(store)
    )

    assert index.index_pending(user_id="u1") == 2
    hits = index.recall(user_id="u1", query="茅台的语义", limit=1)

    assert [fact.key for fact in hits] == ["maotai"]
    # 命中的是完整记录，不是只从向量库里捞到的 id。
    assert isinstance(hits[0], LtmFactRecord)


def test_semantic_index_respects_exclude_and_user_scope(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="a", statement="茅台")
    store.upsert_ltm_fact(user_id="u2", key="a", statement="茅台")
    index = SemanticIndex(
        store=store, embedder=_StubEmbedder(), vector_store=SqliteVectorStore(store)
    )
    index.index_pending(user_id="u1")
    index.index_pending(user_id="u2")

    assert index.recall(user_id="u1", query="茅台", exclude_keys={"a"}) == []
    # 用户作用域：u1 的检索不会命中 u2 的条目。
    assert all(fact.key == "a" for fact in index.recall(user_id="u1", query="茅台"))
    assert len(store.list_ltm_facts(user_id="u2")) == 1


def test_semantic_index_returns_empty_when_embedding_fails(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="a", statement="茅台")
    index = SemanticIndex(
        store=store, embedder=_FailingEmbedder(), vector_store=SqliteVectorStore(store)
    )

    assert index.index_pending(user_id="u1") == 0
    assert index.recall(user_id="u1", query="茅台") == []


class _StubEmbedder:
    """确定性嵌入替身：按关键词打分，使相似度的排序可预测。"""

    model = "stub"

    def embed(self, texts):
        return [[1.0, 0.0] if "茅台" in text else [0.0, 1.0] for text in texts]

    def embed_one(self, text):
        return self.embed([text])[0]

    @property
    def dim(self):
        return 2


class _FailingEmbedder:
    model = "failing"

    def embed(self, texts):
        return None

    def embed_one(self, text):
        return None


# --- Qdrant 后端：不可用时逐级降级，绝不抛出 --------------------------------

def _qdrant_settings(tmp_path, **vector_db) -> Settings:
    return Settings(
        ltm={
            "vector_db": {
                "url": "http://localhost:6333",
                "collection": "test_sem",
                **vector_db,
            }
        },
        data={"cache_dir": tmp_path / "cache"},
    )


def test_build_vector_store_selects_qdrant_when_url_configured(tmp_path):
    from finharness.context.memory.vector import QdrantVectorStore

    store = MemoryStore(tmp_path / "memory.db")
    selected = build_vector_store(_qdrant_settings(tmp_path), store)

    assert isinstance(selected, QdrantVectorStore)


def test_qdrant_store_degrades_when_the_service_is_unreachable(tmp_path):
    """服务不可达时 upsert/search 返回失败/空，而不是抛出。"""
    from finharness.context.memory.vector import QdrantVectorStore

    backend = QdrantVectorStore(
        url="http://127.0.0.1:1", collection="x", timeout_s=0.01
    )

    assert backend.upsert(
        point_id="fa_1", vector=[1.0, 0.0], user_id="u1", payload={}
    ) is False
    assert backend.search(vector=[1.0, 0.0], user_id="u1", limit=5) == []
    assert backend.delete(point_id="fa_1", user_id="u1") is False


def test_semantic_index_with_unreachable_qdrant_keeps_fact_pending(tmp_path):
    """Qdrant 配了但挂掉：向量不入库、不置标记，条目保持待回填。

    记录本体已在 SQLite，损失只有"本次没有语义召回"；条目留在回填队列，
    服务恢复后由启动迁移/回填补上，而不是悄悄丢进本地 BLOB 形成第二真相。
    """
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="maotai", statement="茅台是白酒龙头")
    unavailable = QdrantVectorStore(url="http://127.0.0.1:1", collection="x", timeout_s=0.01)
    index = SemanticIndex(store=store, embedder=_StubEmbedder(), vector_store=unavailable)

    assert index.index_pending(user_id="u1") == 0
    fact = store.get_ltm_fact_by_key(user_id="u1", key="maotai")
    assert fact.has_embedding is False
    # 待回填队列保留该条目，服务恢复后可补索引。
    assert [f.key for f in store.facts_missing_embedding(user_id="u1")] == ["maotai"]
    assert index.recall(user_id="u1", query="茅台的语义", limit=1) == []


class _FakeQdrantClient:
    """记录 upsert/delete 调用的替身，替代真实 Qdrant 服务。"""

    def __init__(self):
        self.points = []
        self._collections: list[str] = []

    def get_collections(self):
        return type("R", (), {"collections": [type("C", (), {"name": n}) for n in self._collections]})()

    def create_collection(self, collection_name, vectors_config):
        self._collections.append(collection_name)

    def create_payload_index(self, **kwargs):
        pass

    def upsert(self, collection_name, points):
        self.points.extend(points)
        return True

    def delete(self, collection_name, points_selector):
        return True

    def query_points(self, collection_name, query, limit, query_filter, with_payload):
        return type("R", (), {"points": []})()


def _qdrant_index(store, client) -> SemanticIndex:
    from finharness.context.memory.vector import QdrantVectorStore

    return SemanticIndex(
        store=store,
        embedder=_StubEmbedder(),
        vector_store=QdrantVectorStore(url="http://localhost:6333", collection="x", client=client),
    )


def test_qdrant_mode_writes_vector_only_to_qdrant(tmp_path):
    """配了 Qdrant：向量只进 Qdrant，SQLite 不落 BLOB，以 indexed_at 标记。

    点 ID 是 fa_uid 的确定性 UUID5 派生（Qdrant 1.14+ 不收任意字符串 id），
    原始 fa_uid 存进载荷供召回回表寻址。
    """
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="maotai", statement="茅台是白酒龙头")
    client = _FakeQdrantClient()
    index = _qdrant_index(store, client)

    assert index.index_fact(user_id="u1", key="maotai") is True

    fact = store.get_ltm_fact_by_key(user_id="u1", key="maotai")
    assert fact.has_embedding is True          # 由 indexed_at 承接
    assert store.local_vector_facts() == []    # 没有本地 BLOB 副本
    assert store.facts_missing_embedding(user_id="u1") == []  # 已标记，不再回填
    # 点 ID 已派生为 UUID、原始 fa_uid 在载荷里。
    assert [str(p.id) for p in client.points] == [
        QdrantVectorStore._point_id(fact.fa_uid)
    ]
    assert [p.payload["fa_uid"] for p in client.points] == [fact.fa_uid]


def test_migrate_local_vectors_moves_blob_into_qdrant(tmp_path):
    """存量 BLOB 向量迁入 Qdrant：成功后清空 BLOB、置标记、Qdrant 收到点。"""
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="maotai", statement="茅台是白酒龙头")
    store.set_ltm_fact_embedding(user_id="u1", key="maotai", vector=[1.0, 0.0], model="old")
    client = _FakeQdrantClient()
    index = _qdrant_index(store, client)

    assert index.migrate_local_vectors() == 1

    fact = store.get_ltm_fact_by_key(user_id="u1", key="maotai")
    assert [str(p.id) for p in client.points] == [
        QdrantVectorStore._point_id(fact.fa_uid)
    ]
    assert [p.payload["fa_uid"] for p in client.points] == [fact.fa_uid]
    assert store.local_vector_facts() == []    # BLOB 已清空
    assert store.facts_missing_embedding(user_id="u1") == []  # indexed_at 已置
    assert fact.has_embedding is True


def test_migrate_local_vectors_is_noop_without_qdrant(tmp_path):
    """无向量库部署：BLOB 就是真源，迁移不做任何事。"""
    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="a", statement="茅台")
    store.set_ltm_fact_embedding(user_id="u1", key="a", vector=[1.0, 0.0])
    index = SemanticIndex(store=store, embedder=_StubEmbedder(), vector_store=SqliteVectorStore(store))

    assert index.migrate_local_vectors() == 0
    assert len(store.local_vector_facts()) == 1


def test_migrate_local_vectors_keeps_blob_when_qdrant_unreachable(tmp_path):
    """Qdrant 不可达时不删 BLOB，下次启动重试——不丢已积累的向量。"""
    from finharness.context.memory.vector import QdrantVectorStore

    store = MemoryStore(tmp_path / "memory.db")
    store.upsert_ltm_fact(user_id="u1", key="a", statement="茅台")
    store.set_ltm_fact_embedding(user_id="u1", key="a", vector=[1.0, 0.0])
    unavailable = QdrantVectorStore(url="http://127.0.0.1:1", collection="x", timeout_s=0.01)
    index = SemanticIndex(store=store, embedder=_StubEmbedder(), vector_store=unavailable)

    assert index.migrate_local_vectors() == 0
    assert len(store.local_vector_facts()) == 1
