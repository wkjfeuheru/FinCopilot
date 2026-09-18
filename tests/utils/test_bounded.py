"""``BoundedMap``：进程级注册表统一的有界语义。"""

from finharness.utils.bounded import BoundedMap


def test_evicts_the_oldest_entry_past_the_cap():
    mapping: BoundedMap[str, int] = BoundedMap(max_size=2)

    mapping["a"] = 1
    mapping["b"] = 2
    mapping["c"] = 3

    assert len(mapping) == 2
    assert "a" not in mapping
    assert mapping["b"] == 2
    assert mapping["c"] == 3


def test_read_refreshes_recency_so_active_entries_survive():
    """正在被读取的条目不应被后来者挤掉。"""
    mapping: BoundedMap[str, int] = BoundedMap(max_size=2)
    mapping["a"] = 1
    mapping["b"] = 2

    _ = mapping["a"]  # 触碰 a，使其成为最近使用
    mapping["c"] = 3

    assert "a" in mapping
    assert "b" not in mapping


def test_zero_cap_means_unbounded():
    mapping: BoundedMap[int, int] = BoundedMap(max_size=0)

    for index in range(500):
        mapping[index] = index

    assert len(mapping) == 500


def test_peek_does_not_change_recency():
    mapping: BoundedMap[str, int] = BoundedMap(max_size=2)
    mapping["a"] = 1
    mapping["b"] = 2

    assert mapping.peek("a") == 1
    mapping["c"] = 3

    assert "a" not in mapping, "peek 不应把 a 变成最近使用"


def test_supports_the_dict_surface_used_by_registries():
    mapping: BoundedMap[str, int] = BoundedMap(max_size=3, initial={"a": 1})
    mapping.update({"b": 2})

    assert mapping.get("a") == 1
    assert sorted(mapping.keys()) == ["a", "b"]
    assert sorted((k for k in mapping), ) == ["a", "b"]
    assert mapping.pop("a") == 1
    assert mapping.pop("missing", None) is None

    mapping.clear()
    assert len(mapping) == 0
