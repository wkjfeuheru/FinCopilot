"""有界的插入序（LRU）映射：为进程级注册表提供上限。

长驻进程里最容易出问题的一类状态，是"按 id 索引、只增不减"的普通 dict：
每个会话/对话/用户都插一条，谁都不负责删。会话注册表有 TTL 淘汰，但挂在它
旁边的辅助映射（引用、用户级数据访问）往往没有，于是它们比会话活得更久，
把已经不用的对象永久钉在内存里。

这里把"有上限的映射"做成一个可复用的类型，让这类注册表有一处统一、可测试
的淘汰语义，而不是各自实现一遍。读取会把键标记为最近使用，因此正在被服务的
对话不会被后来者挤掉。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

__all__ = ["BoundedMap"]


class BoundedMap(MutableMapping[K, V], Generic[K, V]):
    """最多保存 ``max_size`` 项的映射；插入超出时淘汰最久未使用的一项。

    ``max_size <= 0`` 表示不设上限（保留普通 dict 行为），使调用方可以沿用
    "未配置即不限制"的约定。
    """

    def __init__(self, max_size: int = 0, initial: dict[K, V] | None = None) -> None:
        self.max_size = int(max_size)
        self._items: OrderedDict[K, V] = OrderedDict()
        if initial:
            for key, value in initial.items():
                self[key] = value

    def __getitem__(self, key: K) -> V:
        value = self._items[key]
        # 命中即视为最近使用：否则一个持续被读取的条目会被新插入的条目挤走。
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        self._evict()

    def __delitem__(self, key: K) -> None:
        del self._items[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def peek(self, key: K, default: V | None = None) -> V | None:
        """读取但不更新最近使用顺序。"""
        return self._items.get(key, default)

    def pop(self, key: K, default: V | None = None) -> V | None:  # type: ignore[override]
        return self._items.pop(key, default)

    def clear(self) -> None:
        self._items.clear()

    def keys(self):  # type: ignore[override]
        return self._items.keys()

    def values(self):  # type: ignore[override]
        return self._items.values()

    def items(self):  # type: ignore[override]
        return self._items.items()

    def _evict(self) -> None:
        if self.max_size <= 0:
            return
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
