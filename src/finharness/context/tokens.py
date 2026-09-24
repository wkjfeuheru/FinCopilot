"""用于上下文预算决策的 token 计数。

压缩以“下一次请求将花费多少”为触发条件，因此需要在请求发出之前完成计数。
tiktoken 给出真实计数；当它不可用（或无法获取其词表）时，改用文档约定的
字符数近似值，并告知调用方实际走了哪条路径。

词表缓存在项目内部，而不是系统临时目录：一次冷缓存大约要下载两分钟，而临时
目录会被清理，从而把这件事变成间歇性的卡顿。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from finharness.utils.text import CHARS_PER_TOKEN

# cl100k_base 是系统所处理的中英混合文本最接近且广泛可用的词表。
ENCODING_NAME = "cl100k_base"
# 词表统一缓存在项目本地，独立于数据缓存的布局，这样测试用的临时目录
# 就不会各自再下载一份副本。
# parents: [0]=context, [1]=finharness, [2]=src, [3]=repo root
VOCAB_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache" / "tiktoken"


@dataclass(frozen=True, slots=True)
class TokenCount:
    tokens: int
    exact: bool


class TokenCounter:
    """统计 token 数，优先使用 tiktoken，不可用时降级为估算。"""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        # 词表在所有计数器之间共享：它体积大、获取慢，且无论调用方使用哪个
        # 缓存目录，内容都完全相同。让每个实例指向各自的目录会导致每出现一个
        # 新目录都重新下载该文件（首次使用时实测约 100 秒）。
        directory = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._encoder = None
        self._failed = False
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            # 缓存目录不可写（只读镜像、非 root 运行时仓库根属主为 root）绝不能
            # 让构造失败：契约是"编码器不可用时改用字符近似"，把一个观测性的
            # 缓存目录问题升级成请求 500 就违背了它。
            self._failed = True
            return
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(directory))

    def _load(self):
        """惰性加载并缓存编码器；失败时记录标记并返回 None。"""
        if self._encoder is not None or self._failed:
            return self._encoder
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(ENCODING_NAME)
        except Exception:  # noqa: BLE001 - 任何失败都意味着“回退”
            self._failed = True
            self._encoder = None
        return self._encoder

    def warmup(self) -> bool:
        """在启动时获取一次词表；返回它是否可用。"""
        return self._load() is not None

    def count(self, text: str | None) -> TokenCount:
        """统计一段文本的 token 数；编码器不可用时退化为字符近似。"""
        if not text:
            return TokenCount(tokens=0, exact=True)
        encoder = self._load()
        if encoder is None:
            return TokenCount(tokens=int(len(text) / CHARS_PER_TOKEN), exact=False)
        try:
            return TokenCount(tokens=len(encoder.encode(text)), exact=True)
        except Exception:  # noqa: BLE001 - 将不可用的编码器视为不存在
            self._failed = True
            return TokenCount(tokens=int(len(text) / CHARS_PER_TOKEN), exact=False)

    def count_many(self, texts) -> int:
        """统计多段文本的 token 总数。"""
        return sum(self.count(text).tokens for text in texts)


def truncate_to_tokens(
    text: str, counter: TokenCounter, limit: int, *, marker: str = "…"
) -> str:
    """把文本裁剪到指定 token 预算，并落在字符边界上。

    凡是记忆层需要适配注入预算的地方都会用到（摘要分段、召回、长期召回）。
    标记字符也计入预算，因此结果绝不会超过上限 —— 否则把内容恰好按预算大小
    设定的调用方，会因标记字符的长度而溢出并触发又一轮处理。
    """
    if limit <= 0 or counter.count(text).tokens <= limit:
        return text
    marker_tokens = counter.count(marker).tokens
    body_budget = max(limit - marker_tokens, 0)
    low, high = 0, len(text)
    # 计数对前缀长度单调不减，因此用二分法找到能放下的最长前缀。
    while low < high:
        middle = (low + high + 1) // 2
        if counter.count(text[:middle]).tokens <= body_budget:
            low = middle
        else:
            high = middle - 1
    prefix = text[:low].rstrip()
    if len(prefix) < len(text):
        return prefix + marker
    return prefix


def _default_cache_dir() -> Path:
    """项目本地的词表缓存；可由环境变量覆盖。"""
    override = os.environ.get("TIKTOKEN_CACHE_DIR")
    return Path(override) if override else VOCAB_CACHE_DIR


_SHARED_COUNTER: TokenCounter | None = None


def default_counter() -> TokenCounter:
    """进程内共享的计数器。

    在渲染/注入路径上按需构造计数器会各自解析一次词表；共享一个实例让这些调用
    保持廉价，且与 :mod:`finharness.tools.base` 中渲染侧用的是同一份计数口径。
    """
    global _SHARED_COUNTER
    if _SHARED_COUNTER is None:
        _SHARED_COUNTER = TokenCounter()
    return _SHARED_COUNTER
