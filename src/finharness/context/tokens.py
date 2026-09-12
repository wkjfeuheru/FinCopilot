"""Token counting for context-budget decisions.

Compaction triggers on "how much will the next request cost", so a count is
needed before the request is sent. tiktoken gives a real count; where it is
unavailable (or its vocabulary cannot be fetched) the documented character
approximation is used instead, and the caller is told which path ran.

The vocabulary is cached inside the project rather than the system temp
directory: a cold cache costs roughly two minutes of downloading, and temp
directories get cleaned, which would turn that into an intermittent stall.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# cl100k_base is the closest widely-available vocabulary for the mixed
# Chinese/English traffic this system sees.
ENCODING_NAME = "cl100k_base"
# A single project-local cache for the vocabulary, independent of the data cache
# layout so test temp directories do not each fetch their own copy.
# parents: [0]=context, [1]=finharness, [2]=src, [3]=repo root
VOCAB_CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache" / "tiktoken"
# Documented fallback (docs 03.6.3): Chinese-heavy text runs about 1.7 chars
# per token.
CHARS_PER_TOKEN = 1.7


@dataclass(frozen=True, slots=True)
class TokenCount:
    tokens: int
    exact: bool


class TokenCounter:
    """Counts tokens, preferring tiktoken and degrading to an estimate."""

    def __init__(self, *, cache_dir: str | Path | None = None) -> None:
        # The vocabulary is shared across every counter: it is large, slow to
        # fetch, and identical regardless of which cache directory a caller
        # happens to use. Pointing per-instance directories at it would make each
        # new directory re-download the file (observed as ~100s on first use).
        directory = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(directory))
        self._encoder = None
        self._failed = False

    def _load(self):
        if self._encoder is not None or self._failed:
            return self._encoder
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(ENCODING_NAME)
        except Exception:  # noqa: BLE001 - any failure means "fall back"
            self._failed = True
            self._encoder = None
        return self._encoder

    def warmup(self) -> bool:
        """Fetch the vocabulary once at startup; returns whether it is usable."""
        return self._load() is not None

    def count(self, text: str | None) -> TokenCount:
        if not text:
            return TokenCount(tokens=0, exact=True)
        encoder = self._load()
        if encoder is None:
            return TokenCount(tokens=int(len(text) / CHARS_PER_TOKEN), exact=False)
        try:
            return TokenCount(tokens=len(encoder.encode(text)), exact=True)
        except Exception:  # noqa: BLE001 - treat unusable encoder as absent
            self._failed = True
            return TokenCount(tokens=int(len(text) / CHARS_PER_TOKEN), exact=False)

    def count_many(self, texts) -> int:
        return sum(self.count(text).tokens for text in texts)


def truncate_to_tokens(
    text: str, counter: "TokenCounter", limit: int, *, marker: str = "…"
) -> str:
    """Cut text to a token budget, landing on a character boundary.

    Used wherever a memory layer must fit an injection budget (summary segments,
    recall, long-term recall). The marker is charged *against* the budget, so the
    result never exceeds the limit — otherwise a caller sizing content exactly to
    the budget would overflow by the marker's length and trigger another round.
    """
    if limit <= 0 or counter.count(text).tokens <= limit:
        return text
    marker_tokens = counter.count(marker).tokens
    body_budget = max(limit - marker_tokens, 0)
    low, high = 0, len(text)
    # Counting is monotone in prefix length, so the longest fitting prefix is
    # found by bisection.
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
    """Project-local vocabulary cache; overridable by the environment."""
    override = os.environ.get("TIKTOKEN_CACHE_DIR")
    return Path(override) if override else VOCAB_CACHE_DIR
