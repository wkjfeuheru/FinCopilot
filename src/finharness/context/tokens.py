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
        if cache_dir is not None:
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            # tiktoken reads this when resolving its vocabulary blob.
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
