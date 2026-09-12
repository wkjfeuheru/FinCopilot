"""The system prompt: one definition, loaded from a file (see prompts/system.md).

The prompt is a product asset — it governs when the agent plans, how it cites
data, and when it stops — so it lives in a reviewable markdown file rather than
a string literal, and every entry point (server, tests) loads this same text.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.md"


class PromptNotFoundError(RuntimeError):
    """Raised when the system prompt asset is missing or empty."""


@lru_cache(maxsize=1)
def system_prompt(path: str | Path | None = None) -> str:
    """Return the system prompt text, or raise if the asset is unusable."""
    target = Path(path) if path is not None else PROMPT_PATH
    try:
        text = target.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PromptNotFoundError(f"无法读取系统提示词文件：{target}") from exc
    if not text:
        raise PromptNotFoundError(f"系统提示词文件为空：{target}")
    return text
