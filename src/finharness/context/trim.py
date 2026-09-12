"""Context trimming: what gets trimmed before it reaches the model (docs 3.6.1).

Trimming only affects the *context copy*. Full data stays in the cache parquet,
so nothing is lost — the model can read it back when it needs detail.
"""

from __future__ import annotations

from typing import Any

# Fallback description budget when the caller supplies none.
DEFAULT_MAX_DESC_LEN = 60


def trim_schema(tool_schema: dict, *, max_desc_len: int = DEFAULT_MAX_DESC_LEN) -> dict:
    """Shorten a tool schema's description in place, preserving its shape.

    Every resident tool's schema is re-sent on every request, so description
    length is a fixed per-turn cost (docs 3.6.1). The cut is on a character
    boundary and marked so a reader can tell it was shortened.
    """
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return tool_schema
    description = function.get("description")
    if isinstance(description, str) and len(description) > max_desc_len:
        function["description"] = description[: max_desc_len - 1].rstrip() + "…"
    return tool_schema


def schema_tokens(schema: dict, *, counter: Any | None = None) -> int:
    """Count the tokens a single schema contributes to a request."""
    function = schema.get("function", {})
    text = str(function.get("description", "")) + str(function.get("parameters", ""))
    if counter is None:
        return int(len(text) / 1.7)
    return counter.count(text).tokens
