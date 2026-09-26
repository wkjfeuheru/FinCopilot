"""把工具参数投影成可下发给前端的摘要。

``tool_status`` 过去只有 call_id / name / status。覆盖条要写「正在读取白酒行业
表现」，就必须带上标的，但又不能把 args 原样倾倒出去——文件正文、密钥、共享
背景都不该出现在 SSE 里。本模块是那道白名单。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from finharness.shared.agents import MAX_SPAWN_TASKS

_SCALAR_KEYS = ("industry", "symbol", "query", "view", "period", "top", "years")
_PATH_KEYS = ("path", "file", "filename")
_MAX_STR = 80


def _clip(value: Any, *, limit: int = _MAX_STR) -> str:
    text = str(value).strip().splitlines()[0].strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def tool_status_summary(name: str, args: Mapping[str, Any] | None) -> dict[str, Any]:
    """返回允许出现在 ``tool_status.summary`` 里的字段；没有可展示项时为空 dict。"""
    payload = dict(args or {})
    out: dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        value = payload.get(key)
        if value is None or value == "":
            continue
        if key in {"top", "years"}:
            try:
                out[key] = int(value)
                continue
            except (TypeError, ValueError):
                pass
        out[key] = _clip(value)
    for key in _PATH_KEYS:
        raw = payload.get(key)
        if raw:
            out["file"] = Path(str(raw)).name
            break
    if name == "spawn_agent":
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            clipped = [_clip(task) for task in tasks if task]
            if clipped:
                out["tasks"] = clipped[:MAX_SPAWN_TASKS]
    return out
