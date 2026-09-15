"""Server-sent event（SSE）分帧。"""

import json


def encode_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
