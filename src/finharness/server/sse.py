"""Server-sent event（SSE）分帧。"""

import json

# 心跳间隔（秒）。两次引擎事件之间的静默可能长达数十秒（例如逐份研报做
# 摘要），而中间的代理或浏览器可能在这种静默下把连接攒住甚至截断。定期
# 下发一个 SSE 注释帧既保活，也让"还在跑"与"连接已死"可以被区分开——
# 后者正是前端界面永久停在"运行中"的原因之一。
HEARTBEAT_S = 15.0


def encode_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def encode_comment(text: str = "keep-alive") -> str:
    """编码一个 SSE 注释帧（心跳）。

    注释帧以 ``:`` 开头且不含 ``event:``/``data:``，按规范会被客户端忽略，
    因此不会污染事件流；它的唯一作用是让字节持续流动。
    """
    return f": {text}\n\n"
