"""稳定的 JSON 序列化：相同语义的载荷在任何进程里产出同一串字节。

缓存键、引用指纹与轨迹指纹都建立在"把参数规范化成字符串"之上。只要两处
序列化的选项有一处不同（少设一个 ``sort_keys``、忘了 ``default=str``），
同一个请求就会算出两个不同的键，表现为缓存永不命中或指纹对不上——这类
缺陷不会报错，只会静默降级。因此把规范化口径固定在一处。

注意用途边界：这是给**参与计算的载荷**（键、指纹）用的，不用于日志行或
SSE 帧——那些对键序与缩进没有要求，各有自己的格式。
"""

from __future__ import annotations

import json
from typing import Any


def stable_dumps(value: Any) -> str:
    """把载荷规范化为确定性 JSON 字符串（键序固定、保留非 ASCII、可回退）。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
