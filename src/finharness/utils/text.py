"""与领域无关的文本度量常量。

``CHARS_PER_TOKEN`` 是文档约定的中文文本近似值，被 context（token 计数回退）、
tools（描述预算换算）与 shared（分片估算）共同使用。它原先定义在
``context/tokens.py``，导致下层模块为拿一个常量而反向依赖 context；下沉到 utils
后，各层都只向下取用。
"""

from __future__ import annotations

# 文档约定的回退方案（docs 03.6.3）：中文为主文本约每 token 1.7 个字符。
CHARS_PER_TOKEN = 1.7
