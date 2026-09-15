"""日志脱敏：观测数据的唯一入口，绝不让凭据进入日志或追踪（docs 03.14）。

两件事分开做：

* **键名匹配**——``api_key``/``token``/``password`` 这类字段整体替换为
  ``<redacted>``，这是审计日志过去的做法，行为保持不变。
* **值模式匹配**——密钥也可能藏在自由文本里（上游报错体、工具参数、URL）。
  对 ``Bearer``/``sk-``/``tvly-`` 等已知形态与超长十六进制串做替换，避免
  在捕获完整 payload 时把凭据原样写进磁盘。
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "MAX_ARG_VALUE_LEN",
    "REDACTED_KEYS",
    "redact",
    "redact_field",
    "redact_text",
    "summarize_args",
]

MAX_ARG_VALUE_LEN = 200
REDACTED = "<redacted>"

# 键名匹配：先归一化（小写、去掉 ``-``/``_``），再做精确匹配，因此
# ``access_token``/``x-api-key``/``apiKey`` 都会命中，而 ``max_tokens``、
# ``budget_tokens`` 这类形近字段不会——它们不是凭据。
REDACTED_KEYS = (
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "credential",
    "credentials",
    "private_key",
)

# 归一化后以此为后缀的键也视为凭据（``refresh_token``、``api_secret``…）。
# ``token`` 是单数后缀，因此 ``tokens`` 结尾的计数类字段不受影响。
_SECRET_SUFFIXES = ("token", "secret", "password", "passwd", "apikey", "privatekey")
_REDACTED_NORMALIZED = frozenset(name.replace("_", "") for name in REDACTED_KEYS)

_MAX_DEPTH = 6

# 值模式：命中即整体替换。顺序有意义——先处理带前缀的形态，再兜底长随机串。
# 最后一条要求"至少含一个数字"，否则一段无空格的纯字母文本（例如被截断的长
# 单词）会被误判为密钥；真实凭据几乎总含数字。
_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\btvly-[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{40,}\b"),
)


def _is_secret_key(key: Any) -> bool:
    """键名是否指向凭据；归一化后精确匹配或后缀匹配。"""
    text = str(key).lower().replace("-", "").replace("_", "")
    if text in _REDACTED_NORMALIZED:
        return True
    return text.endswith(_SECRET_SUFFIXES)


def redact_text(text: str) -> str:
    """把自由文本中形似密钥的片段替换为 ``<redacted>``。"""
    for pattern in _VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact(value: Any, *, _depth: int = 0) -> Any:
    """递归脱敏任意可序列化值；深层结构原样截断而非展开。

    返回结构与入参一致（dict 仍是 dict，list 仍是 list），因此可以直接喂给
    ``json.dumps``。无法判断的深层次会被替换为 ``"<truncated>"``。
    """

    if _depth >= _MAX_DEPTH:
        return "<truncated>"
    if isinstance(value, dict):
        return {
            key: (REDACTED if _is_secret_key(key) else redact(item, _depth=_depth + 1))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, _depth=_depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_field(name: Any, value: Any) -> Any:
    """按字段名脱敏单个值；日志 ``extra`` 这类扁平结构用这个入口。

    ``redact`` 只在 dict 里看到键名时才会按键脱敏，而日志记录在序列化前是
    一个个具名属性，因此需要显式传入键名。
    """
    if _is_secret_key(name):
        return REDACTED
    return redact(value)


def _render_scalar(value: Any) -> str:
    """把单个值渲染成短文本；嵌套容器先做 JSON 化再脱敏。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    try:
        import json

        return json.dumps(redact(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def summarize_args(args: dict) -> str:
    """渲染参数快照，截断过长的值并遮蔽机密。

    键名命中凭据清单时整体替换；否则渲染成文本、脱敏、再按
    ``MAX_ARG_VALUE_LEN`` 截断。输出形如 ``symbol=600519,note=x…``。
    """

    parts: list[str] = []
    for key, value in args.items():
        if _is_secret_key(key):
            parts.append(f"{key}={REDACTED}")
            continue
        text = redact_text(_render_scalar(value))
        if len(text) > MAX_ARG_VALUE_LEN:
            text = text[:MAX_ARG_VALUE_LEN] + "…"
        parts.append(f"{key}={text}")
    return ",".join(parts)
