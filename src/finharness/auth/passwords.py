"""密码哈希：PBKDF2-HMAC-SHA256，格式自描述以便日后提升参数。

哈希串形如 ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``，
迭代数随串存储——校验时按串内参数重算，因此提升迭代数不会
使既有密码失效。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ITERATIONS = 600_000
_SALT_BYTES = 16
_KEY_BYTES = 32
_ALGORITHM = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    """生成自描述的密码哈希串。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return "$".join((_ALGORITHM, str(_ITERATIONS), salt.hex(), digest.hex()))


def verify_password(password: str, stored: str) -> bool:
    """恒时校验密码；无法解析的哈希串一律拒绝。"""
    try:
        algorithm, iterations_text, salt_hex, hash_hex = stored.split("$")
        if algorithm != _ALGORITHM:
            return False
        iterations = int(iterations_text)
        if iterations < 1:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), iterations
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)
