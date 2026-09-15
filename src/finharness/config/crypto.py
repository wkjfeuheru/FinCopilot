"""Provider 密钥的静态加密。

威胁模型是数据库文件泄露或被意外提交，而非主机被入侵：Fernet 密钥与数据库
位于同一台机器上、就放在数据库旁边，且已从版本控制中排除。
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class SecretCipherError(RuntimeError):
    """存储的密钥无法解密时抛出。"""


class SecretCipher:
    """基于 Fernet（AES-128-CBC + HMAC）实现的可逆加密。"""

    def __init__(self, key_path: Path) -> None:
        self.key_path = Path(key_path)
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        """加载密钥文件；不存在时生成新密钥并以 0600 权限写入。

        密钥文件为空时抛出 ``SecretCipherError``。
        """

        if self.key_path.exists():
            key = self.key_path.read_bytes().strip()
            if not key:
                raise SecretCipherError(f"密钥文件为空: {self.key_path}")
            return key
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        descriptor = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(key)
        return key

    def encrypt(self, plaintext: str) -> bytes:
        """加密明文并返回 Fernet token 字节串。"""

        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, token: bytes) -> str:
        """解密 token 并返回 UTF-8 明文。

        token 无效或密钥不匹配时抛出 ``SecretCipherError``。
        """

        try:
            return self._fernet.decrypt(token).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            raise SecretCipherError("无法解密存储的密钥，密钥文件可能已更换") from exc
