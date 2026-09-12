"""At-rest encryption for provider secrets.

The threat model is a leaked or accidentally committed database file, not a
compromised host: the Fernet key lives beside the database on the same machine
and is excluded from version control.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class SecretCipherError(RuntimeError):
    """Raised when a stored secret cannot be decrypted."""


class SecretCipher:
    """Reversible encryption built on Fernet (AES-128-CBC + HMAC)."""

    def __init__(self, key_path: Path) -> None:
        self.key_path = Path(key_path)
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
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
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, token: bytes) -> str:
        try:
            return self._fernet.decrypt(token).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            raise SecretCipherError("无法解密存储的密钥，密钥文件可能已更换") from exc
