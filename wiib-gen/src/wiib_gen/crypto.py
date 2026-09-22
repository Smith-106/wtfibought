"""用户 API key 的 AES-256-GCM 加解密，与 Java ApiKeyCrypto 同格式。

密文 = base64(iv 12 字节 + 密文 + tag 16 字节)，密钥 = base64 解开的 32 字节 WIIB_TRADER_KEY_SECRET。
密钥没配好或密文过不了 tag 校验一律响亮报错，绝不静默放行。密钥与明文任何情况不进日志。
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import config

IV_LEN = 12


class CryptoError(RuntimeError):
    """加解密失败；只带一句话，不带密文细节"""


def _key(secret_base64: str | None = None) -> bytes:
    raw = config().key_secret if secret_base64 is None else secret_base64
    if not raw or not raw.strip():
        raise CryptoError("未配置 WIIB_TRADER_KEY_SECRET，无法加解密用户 API key")
    try:
        decoded = base64.b64decode(raw.strip(), validate=True)
    except Exception:
        raise CryptoError("WIIB_TRADER_KEY_SECRET 必须是 base64 的 32 字节") from None
    if len(decoded) != 32:
        raise CryptoError("WIIB_TRADER_KEY_SECRET 必须是 base64 的 32 字节")
    return decoded


def encrypt(plain: str, secret_base64: str | None = None) -> str:
    iv = os.urandom(IV_LEN)
    sealed = AESGCM(_key(secret_base64)).encrypt(iv, plain.encode("utf-8"), None)
    return base64.b64encode(iv + sealed).decode("ascii")


def decrypt(enc_base64: str, secret_base64: str | None = None) -> str:
    key = _key(secret_base64)
    try:
        raw = base64.b64decode(enc_base64)
        return AESGCM(key).decrypt(raw[:IV_LEN], raw[IV_LEN:], None).decode("utf-8")
    except Exception:
        # tag 校验失败（密文被改 / 密钥不对）也落这里
        raise CryptoError("API key 解密失败") from None
