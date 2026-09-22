import base64

import pytest

from wiib_gen.crypto import IV_LEN, CryptoError, decrypt, encrypt

SECRET = base64.b64encode(bytes(range(32))).decode()


def test_roundtrip():
    enc = encrypt("sk-abc-123", SECRET)
    assert decrypt(enc, SECRET) == "sk-abc-123"


def test_format_is_iv_plus_cipher_plus_tag():
    plain = "中文 key 也要对得上"
    raw = base64.b64decode(encrypt(plain, SECRET))
    assert len(raw) == IV_LEN + len(plain.encode()) + 16


def test_iv_is_fresh_each_time():
    assert encrypt("same", SECRET) != encrypt("same", SECRET)


def test_secret_must_be_32_bytes():
    short = base64.b64encode(b"too-short").decode()
    with pytest.raises(CryptoError, match="32 字节"):
        encrypt("x", short)


def test_secret_missing():
    with pytest.raises(CryptoError, match="WIIB_TRADER_KEY_SECRET"):
        decrypt("whatever", "")


def test_tampered_cipher_fails():
    raw = bytearray(base64.b64decode(encrypt("sk-abc-123", SECRET)))
    raw[-1] ^= 0xFF
    with pytest.raises(CryptoError, match="解密失败"):
        decrypt(base64.b64encode(raw).decode(), SECRET)


def test_wrong_key_fails():
    other = base64.b64encode(bytes(range(1, 33))).decode()
    with pytest.raises(CryptoError, match="解密失败"):
        decrypt(encrypt("sk-abc-123", SECRET), other)
