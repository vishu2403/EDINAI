"""Shared password hashing helpers using bcrypt."""
from __future__ import annotations

import bcrypt

_MAX_BCRYPT_BYTES = 72


def _truncate(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) <= _MAX_BCRYPT_BYTES:
        return encoded

    trimmed = encoded[:_MAX_BCRYPT_BYTES]
    while True:
        try:
            return trimmed.decode("utf-8").encode("utf-8")
        except UnicodeDecodeError:
            trimmed = trimmed[:-1]
            if not trimmed:
                raise ValueError("Password is empty after truncation")


def truncate_password(password: str) -> str:
    """Return a UTF-8 string whose encoded form fits bcrypt requirements."""
    return _truncate(password).decode("utf-8")


def hash_password(password: str) -> str:
    safe = _truncate(password)
    return bcrypt.hashpw(safe, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        safe = _truncate(password)
        return bcrypt.checkpw(safe, hashed_password.encode("utf-8"))
    except ValueError:
        return False
