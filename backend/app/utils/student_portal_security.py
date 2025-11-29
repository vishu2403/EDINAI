"""Password hashing helpers for the student portal."""
from __future__ import annotations

from .passwords import hash_password as _bcrypt_hash_password, verify_password as _bcrypt_verify_password


def hash_password(password: str) -> str:
    return _bcrypt_hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return _bcrypt_verify_password(plain_password, hashed_password)
    except Exception:
        return False
