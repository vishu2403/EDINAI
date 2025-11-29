"""Compatibility helpers for passlib+bcrypt."""
from __future__ import annotations

from typing import Callable

import bcrypt


_MAX_BCRYPT_BYTES = 72


def _ensure_about_attribute() -> None:
    if hasattr(bcrypt, "__about__"):
        return

    class _About:  # pragma: no cover - compatibility shim
        __slots__ = ("__version__",)

        def __init__(self, version: str) -> None:
            self.__version__ = version

    version = getattr(bcrypt, "__version__", "0")
    bcrypt.__about__ = _About(version)


def _truncate_secret(secret: bytes) -> bytes:
    if len(secret) <= _MAX_BCRYPT_BYTES:
        return secret
    return secret[:_MAX_BCRYPT_BYTES]


def _wrap_hash_function(func: Callable[[bytes, bytes], bytes]) -> Callable[[bytes, bytes], bytes]:
    def _wrapper(secret: bytes, salt: bytes) -> bytes:
        if isinstance(secret, str):  # pragma: no cover - bcrypt expects bytes
            secret_bytes = secret.encode("utf-8")
        elif isinstance(secret, memoryview):
            secret_bytes = secret.tobytes()
        else:
            secret_bytes = secret
        return func(_truncate_secret(secret_bytes), salt)

    return _wrapper


def _wrap_check_function(func: Callable[[bytes, bytes], bool]) -> Callable[[bytes, bytes], bool]:
    def _wrapper(secret: bytes, hashed: bytes) -> bool:
        if isinstance(secret, str):  # pragma: no cover
            secret_bytes = secret.encode("utf-8")
        elif isinstance(secret, memoryview):
            secret_bytes = secret.tobytes()
        else:
            secret_bytes = secret
        return func(_truncate_secret(secret_bytes), hashed)

    return _wrapper


def _ensure_truncation_wrappers() -> None:
    bcrypt.hashpw = _wrap_hash_function(bcrypt.hashpw)
    if hasattr(bcrypt, "checkpw"):
        bcrypt.checkpw = _wrap_check_function(bcrypt.checkpw)


_ensure_about_attribute()
_ensure_truncation_wrappers()
