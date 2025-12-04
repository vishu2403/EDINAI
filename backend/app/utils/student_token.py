"""Helpers for decoding student JWT tokens used by realtime services."""
from __future__ import annotations

from typing import Any

from jose import JWTError, jwt

from ..config import settings

_STUDENT_JWT_ALGORITHM = "HS256"


def decode_student_token(token: str) -> str:
    """Return enrollment number from student JWT or raise ValueError."""
    if not token:
        raise ValueError("Missing student token")

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[_STUDENT_JWT_ALGORITHM],
        )
    except JWTError as exc:  # pragma: no cover - token decoding failure
        raise ValueError("Invalid student token") from exc

    enrollment = payload.get("enrollment_number")
    if not enrollment:
        raise ValueError("Invalid student token payload")

    return str(enrollment)
