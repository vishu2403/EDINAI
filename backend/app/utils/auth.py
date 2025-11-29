"""Auth helpers for the modular example backend."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from jose import jwt

from ..config import settings

ALGORITHM = "HS256"


def create_access_token(data: Dict[str, Any], expires_minutes: int | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes or settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
