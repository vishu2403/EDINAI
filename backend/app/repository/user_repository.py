"""Repository layer for user persistence.

Currently uses an in-memory store to keep the example self-contained.
swap with real database calls when wiring to PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..models import User

_USERS: Dict[int, User] = {}
_USER_SEQUENCE = 0


def _next_id() -> int:
    global _USER_SEQUENCE
    _USER_SEQUENCE += 1
    return _USER_SEQUENCE


async def create_user(*, name: str, email: str, role: str) -> User:
    user = User(
        id=_next_id(),
        name=name,
        email=email,
        role=role,
        created_at=datetime.utcnow(),
    )
    _USERS[user.id] = user
    return user


async def get_user_by_id(user_id: int) -> Optional[User]:
    return _USERS.get(user_id)


async def list_users() -> List[User]:
    return list(_USERS.values())
