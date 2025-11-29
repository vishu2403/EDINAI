"""Business logic for user operations."""
from __future__ import annotations

from dataclasses import asdict
from typing import List

from ..repository import user_repository
from ..schemas import UserCreate, UserResponse


async def register_user(payload: UserCreate) -> UserResponse:
    """Create a new user and return serialized response."""

    user = await user_repository.create_user(**payload.model_dump())
    return UserResponse(**asdict(user))


async def list_users() -> List[UserResponse]:
    """Return all registered users."""

    users = await user_repository.list_users()
    return [UserResponse(**asdict(user)) for user in users]
