"""User routes for the modular example backend."""
from __future__ import annotations

from fastapi import APIRouter, status

from ..schemas import ResponseEnvelope, UserCreate, UserResponse
from ..services import user_service

router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=ResponseEnvelope, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreate) -> ResponseEnvelope:
    user = await user_service.register_user(payload)
    return ResponseEnvelope(status=True, message="User created", data=user)


@router.get("/", response_model=ResponseEnvelope)
async def list_users() -> ResponseEnvelope:
    users = await user_service.list_users()
    return ResponseEnvelope(status=True, message="Users fetched", data=users)
