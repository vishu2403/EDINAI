"""Pydantic schemas for user CRUD operations."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: EmailStr
    role: str = Field(..., description="Role label such as admin/member")


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    created_at: datetime
    active: Optional[bool] = True

    class Config:
        from_attributes = True
