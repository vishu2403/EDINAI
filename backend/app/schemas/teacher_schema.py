"""Pydantic schemas for teacher data."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TeacherCreate(BaseModel):
    user_id: int = Field(..., ge=1)
    subject: str = Field(..., min_length=2, max_length=40)
    experience_years: Optional[int] = Field(None, ge=0, le=60)


class TeacherResponse(BaseModel):
    id: int
    user_id: int
    subject: str
    experience_years: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True
