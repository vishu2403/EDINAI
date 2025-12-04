"""Pydantic schemas for student data."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StudentCreate(BaseModel):
    user_id: int = Field(..., ge=1)
    enrollment_no: str = Field(..., min_length=5, max_length=16)
    grade: str = Field(..., min_length=1, max_length=10)
    section: Optional[str] = Field(None, max_length=5)


class StudentResponse(BaseModel):
    id: int
    user_id: int
    enrollment_no: str
    grade: str
    section: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
