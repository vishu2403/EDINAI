"""Admin, member, and package schemas reused by dashboard APIs."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, validator


class WorkType(str, Enum):
    CHAPTER = "chapter"
    STUDENT = "student"
    LECTURE = "lecture"


class AdminResponse(BaseModel):
    admin_id: int
    name: str
    email: EmailStr
    package: Optional[str]
    start_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    has_inai_credentials: bool
    active: bool
    is_super_admin: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class _WorkTypeNormalizer:
    @staticmethod
    def normalize(value: str | WorkType) -> str | WorkType:
        if isinstance(value, WorkType):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            alias_map = {
                "chapter management": WorkType.CHAPTER.value,
                "chapter_management": WorkType.CHAPTER.value,
                "student management": WorkType.STUDENT.value,
                "student_management": WorkType.STUDENT.value,
                "lecture management": WorkType.LECTURE.value,
                "lecture_management": WorkType.LECTURE.value,
            }
            return alias_map.get(normalized, normalized)
        return value

class MemberCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    designation: str
    work_type: WorkType
    phone_number: Optional[str] = None

    @validator("work_type", pre=True)
    def normalize_work_type(cls, value: str | WorkType):  # type: ignore[override]
        return _WorkTypeNormalizer.normalize(value)

class MemberUpdate(BaseModel):
    name: Optional[str] = None
    designation: Optional[str] = None
    work_type: Optional[WorkType] = None
    phone_number: Optional[str] = None
    password: Optional[str] = None
    email: Optional[EmailStr] = None

    @validator("work_type", pre=True)
    def normalize_work_type(cls, value: str | WorkType):  # type: ignore[override]
        return _WorkTypeNormalizer.normalize(value)


class MemberResponse(BaseModel):
    member_id: int
    admin_id: int
    name: str
    designation: str
    email: str
    phone_number: Optional[str]
    work_type: str
    role_id: Optional[str] = None
    active: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class PackageResponse(BaseModel):
    id: int
    name: str
    price: float
    duration_days: int
    video_limit: int
    max_quality: str
    max_minutes_per_lecture: int
    ai_videos_per_lecture: int
    topics_per_lecture: int
    extra_credit_price: float
    extra_ai_video_price: float
    discount_rate: float
    support_level: str
    features: Optional[str]
    notes: Optional[str]

    class Config:
        from_attributes = True
