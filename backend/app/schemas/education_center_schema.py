"""Schemas for education center onboarding."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, validator

from .contact_schema import ContactResponse


class EducationCenterResponse(BaseModel):
    id: int
    admin_id: int
    name: str
    upload_image: Optional[str]
    center_photos: Optional[List[str]]
    logo: Optional[str]
    other_activities: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True


class CompleteOnboardingRequest(BaseModel):
    # Contact fields
    first_name: str
    last_name: str
    address: str
    designation: str
    phone_number: str
    date_of_birth: str  # DD-MM-YYYY

    # Education center fields
    education_center_name: str
    upload_image: Optional[str] = Field(
        default=None,
        description="Profile image as Base64 data URL (data:image/...;base64,xxx)",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
    )
    center_photos: Optional[List[str]] = Field(
        default=None,
        description="List of center photo Base64 data URLs",
        example=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    )
    logo: Optional[str] = Field(
        default=None,
        description="Logo image as Base64 data URL",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
    )
    other_activities: Optional[List[str]] = Field(
        default=None,
        description="List of Base64 data URLs for other activity proofs",
        example=["data:application/pdf;base64,JVBERi0xLjUKJcTl8uXrp..."],
    )

    # INAI credentials
    inai_email: EmailStr
    inai_password: str

    @validator("date_of_birth")
    def validate_dob(cls, value: str) -> str:  # noqa: D417
        datetime.strptime(value, "%d-%m-%Y")
        return value


class CompleteOnboardingResponse(BaseModel):
    contact: ContactResponse
    education_center: EducationCenterResponse
    onboarding_completed: bool
    new_token: str
    admin_info: Dict[str, Any]


__all__ = [
    "EducationCenterResponse",
    "CompleteOnboardingRequest",
    "CompleteOnboardingResponse",
]