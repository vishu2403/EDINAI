"""Pydantic schemas for admin contact onboarding."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, EmailStr, constr


class ContactBase(BaseModel):
    first_name: constr(min_length=1)
    last_name: Optional[str] = None
    education_center_name: Optional[str] = None
    address: Optional[str] = None
    designation: Optional[str] = None
    phone_number: Optional[str] = None
    dob: Optional[date] = None
    inai_email: Optional[EmailStr] = None


class ContactCreate(ContactBase):
    inai_password: Optional[str] = None


class ContactRead(BaseModel):
    id: int
    admin_id: int
    first_name: str
    last_name: Optional[str] = None
    education_center_name: Optional[str] = None
    address: Optional[str] = None
    designation: Optional[str] = None
    phone_number: Optional[str] = None
    dob: Optional[str] = None
    inai_email: Optional[str] = None
    image_path: Optional[List[str]] = None
    center_photos: Optional[List[str]] = None
    logo_path: Optional[List[str]] = None
    other_activities_path: Optional[List[str]] = None

    class Config:
        from_attributes = True


class ContactResponse(ContactRead):
    """Backward-compatible name used by onboarding routes."""

