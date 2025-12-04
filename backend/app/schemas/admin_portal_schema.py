"""Pydantic schemas for the admin portal module."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, constr


class PackagePlan(str, Enum):
    """Subscription tiers supported by the admin portal."""

    PLAN_20K = "20k"
    PLAN_50K = "50k"
    PLAN_100K = "100k"


class Validity(str, Enum):
    """Supported validity durations."""

    ONE_YEAR = "1_year"


class AdminPortalHealthResponse(BaseModel):
    status: str
    module: str


class AdminBase(BaseModel):
    full_name: str = Field(..., min_length=1)
    email: EmailStr
    package_plan: PackagePlan
    validity: Validity


class AdminCreate(AdminBase):
    password: constr(min_length=6)


class AdminResponse(AdminBase):
    admin_aid: int

    class Config:
        from_attributes = True


class AdminRegisterResponse(BaseModel):
    message: str
    admin: AdminResponse


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    message: str
    user_type: str
    admin_id: Optional[int] = None
    member_id: Optional[int] = None
    access_token: str
    refresh_token: str
    work_type: Optional[str] = None
    login_status: Optional[str] = None
    contact_exists: Optional[bool] = None


class LogoutResponse(BaseModel):
    message: str
    user_type: Optional[str] = None
    user_id: Optional[str] = None


class DonutSegment(BaseModel):
    label: str
    value: int


class TotalStaffUI(BaseModel):
    title: Optional[str] = None
    segments: List[DonutSegment] = Field(default_factory=list)


class CardItem(BaseModel):
    title: str
    value: Any


class DashboardUIData(BaseModel):
    cards: Dict[str, CardItem] = Field(default_factory=dict)
    total_staff: Optional[TotalStaffUI] = None
    admin_info: Dict[str, Any] = Field(default_factory=dict)
    package_info: Optional[Dict[str, Any]] = None
    member_statistics: Optional[Dict[str, Any]] = None
    account_status: Optional[Dict[str, Any]] = None
    package_usage: Optional[Dict[str, Any]] = None
    recent_activity: Optional[Dict[str, Any]] = None


class DashboardUIResponse(BaseModel):
    status: bool
    message: str
    data: Optional[DashboardUIData] = None
