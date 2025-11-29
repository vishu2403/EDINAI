"""Business logic for admin portal member management."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from passlib.context import CryptContext

from ..repository import admin_management_repository, admin_portal_repository, registration_repository
from ..schemas import MemberCreate, MemberResponse, WorkType
from ..utils.role_generator import generate_role_id
from ..utils.passwords import truncate_password
from ..utils import bcrypt_compat  # noqa: F401

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _sync_registration_admin(reg_admin: Dict[str, Any]) -> None:
    """Ensure new-style administrators also exist in legacy admins table for FK references."""

    admin_id = reg_admin.get("admin_aid") or reg_admin.get("admin_id")
    if not admin_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Admin record missing identifier")

    name = reg_admin.get("full_name") or reg_admin.get("name")
    if not name:
        name = reg_admin.get("email", "Admin")

    created_at = reg_admin.get("created_at") or datetime.utcnow()
    validity = (reg_admin.get("validity") or reg_admin.get("package_validity") or "1_year").lower()
    validity_days = 365
    if "month" in validity:
        try:
            months = int("".join(filter(str.isdigit, validity)) or 0)
            validity_days = max(30 * months, 30)
        except ValueError:
            validity_days = 365
    elif validity in {"lifetime", "lifetime_plan"}:
        validity_days = 365 * 10

    expiry_date = created_at + timedelta(days=validity_days)

    email = (reg_admin.get("email") or "").lower()
    mirror_email = email
    if admin_management_repository.admin_exists_by_email(mirror_email, exclude_admin_id=admin_id):
        mirror_email = f"{admin_id}_{mirror_email}"

    admin_management_repository.create_admin(
        admin_id=admin_id,
        name=name,
        email=mirror_email,
        password=reg_admin.get("password"),
        package=reg_admin.get("package_plan") or reg_admin.get("package"),
        start_date=created_at,
        expiry_date=expiry_date,
        has_inai_credentials=bool(
            reg_admin.get("inai_email") and reg_admin.get("inai_password_encrypted")
        ),
        active=True,
        is_super_admin=False,
        created_at=created_at,
        updated_at=created_at,
    )


def _ensure_admin_exists(admin_id: int) -> None:
    if admin_management_repository.admin_exists_by_id(admin_id):
        return
    reg_admin = registration_repository.get_admin_by_id(admin_id)
    if reg_admin:
        _sync_registration_admin(reg_admin)
        return
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")


def list_members(admin_id: int, *, work_type: Optional[WorkType], active_only: bool) -> List[Dict[str, object]]:
    _ensure_admin_exists(admin_id)

    members = admin_portal_repository.list_portal_members(
        admin_id,
        work_type=work_type.value if work_type else None,
        active_only=active_only,
    )
    return [MemberResponse(**member).model_dump() for member in members]


def get_member(member_id: int) -> Dict[str, object]:
    member = admin_portal_repository.get_portal_member_by_id(member_id)
    if not member:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    return MemberResponse(**member).model_dump()


def _hash_password(password: str) -> str:
    safe_password = truncate_password(password)
    return _pwd_context.hash(safe_password)


def create_member(admin_id: int, payload: MemberCreate) -> Dict[str, object]:
    _ensure_admin_exists(admin_id)

    if admin_portal_repository.member_exists_for_admin(email=payload.email, admin_id=admin_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Member with this email already exists for the admin",
        )

    hashed_password = _hash_password(payload.password)
    role_id = generate_role_id(payload.work_type.value)

    member_record = admin_portal_repository.create_portal_member(
        admin_id=admin_id,
        name=payload.name.strip(),
        designation=payload.designation.strip(),
        email=payload.email.lower().strip(),
        phone_number=payload.phone_number.strip() if payload.phone_number else None,
        work_type=payload.work_type.value,
        password=hashed_password,
        role_id=role_id,
        active=True,
        created_at=datetime.utcnow(),
    )

    return MemberResponse(**member_record).model_dump()
