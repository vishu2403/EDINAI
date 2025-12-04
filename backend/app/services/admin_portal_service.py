"""Business logic for admin portal member management."""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, UploadFile, status
from passlib.context import CryptContext

from ..repository import (
    admin_management_repository,
    admin_portal_repository,
    auth_repository,
    member_repository,
    registration_repository,
)
from ..contact import crud as contact_crud
from ..schemas import ContactCreate, MemberCreate, MemberResponse, MemberUpdate, WorkType
from ..utils.role_generator import generate_role_id
from ..utils.passwords import truncate_password
from ..utils.file_handler import save_uploaded_file
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

def _first_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    for part in value.split(";"):
        cleaned = part.strip()
        if cleaned:
            return cleaned
    return None


def get_admin_profile(admin_id: int) -> Dict[str, Any]:
    auth_admin = auth_repository.get_admin_by_id(admin_id)
    reg_admin = registration_repository.get_admin_by_id(admin_id)
    contact = contact_crud.get_contact_by_admin(admin_id)

    if not auth_admin and not reg_admin and not contact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")

    email = (
        (contact and contact.inai_email)
        or (reg_admin or {}).get("email")
        or (auth_admin or {}).get("email")
    )

    phone_number = (
        (contact and contact.phone_number)
        or (reg_admin or {}).get("phone_number")
        or (auth_admin or {}).get("phone")
    )

    designation = (
        (contact and contact.designation)
        or (reg_admin or {}).get("designation")
        or (auth_admin or {}).get("designation")
    )

    photo = _first_path(contact.image_path) if contact else None

    full_name = None
    if contact:
        full_name = " ".join(filter(None, [contact.first_name, contact.last_name])) or None
    if not full_name:
        full_name = (reg_admin or {}).get("full_name")
    if not full_name:
        full_name = (auth_admin or {}).get("name")
    if not full_name and email:
        full_name = email

    return {
        "full_name": full_name,
        "email": email,
        "phone_number": phone_number,
        "designation": designation,
        "photo": photo,
    }


def _split_full_name(full_name: str) -> tuple[str, Optional[str]]:
    parts = full_name.strip().split()
    if not parts:
        return "", None
    if len(parts) == 1:
        return parts[0], None
    return " ".join(parts[:-1]), parts[-1]


async def update_admin_profile(
    admin_id: int,
    *,
    full_name: str,
    phone_number: Optional[str] = None,
    photo_file: Optional[UploadFile] = None,
) -> Dict[str, Any]:
    trimmed_name = (full_name or "").strip()
    if not trimmed_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Full name is required")

    first_name, last_name = _split_full_name(trimmed_name)
    if not first_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="First name is required")

    trimmed_phone = phone_number.strip() if phone_number else None
    if trimmed_phone == "":
        trimmed_phone = None

    contact = contact_crud.get_contact_by_admin(admin_id)

    photo_path: Optional[str] = None
    if photo_file is not None:
        upload_info = await save_uploaded_file(
            photo_file,
            subfolder=f"education_center/admin_{admin_id}/profile",
        )
        photo_path = upload_info["file_path"]

    updates: Dict[str, Any] = {
        "first_name": first_name,
        "last_name": last_name,
    }

    if trimmed_phone is not None:
        updates["phone_number"] = trimmed_phone

    if photo_path is not None:
        updates["image_path"] = photo_path

    if contact:
        contact_crud.update_contact(contact.id, **updates)
    else:
        contact_crud.create_contact(
            ContactCreate(
                first_name=first_name,
                last_name=last_name,
                phone_number=trimmed_phone,
            ),
            admin_id=admin_id,
            created_by=admin_id,
            image_path=photo_path,
        )

    return get_admin_profile(admin_id)


def _hash_password(password: str) -> str:
    safe_password = truncate_password(password)
    return _pwd_context.hash(safe_password)


def create_member(admin_id: int, payload: MemberCreate) -> tuple[Dict[str, object], bool]:
    _ensure_admin_exists(admin_id)

    normalized_email = payload.email.strip().lower()

    existing_member = member_repository.get_member_by_email(normalized_email)
    if existing_member:
        if existing_member.get("admin_id") == admin_id:
            return MemberResponse(**existing_member).model_dump(), False
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists for another admin",
        )

    hashed_password = _hash_password(payload.password)
    role_id = generate_role_id(payload.work_type.value)

    member_record = admin_portal_repository.create_portal_member(
        admin_id=admin_id,
        name=payload.name.strip(),
        designation=payload.designation.strip(),
        email=normalized_email,
        phone_number=payload.phone_number.strip() if payload.phone_number else None,
        work_type=payload.work_type.value,
        password=hashed_password,
        role_id=role_id,
        active=True,
        created_at=datetime.utcnow(),
    )

    return MemberResponse(**member_record).model_dump(), True


def update_member(admin_id: int, member_id: int, payload: MemberUpdate) -> Dict[str, object]:
    _ensure_admin_exists(admin_id)

    existing = admin_portal_repository.get_portal_member_by_id(member_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    if existing["admin_id"] != admin_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot update member for another admin")

    update_fields: Dict[str, Any] = {}
    if payload.name is not None:
        update_fields["name"] = payload.name.strip()
    if payload.designation is not None:
        update_fields["designation"] = payload.designation.strip()
    if payload.phone_number is not None:
        trimmed_phone = payload.phone_number.strip()
        update_fields["phone_number"] = trimmed_phone or None
    if payload.work_type is not None:
        update_fields["work_type"] = payload.work_type.value if isinstance(payload.work_type, WorkType) else payload.work_type
    if payload.password:
        update_fields["password"] = _hash_password(payload.password)
    if payload.email is not None:
        update_fields["email"] = payload.email.lower().strip()

    if not update_fields:
        return MemberResponse(**existing).model_dump()

    updated = member_repository.update_member(member_id, **update_fields)
    if not updated:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update member")

    return MemberResponse(**updated).model_dump()


def delete_member(admin_id: int, member_id: int) -> None:
    _ensure_admin_exists(admin_id)

    existing = admin_portal_repository.get_portal_member_by_id(member_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    if existing["admin_id"] != admin_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete member for another admin")

    deleted = member_repository.delete_member(member_id)
    if deleted is False:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete member")