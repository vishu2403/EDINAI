"""Authentication service layer."""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, Tuple

from fastapi import HTTPException, status
from passlib.context import CryptContext

from ..config import settings
from ..repository import (
    admin_portal_repository,
    admin_management_repository,
    auth_repository,
    member_repository,
    registration_repository,
)
from ..schemas import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    ResetPasswordRequest,
)
from ..utils.dependencies import create_access_token
from ..utils.role_generator import generate_role_id
from ..utils.passwords import truncate_password
from ..utils import bcrypt_compat  # noqa: F401  # ensure bcrypt shim loads
from ..utils.password_reset_store import store_reset_token, consume_reset_token
from .email_service import send_email, EmailNotConfiguredError

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def ensure_dev_admin_account() -> None:
    """Seed a default dev admin account for local environments."""

    email = settings.dev_admin_email
    password = settings.dev_admin_password

    if not email or not password:
        return

    normalized_email = email.lower().strip()
    if auth_repository.fetch_admin_by_email(normalized_email):
        return

    now = datetime.utcnow()
    expiry_date = now + timedelta(days=settings.dev_admin_expiry_days)

    try:
        admin_management_repository.create_admin(
            name=settings.dev_admin_name,
            email=normalized_email,
            password=get_password_hash(password),
            package=settings.dev_admin_package,
            start_date=now,
            expiry_date=expiry_date,
            has_inai_credentials=True,
            active=True,
            is_super_admin=True,
            created_at=now,
            updated_at=now,
        )
        logger.info("Seeded dev admin account for %s", normalized_email)
    except Exception:
        logger.exception("Failed to seed dev admin account")


def _ensure_unicode_hash(hashed_password: object) -> str:
    """Return a unicode bcrypt hash regardless of DB storage type."""

    if isinstance(hashed_password, memoryview):
        hashed_password = hashed_password.tobytes()
    if isinstance(hashed_password, (bytes, bytearray)):
        return hashed_password.decode("utf-8")
    if isinstance(hashed_password, str):
        return hashed_password
    raise TypeError("Unsupported hashed password type")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        normalized_hash = _ensure_unicode_hash(hashed_password)
        safe_password = truncate_password(plain_password)
        return pwd_context.verify(safe_password, normalized_hash)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    safe_password = truncate_password(password)
    return pwd_context.hash(safe_password)


def _build_admin_login_response(admin: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    auth_repository.update_admin_last_login(admin["admin_id"], datetime.utcnow())

    token_data = {
        "role": "admin",
        "id": admin["admin_id"],
        "package": admin.get("package"),
        "has_inai_credentials": admin.get("has_inai_credentials"),
        "is_super_admin": admin.get("is_super_admin"),
    }

    access_token = create_access_token(data=token_data)

    payload = {
        "token": access_token,
        "role": "admin",
        "id": admin["admin_id"],
        "name": admin.get("name"),
        "email": admin.get("email"),
        "package": admin.get("package"),
        "has_inai_credentials": admin.get("has_inai_credentials"),
        "is_super_admin": admin.get("is_super_admin"),
        "contact_exists": admin.get("contact_exists", True),
    }
    return "Login successful", payload


def _build_portal_admin_login_response(admin: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    registration_repository.update_admin_last_login(admin["admin_aid"], datetime.utcnow())

    has_inai_credentials = bool(admin.get("inai_email") and admin.get("inai_password_encrypted"))

    token_data = {
        "role": "admin",
        "id": admin["admin_aid"],
        "package": admin.get("package_plan"),
        "has_inai_credentials": has_inai_credentials,
        "is_super_admin": False,
    }

    access_token = create_access_token(data=token_data)

    payload = {
        "token": access_token,
        "role": "admin",
        "id": admin["admin_aid"],
        "name": admin.get("full_name"),
        "email": admin.get("email"),
        "package": admin.get("package_plan"),
        "has_inai_credentials": has_inai_credentials,
        "is_super_admin": False,
        "contact_exists": admin.get("contact_exists", True),
    }
    return "Login successful", payload


def login(payload: LoginRequest) -> Tuple[str, Dict[str, object]]:
    email = payload.email.lower().strip()
    password = payload.password

    admin = auth_repository.fetch_admin_by_email(email)
    if admin:
        if not verify_password(password, admin["password"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

        if not admin.get("active", True):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is inactive")

        if datetime.utcnow() > admin["expiry_date"]:
            days_expired = (datetime.utcnow() - admin["expiry_date"]).days
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Account expired {days_expired} days ago on {admin['expiry_date'].strftime('%Y-%m-%d')}",
            )

        return _build_admin_login_response(admin)

    portal_admin = registration_repository.get_admin_by_email(email)
    if portal_admin:
        if not verify_password(password, portal_admin["password"]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

        return _build_portal_admin_login_response(portal_admin)

    portal_member = admin_portal_repository.get_portal_member_by_email(email)
    if not portal_member or not portal_member.get("password"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not verify_password(password, portal_member["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not portal_member.get("active", True):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is inactive")

    work_type_map = {
        "chapter_management": "chapter",
        "chapter": "chapter",
        "student_management": "student",
        "student": "student",
        "lecture_management": "lecture",
        "lecture": "lecture",
    }
    legacy_work_type = work_type_map.get(portal_member.get("work_type"), portal_member.get("work_type"))

    legacy_member = auth_repository.fetch_member_by_email(email)
    if legacy_member is None:
        try:
            legacy_member = member_repository.create_member(
                admin_id=portal_member["admin_id"],
                name=portal_member.get("name"),
                designation=portal_member.get("designation"),
                email=portal_member.get("email"),
                phone_number=portal_member.get("phone_number"),
                work_type=legacy_work_type,
                password=portal_member["password"],
                role_id=generate_role_id(legacy_work_type),
                active=True,
                created_at=datetime.utcnow(),
            )
        except Exception:
            legacy_member = auth_repository.fetch_member_by_email(email)

    if legacy_member:
        auth_repository.update_member_last_login(legacy_member["member_id"], datetime.utcnow())

    portal_member_id = (
        legacy_member["member_id"]
        if legacy_member
        else portal_member.get("member_id")
        or portal_member.get("id")
        or portal_member.get("mid")
    )

    token_data = {
        "role": "member",
        "id": portal_member_id,
        "work_type": legacy_work_type,
        "admin_id": portal_member["admin_id"],
    }
    access_token = create_access_token(data=token_data)

    payload = {
        "token": access_token,
        "role": "member",
        "id": token_data["id"],
        "name": portal_member.get("name"),
        "email": portal_member.get("email"),
        "work_type": legacy_work_type,
        "admin_id": portal_member["admin_id"],
    }
    return "Login successful", payload


def change_password(request: ChangePasswordRequest, current_user: Dict[str, object]) -> None:
    user_obj = current_user["user_obj"]
    if not verify_password(request.old_password, user_obj["password"]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    hashed = get_password_hash(request.new_password)
    if current_user["role"] == "admin":
        admin_id = user_obj.get("admin_id") or user_obj.get("admin_aid") or user_obj.get("id")
        auth_repository.update_admin_password(admin_id, hashed)
        if user_obj.get("email"):
            reg_admin = registration_repository.get_admin_by_email(user_obj["email"].lower())
            if reg_admin and reg_admin.get("admin_aid"):
                registration_repository.update_admin_password(reg_admin["admin_aid"], hashed)
    else:
        auth_repository.update_member_password(user_obj["member_id"], hashed)


def forgot_password(request: ForgotPasswordRequest) -> Tuple[str, Dict[str, object]]:
    email = request.email.lower().strip()
    legacy_admin = auth_repository.fetch_admin_by_email(email)
    registration_admin = registration_repository.get_admin_by_email(email)
    member = auth_repository.fetch_member_by_email(email)

    response_message = "If the email exists, a reset link has been sent"

    if not any((legacy_admin, registration_admin, member)):
        return response_message, {}

    password_reset_url = getattr(settings, "password_reset_url", None)
    if not password_reset_url:
        logger.warning("PASSWORD_RESET_URL is not configured; cannot send reset link")
        return response_message, {}

    reset_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)
    store_reset_token(email=email, token=reset_token, expires_at=expires_at)

    reset_link = f"{password_reset_url}?token={reset_token}&email={email}"

    html_body = f"""
        <p>Hello,</p>
        <p>You requested a password reset.</p>
        <p><a href="{reset_link}">Click here to reset your password</a></p>
        <p>If you didn't request this, ignore this email.</p>
    """

    try:
        send_email(
            to_email=email,
            subject="Reset your INAI password",
            html_body=html_body,
            text_body=f"Reset your password using this link: {reset_link}",
        )
    except EmailNotConfiguredError:
        logger.warning("Email not configured; unable to send password reset email")
    except Exception:
        logger.exception("Failed to send password reset email")

    return response_message, {}


def reset_password(request: ResetPasswordRequest) -> None:
    email = request.email.lower().strip()
    reset_token = request.reset_token
    new_password = request.new_password

    if not reset_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reset token is required")

    if not consume_reset_token(email=email, token=reset_token):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

    legacy_admin = auth_repository.fetch_admin_by_email(email)
    registration_admin = registration_repository.get_admin_by_email(email)
    member = auth_repository.fetch_member_by_email(email)

    if not any((legacy_admin, registration_admin, member)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    hashed = get_password_hash(new_password)
    now = datetime.utcnow()

    if legacy_admin:
        auth_repository.update_admin_password(legacy_admin["admin_id"], hashed, updated_at=now)
    if registration_admin:
        reg_admin_id = registration_admin.get("admin_aid") or registration_admin.get("admin_id")
        if reg_admin_id is not None:
            registration_repository.update_admin_password(reg_admin_id, hashed, updated_at=now)
            if not legacy_admin:
                mirrored_admin = auth_repository.get_admin_by_id(reg_admin_id)
                if mirrored_admin:
                    auth_repository.update_admin_password(mirrored_admin["admin_id"], hashed, updated_at=now)
    if member:
        auth_repository.update_member_password(member["member_id"], hashed)


def get_current_user_info(current_user: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
    user_obj = current_user["user_obj"]

    if current_user["role"] == "admin":
        expiry_date = user_obj.get("expiry_date")
        data = {
            "role": "admin",
            "id": user_obj.get("admin_id"),
            "name": user_obj.get("name"),
            "email": user_obj.get("email"),
            "package": user_obj.get("package"),
            "has_inai_credentials": user_obj.get("has_inai_credentials"),
            "is_super_admin": user_obj.get("is_super_admin"),
            "expiry_date": expiry_date.isoformat() if isinstance(expiry_date, datetime) else expiry_date,
            "days_until_expiry": (expiry_date - datetime.utcnow()).days if isinstance(expiry_date, datetime) else None,
        }
    else:
        work_type = user_obj.get("work_type")
        data = {
            "role": "member",
            "id": user_obj.get("member_id"),
            "name": user_obj.get("name"),
            "email": user_obj.get("email"),
            "work_type": work_type.value if hasattr(work_type, "value") else work_type,
            "role_id": user_obj.get("role_id"),
            "admin_id": user_obj.get("admin_id"),
        }

    return "User information retrieved", data