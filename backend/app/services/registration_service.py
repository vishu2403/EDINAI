"""Business logic for admin registration and unified login."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict

from fastapi import HTTPException, status
from jose import jwt
from ..config import settings
from ..contact import crud as contact_crud
from ..repository import registration_repository
from ..schemas.admin_portal_schema import (
    AdminCreate,
    AdminRegisterResponse,
    AdminResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
)
from ..utils.passwords import hash_password as bcrypt_hash, verify_password as bcrypt_verify
from ..utils.session_store import valid_tokens


def hash_password(password: str) -> str:
    return bcrypt_hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt_verify(plain_password, hashed_password)


def _create_access_token(data: Dict[str, str], *, minutes: int | None = None) -> str:
    expire_delta = timedelta(minutes=minutes or settings.access_token_expire_minutes)
    payload = data.copy()
    payload.update({"exp": datetime.now(timezone.utc) + expire_delta, "type": "access"})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def _create_refresh_token(data: Dict[str, str]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    payload = {"sub": data.get("sub"), "exp": expire, "type": "refresh"}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def _contact_exists(admin_id: int) -> bool:
    contact = contact_crud.get_contact_by_admin(admin_id)
    if not contact:
        return False

    required_fields = [
        contact.first_name,
        contact.address,
        contact.designation,
        contact.phone_number,
        contact.education_center_name,
        contact.inai_email,
        contact.inai_password_encrypted,
    ]
    return all(bool(field) for field in required_fields)


def create_admin(admin: AdminCreate) -> AdminRegisterResponse:
    domain = admin.email.split("@")[-1].lower()
    if domain not in settings.allowed_email_domains:
        raise HTTPException(status_code=403, detail=f"Email domain '{domain}' is not allowed.")

    if registration_repository.admin_exists_by_email(admin.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(admin.password)
    new_admin = registration_repository.create_admin(
        full_name=admin.full_name,
        email=admin.email,
        password=hashed_pw,
        package_plan=admin.package_plan.value,
        validity=admin.validity.value,
    )

    admin_payload = AdminResponse(
        admin_aid=new_admin["admin_aid"],
        full_name=new_admin["full_name"],
        email=new_admin["email"],
        package_plan=admin.package_plan,
        validity=admin.validity,
    )

    return AdminRegisterResponse(
        message=f"✅ Admin registered successfully: {new_admin['full_name']}",
        admin=admin_payload,
    )


def login(credentials: LoginRequest) -> LoginResponse:
    email = credentials.email
    password = credentials.password

    admin = registration_repository.get_admin_by_email(email)
    if admin:
        if not verify_password(password, admin["password"]):
            raise HTTPException(status_code=401, detail="Invalid Login Credentials")

        session_id = f"admin_{admin['admin_aid']}_{int(datetime.now(timezone.utc).timestamp())}"
        access_token = _create_access_token(
            {
                "sub": str(admin["admin_aid"]),
                "user_type": "admin",
                "role": "admin",
                "id": admin["admin_aid"],
                "session_id": session_id,
            }
        )
        refresh_token = _create_refresh_token(
            {
                "sub": str(admin["admin_aid"]),
                "user_type": "admin",
                "role": "admin",
                "id": admin["admin_aid"],
                "session_id": session_id,
            }
        )

        contact_pre_exists = _contact_exists(admin["admin_aid"])
        login_status = "already_logged_in" if contact_pre_exists else "first_time"
        valid_tokens[f"admin_{admin['admin_aid']}"] = session_id

        message = (
            f"Welcome {admin['full_name']}"
            if login_status == "first_time"
            else f"🎉 You have already logged in — Welcome back {admin['full_name']}"
        )

        return LoginResponse(
            message=message,
            user_type="admin",
            admin_id=admin["admin_aid"],
            access_token=access_token,
            refresh_token=refresh_token,
            work_type="admin",
            login_status=login_status,
            contact_exists=contact_pre_exists,
        )

    member = registration_repository.get_portal_member_by_email(email)
    if member:
        if not verify_password(password, member["password"]):
            raise HTTPException(status_code=401, detail="Invalid Login Credentials")

        session_id = f"member_{member['member_id']}_{int(datetime.now(timezone.utc).timestamp())}"
        payload = {
            "sub": str(member["member_id"]),
            "user_type": "member",
            "role": "member",
            "id": member["member_id"],
            "session_id": session_id,
            "admin_id": str(member["admin_id"]),
            "work_type": member["work_type"],
        }
        access_token = _create_access_token(payload)
        refresh_token = _create_refresh_token(payload)

        previous_login = member.get("last_login")
        registration_repository.update_member_last_login(
            admin_id=member["admin_id"],
            member_id=member["member_id"],
            when=datetime.now(timezone.utc),
        )

        is_first_login = previous_login is None
        message = (
            f"Welcome {member['name']}!"
            if is_first_login
            else f"🎉 You have already logged in — Welcome back {member['name']}"
        )
        login_status = "first_time" if is_first_login else "already_logged_in"
        valid_tokens[f"admin_{member['admin_id']}"] = session_id

        return LoginResponse(
            message=message,
            user_type="member",
            admin_id=member["admin_id"],
            member_id=member["member_id"],
            access_token=access_token,
            refresh_token=refresh_token,
            work_type=member["work_type"],
            login_status=login_status,
        )

    raise HTTPException(status_code=401, detail="Invalid Login Credentials")


def logout(token: str) -> LogoutResponse:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    token_type = payload.get("type")
    if token_type != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    user_type = payload.get("user_type")
    if not user_id or not user_type:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    key = None
    if user_type == "admin":
        key = f"admin_{user_id}"
    elif user_type == "member":
        admin_id = payload.get("admin_id")
        if admin_id is not None:
            key = f"admin_{admin_id}"

    if key:
        valid_tokens.pop(key, None)

    return LogoutResponse(message="Logout successful", user_type=user_type, user_id=str(user_id))
