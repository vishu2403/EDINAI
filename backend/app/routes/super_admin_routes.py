"""Super Admin routes - separate from auth."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, status
from jose import jwt

from ..config import settings
from ..schemas.auth_schema import LoginRequest as AuthLoginRequest
from ..schemas.response import ResponseBase
from ..utils.session_store import valid_tokens


router = APIRouter(prefix="/dev_admin", tags=["Super Admin"])

# Super Admin credentials
SUPER_ADMIN_EMAIL = "admin@inai.edu"
SUPER_ADMIN_PASSWORD = "superadmin123"


@router.post("/login", response_model=ResponseBase)
async def super_admin_login(payload: AuthLoginRequest) -> ResponseBase:
    """Super Admin login endpoint with hardcoded credentials."""
    if payload.email != SUPER_ADMIN_EMAIL or payload.password != SUPER_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid super admin credentials",
        )

    # Create tokens for super admin
    session_id = f"super_admin_{int(datetime.now(timezone.utc).timestamp())}"
    
    access_token_data = {
        "sub": "super_admin",
        "user_type": "super_admin",
        "role": "super_admin",
        "id": "super_admin",
        "session_id": session_id,
    }
    
    expire_delta = timedelta(minutes=settings.access_token_expire_minutes)
    access_payload = access_token_data.copy()
    access_payload.update({"exp": datetime.now(timezone.utc) + expire_delta, "type": "access"})
    access_token = jwt.encode(access_payload, settings.secret_key, algorithm=settings.algorithm)
    
    expire = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    refresh_payload = {
        "sub": "super_admin",
        "exp": expire,
        "type": "refresh",
        "session_id": session_id,
    }
    refresh_token = jwt.encode(refresh_payload, settings.secret_key, algorithm=settings.algorithm)
    
    valid_tokens["super_admin"] = session_id
    
    return ResponseBase(
        status=True,
        message="🎉 Super Admin Login Successful",
        data={
            "user_type": "super_admin",
            "role": "super_admin",
            "access_token": access_token,
            "token": access_token,
            "refresh_token": refresh_token,
        },
    )
