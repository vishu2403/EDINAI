"""Authentication API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse

from ..schemas import (
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest as AuthLoginRequest,
    ResetPasswordRequest,
    ResponseBase,
)
from ..schemas.admin_portal_schema import LoginRequest as AdminPortalLoginRequest
from ..services import auth_service, registration_service
from ..utils.dependencies import get_current_user


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=ResponseBase)
async def login(payload: AuthLoginRequest) -> ResponseBase:
    login_request = AdminPortalLoginRequest(email=payload.email, password=payload.password)

    try:
        portal_response = registration_service.login(login_request)
        portal_data = portal_response.model_dump(exclude_none=True)
        access_token = portal_data.get("access_token")
        if access_token:
            portal_data.setdefault("token", access_token)
        if "refresh_token" not in portal_data:
            portal_data["refresh_token"] = None
        return ResponseBase(status=True, message=portal_response.message, data=portal_data)
    except HTTPException as exc:
        if exc.status_code not in {status.HTTP_401_UNAUTHORIZED, status.HTTP_404_NOT_FOUND}:
            raise

    message, legacy_data = auth_service.login(payload)
    legacy_payload = dict(legacy_data)
    token = legacy_payload.get("token")
    legacy_payload.setdefault("access_token", token)
    legacy_payload.setdefault("refresh_token", None)
    return ResponseBase(status=True, message=message, data=legacy_payload)


@router.post("/logout", response_model=ResponseBase)
async def logout(authorization: str = Header(None)) -> ResponseBase:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format",
        )

    token = authorization.split(" ", 1)[1]

    try:
        logout_response = registration_service.logout(token)
        logout_data = logout_response.model_dump(exclude_none=True, exclude={"message"})
        return ResponseBase(status=True, message=logout_response.message, data=logout_data or {})
    except HTTPException as exc:
        if exc.status_code not in {status.HTTP_401_UNAUTHORIZED, status.HTTP_404_NOT_FOUND}:
            raise

    return ResponseBase(status=True, message="Logout successful", data={})


@router.post("/change-password", response_model=ResponseBase)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
) -> ResponseBase:
    auth_service.change_password(payload, current_user)
    return ResponseBase(status=True, message="Password changed successfully", data={})


@router.post("/forgot-password", response_model=ResponseBase)
async def forgot_password(payload: ForgotPasswordRequest) -> ResponseBase:
    message, _ = auth_service.forgot_password(payload)
    return JSONResponse({"status": True, "message": message})


@router.post("/reset-password", response_model=ResponseBase)
async def reset_password(payload: ResetPasswordRequest) -> ResponseBase:
    auth_service.reset_password(payload)
    return ResponseBase(status=True, message="Password reset successfully", data={})


@router.get("/me", response_model=ResponseBase)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
) -> ResponseBase:
    message, data = auth_service.get_current_user_info(current_user)
    return ResponseBase(status=True, message=message, data=data)
