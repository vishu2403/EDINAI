"""Registration API routes."""
from __future__ import annotations

from fastapi import APIRouter, status

from ..schemas.admin_portal_schema import AdminCreate, AdminRegisterResponse
from ..services import registration_service


router = APIRouter(prefix="/registration", tags=["Registration"])


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    response_model=AdminRegisterResponse,
)
async def register_admin(payload: AdminCreate) -> AdminRegisterResponse:
    return registration_service.create_admin(payload)
