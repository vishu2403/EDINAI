"""Teacher routes for the modular example backend."""
from __future__ import annotations

from fastapi import APIRouter, status

from ..schemas import ResponseEnvelope, TeacherCreate
from ..services import teacher_service

router = APIRouter(prefix="/teachers", tags=["Teachers"])


@router.post("/", response_model=ResponseEnvelope, status_code=status.HTTP_201_CREATED)
async def create_teacher(payload: TeacherCreate) -> ResponseEnvelope:
    teacher = await teacher_service.register_teacher(payload)
    return ResponseEnvelope(status=True, message="Teacher created", data=teacher)


@router.get("/", response_model=ResponseEnvelope)
async def list_teachers() -> ResponseEnvelope:
    teachers = await teacher_service.list_teachers()
    return ResponseEnvelope(status=True, message="Teachers fetched", data=teachers)
