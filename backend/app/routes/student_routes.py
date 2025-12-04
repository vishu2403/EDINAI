"""Student routes for the modular example backend."""
from __future__ import annotations

from fastapi import APIRouter, status

from ..schemas import ResponseEnvelope, StudentCreate
from ..services import student_service

router = APIRouter(prefix="/students", tags=["Students"])


@router.post("/", response_model=ResponseEnvelope, status_code=status.HTTP_201_CREATED)
async def create_student(payload: StudentCreate) -> ResponseEnvelope:
    student = await student_service.register_student(payload)
    return ResponseEnvelope(status=True, message="Student created", data=student)


@router.get("/", response_model=ResponseEnvelope)
async def list_students() -> ResponseEnvelope:
    students = await student_service.list_students()
    return ResponseEnvelope(status=True, message="Students fetched", data=students)
