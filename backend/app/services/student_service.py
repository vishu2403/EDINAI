"""Business logic for student operations."""
from __future__ import annotations

from dataclasses import asdict
from typing import List

from ..repository import student_repository
from ..schemas import StudentCreate, StudentResponse


async def register_student(payload: StudentCreate) -> StudentResponse:
    student = await student_repository.create_student(**payload.model_dump())
    return StudentResponse(**asdict(student))


async def list_students() -> List[StudentResponse]:
    students = await student_repository.list_students()
    return [StudentResponse(**asdict(student)) for student in students]
