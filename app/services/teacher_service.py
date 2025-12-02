"""Business logic for teacher operations."""
from __future__ import annotations

from dataclasses import asdict
from typing import List

from ..repository import teacher_repository
from ..schemas import TeacherCreate, TeacherResponse


async def register_teacher(payload: TeacherCreate) -> TeacherResponse:
    teacher = await teacher_repository.create_teacher(**payload.model_dump())
    return TeacherResponse(**asdict(teacher))


async def list_teachers() -> List[TeacherResponse]:
    teachers = await teacher_repository.list_teachers()
    return [TeacherResponse(**asdict(teacher)) for teacher in teachers]
