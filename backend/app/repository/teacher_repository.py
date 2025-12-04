"""Repository helpers for teacher records (in-memory example)."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..models import Teacher

_TEACHERS: Dict[int, Teacher] = {}
_TEACHER_SEQUENCE = 0


def _next_id() -> int:
    global _TEACHER_SEQUENCE
    _TEACHER_SEQUENCE += 1
    return _TEACHER_SEQUENCE


async def create_teacher(*, user_id: int, subject: str, experience_years: int | None) -> Teacher:
    teacher = Teacher(
        id=_next_id(),
        user_id=user_id,
        subject=subject,
        experience_years=experience_years,
        created_at=datetime.utcnow(),
    )
    _TEACHERS[teacher.id] = teacher
    return teacher


async def list_teachers() -> List[Teacher]:
    return list(_TEACHERS.values())


async def get_teacher(teacher_id: int) -> Optional[Teacher]:
    return _TEACHERS.get(teacher_id)
