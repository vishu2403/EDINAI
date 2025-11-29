"""Repository helpers for student records (in-memory example)."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..models import Student

_STUDENTS: Dict[int, Student] = {}
_STUDENT_SEQUENCE = 0


def _next_id() -> int:
    global _STUDENT_SEQUENCE
    _STUDENT_SEQUENCE += 1
    return _STUDENT_SEQUENCE


async def create_student(*, user_id: int, enrollment_no: str, grade: str, section: str | None) -> Student:
    student = Student(
        id=_next_id(),
        user_id=user_id,
        enrollment_no=enrollment_no,
        grade=grade,
        section=section,
        created_at=datetime.utcnow(),
    )
    _STUDENTS[student.id] = student
    return student


async def list_students() -> List[Student]:
    return list(_STUDENTS.values())


async def get_student(student_id: int) -> Optional[Student]:
    return _STUDENTS.get(student_id)
