"""Domain model for student records."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class Student:
    """Represents a student enrolled through the portal."""

    id: int
    user_id: int
    enrollment_no: str
    grade: str
    section: Optional[str]
    created_at: datetime
