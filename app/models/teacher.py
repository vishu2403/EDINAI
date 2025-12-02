"""Domain model for teacher records."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(slots=True)
class Teacher:
    """Represents a teacher available inside the portal."""

    id: int
    user_id: int
    subject: str
    experience_years: Optional[int]
    created_at: datetime
