"""Domain model for application users."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class User:
    """Represents an authenticated platform user."""

    id: int
    name: str
    email: str
    role: str
    created_at: datetime
