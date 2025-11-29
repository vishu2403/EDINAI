"""Utility exports for the modular backend."""
from __future__ import annotations

from . import auth, dependencies, role_generator, file_handler, student_portal_security

__all__ = [
    "auth",
    "dependencies",
    "role_generator",
    "file_handler",
    "student_portal_security",
]
