"""Common response envelope used by API endpoints."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ResponseEnvelope(BaseModel):
    """Standard response wrapper for consistent frontend handling."""

    status: bool
    message: str
    data: Optional[Any] = None


class ResponseBase(BaseModel):
    """Legacy-compatible response envelope (status/message/data)."""

    status: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Consistent error payload returned by exception handlers."""

    status: bool = False
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
