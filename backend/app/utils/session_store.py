"""In-memory session tracking shared by registration/contact modules."""
from __future__ import annotations

from typing import Dict

valid_tokens: Dict[str, str] = {}

__all__ = ["valid_tokens"]
