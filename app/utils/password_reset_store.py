"""In-memory store for password reset tokens."""
from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, Tuple

_ResetTokenEntry = Tuple[str, datetime]

_reset_tokens: Dict[str, _ResetTokenEntry] = {}
_lock = Lock()


def _purge_expired_locked() -> None:
    now = datetime.utcnow()
    expired_tokens = [token for token, (_, expires_at) in _reset_tokens.items() if expires_at <= now]
    for token in expired_tokens:
        _reset_tokens.pop(token, None)


def store_reset_token(email: str, token: str, expires_at: datetime) -> None:
    normalized_email = email.lower().strip()
    with _lock:
        _purge_expired_locked()
        _reset_tokens[token] = (normalized_email, expires_at)


def consume_reset_token(email: str, token: str) -> bool:
    normalized_email = email.lower().strip()
    with _lock:
        _purge_expired_locked()
        entry = _reset_tokens.get(token)
        if not entry:
            return False

        stored_email, expires_at = entry
        if stored_email != normalized_email or datetime.utcnow() > expires_at:
            _reset_tokens.pop(token, None)
            return False

        _reset_tokens.pop(token, None)
        return True


def purge_expired_tokens() -> None:
    with _lock:
        _purge_expired_locked()
