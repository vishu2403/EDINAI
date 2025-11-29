"""Admin portal repository helpers wrapping member repository."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from . import member_repository


def _normalize_member(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None

    normalized = dict(row)
    normalized.setdefault("admin_id", normalized.get("admin_id"))
    normalized.setdefault("active", normalized.get("active", True))
    return normalized


def list_portal_members(
    admin_id: int,
    *,
    work_type: Optional[str] = None,
    active_only: bool = False,
) -> List[Dict[str, Any]]:
    members = member_repository.list_members(
        admin_id,
        work_type=work_type,
        active_only=active_only,
    )
    return [_normalize_member(member) or {} for member in members]


def create_portal_member(**fields: Any) -> Dict[str, Any]:
    member = member_repository.create_member(**fields)
    normalized = _normalize_member(member)
    if not normalized:
        raise RuntimeError("Failed to create portal member")
    return normalized


def get_portal_member_by_email(email: str, *, admin_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    member = member_repository.get_member_by_email(email, admin_id=admin_id)
    return _normalize_member(member)


def get_portal_member_by_id(member_id: int) -> Optional[Dict[str, Any]]:
    member = member_repository.get_member_by_id(member_id)
    return _normalize_member(member)


def member_exists_for_admin(*, email: str, admin_id: int) -> bool:
    return member_repository.member_exists(email, admin_id=admin_id)


def update_portal_member_last_login(admin_id: int, member_id: int, when: datetime) -> Optional[Dict[str, Any]]:
    updated = member_repository.update_member(member_id, last_login=when)
    return _normalize_member(updated)
