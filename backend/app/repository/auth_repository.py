"""Psycopg helpers for authentication workflows (ported)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..postgres import get_pg_cursor

_ADMIN_COLUMNS = [
    "admin_id",
    "name",
    "email",
    "password",
    "package",
    "start_date",
    "expiry_date",
    "has_inai_credentials",
    "active",
    "is_super_admin",
    "created_at",
    "updated_at",
    "last_login",
]

_MEMBER_COLUMNS = [
    "member_id",
    "admin_id",
    "name",
    "designation",
    "email",
    "phone_number",
    "work_type",
    "password",
    "role_id",
    "active",
    "created_at",
    "last_login",
]


def _row_to_admin(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {key: row.get(key) for key in _ADMIN_COLUMNS}


def _row_to_member(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {key: row.get(key) for key in _MEMBER_COLUMNS}


def fetch_admin_by_email(email: str) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM admins "
        "WHERE LOWER(email) = LOWER(%(email)s) LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"email": email})
        row = cur.fetchone()
    return _row_to_admin(row)


def get_admin_by_id(admin_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM admins "
        "WHERE admin_id = %(admin_id)s LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        row = cur.fetchone()
    return _row_to_admin(row)


def fetch_member_by_email(email: str) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_MEMBER_COLUMNS)} FROM members "
        "WHERE LOWER(email) = LOWER(%(email)s) LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"email": email})
        row = cur.fetchone()
    return _row_to_member(row)


def update_admin_last_login(admin_id: int, when: datetime) -> None:
    query = "UPDATE admins SET last_login = %(when)s WHERE admin_id = %(admin_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"when": when, "admin_id": admin_id})


def update_member_last_login(member_id: int, when: datetime) -> None:
    query = "UPDATE members SET last_login = %(when)s WHERE member_id = %(member_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"when": when, "member_id": member_id})


def update_admin_password(
    admin_id: int,
    hashed_password: str,
    *,
    updated_at: Optional[datetime] = None,
) -> None:
    query = "UPDATE admins SET password = %(password)s, updated_at = %(updated_at)s WHERE admin_id = %(admin_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(
            query,
            {
                "password": hashed_password,
                "updated_at": updated_at or datetime.utcnow(),
                "admin_id": admin_id,
            },
        )


def update_member_password(member_id: int, hashed_password: str) -> None:
    query = "UPDATE members SET password = %(password)s WHERE member_id = %(member_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"password": hashed_password, "member_id": member_id})
