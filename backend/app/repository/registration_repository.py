"""Raw PostgreSQL helpers for administrator registration and authentication."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..postgres import get_pg_cursor
from . import admin_portal_repository

_ADMIN_COLUMNS = [
    "admin_aid",
    "full_name",
    "email",
    "password",
    "package_plan",
    "validity",
    "inai_email",
    "inai_password_encrypted",
    "refresh_token",
    "token_version",
    "is_logged_in",
    "created_at",
    "updated_at",
]


def _ensure_table_exists() -> None:
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS administrators (
            admin_aid SERIAL PRIMARY KEY,
            full_name VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password TEXT NOT NULL,
            package_plan VARCHAR(50) NOT NULL,
            validity VARCHAR(50) NOT NULL,
            inai_email VARCHAR(255),
            inai_password_encrypted TEXT,
            refresh_token TEXT,
            token_version INTEGER DEFAULT 0,
            is_logged_in BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """

    create_index_sql = (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_administrators_email_lower "
        "ON administrators (LOWER(email))"
    )

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(create_table_sql)
        cur.execute(create_index_sql)


_ensure_table_exists()


def _row_to_admin(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {key: row.get(key) for key in _ADMIN_COLUMNS}


def admin_exists_by_email(email: str) -> bool:
    query = "SELECT 1 FROM administrators WHERE LOWER(email) = LOWER(%(email)s) LIMIT 1"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"email": email})
        return cur.fetchone() is not None


def create_admin(**fields: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    fields = {
        **fields,
        "created_at": now,
        "updated_at": now,
    }

    columns = [
        "full_name",
        "email",
        "password",
        "package_plan",
        "validity",
        "created_at",
        "updated_at",
    ]
    column_clause = ", ".join(columns)
    placeholders = ", ".join(f"%({column})s" for column in columns)

    query = (
        f"INSERT INTO administrators ({column_clause}) "
        f"VALUES ({placeholders}) RETURNING {', '.join(_ADMIN_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, fields)
        row = cur.fetchone()

    admin = _row_to_admin(row)
    if admin is None:
        raise RuntimeError("Failed to create administrator")
    return admin


def get_admin_by_email(email: str) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM administrators "
        "WHERE LOWER(email) = LOWER(%(email)s) LIMIT 1"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"email": email})
        row = cur.fetchone()

    return _row_to_admin(row)


def get_admin_by_id(admin_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM administrators "
        "WHERE admin_aid = %(admin_id)s"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        row = cur.fetchone()

    return _row_to_admin(row)


def update_admin(admin_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_admin_by_id(admin_id)

    allowed = {
        "full_name",
        "email",
        "password",
        "package_plan",
        "validity",
        "inai_email",
        "inai_password_encrypted",
        "refresh_token",
        "token_version",
        "is_logged_in",
        "updated_at",
    }

    invalid = set(fields.keys()) - allowed
    if invalid:
        raise ValueError(f"Invalid administrator columns: {', '.join(invalid)}")

    assignments = [f"{column} = %({column})s" for column in fields.keys()]

    query = (
        f"UPDATE administrators SET {', '.join(assignments)} "
        "WHERE admin_aid = %(admin_id)s RETURNING "
        + ", ".join(_ADMIN_COLUMNS)
    )

    params = {"admin_id": admin_id, **fields}

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()

    return _row_to_admin(row)


def update_admin_last_login(admin_id: int, when: datetime) -> Optional[Dict[str, Any]]:
    return update_admin(admin_id, updated_at=when, is_logged_in=1)


def get_portal_member_by_email(email: str) -> Optional[Dict[str, Any]]:
    return admin_portal_repository.get_portal_member_by_email(email)


def update_member_last_login(*, admin_id: int, member_id: int, when: datetime) -> Optional[Dict[str, Any]]:
    return admin_portal_repository.update_portal_member_last_login(admin_id, member_id, when)


def update_admin_password(
    admin_id: int,
    hashed_password: str,
    *,
    updated_at: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Update an administrator's password and timestamp."""

    return update_admin(
        admin_id,
        password=hashed_password,
        updated_at=updated_at or datetime.now(timezone.utc),
    )
