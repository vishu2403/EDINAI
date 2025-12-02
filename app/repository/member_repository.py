"""Repository helpers for managing admin members (ported)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor

_MEMBER_COLUMNS: List[str] = [
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


def _ensure_members_table_schema() -> None:
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute("SELECT to_regclass('public.members')")
        exists_row = cur.fetchone()
        table_exists = bool(exists_row and exists_row[0])
        if not table_exists:
            return

        cur.execute(
            "ALTER TABLE public.members "
            "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()"
        )
        cur.execute(
            "ALTER TABLE public.members "
            "ALTER COLUMN created_at SET DEFAULT NOW()"
        )
        cur.execute(
            "ALTER TABLE public.members "
            "ADD COLUMN IF NOT EXISTS last_login TIMESTAMPTZ"
        )
        cur.execute(
            "UPDATE public.members SET created_at = NOW() WHERE created_at IS NULL"
        )


_ensure_members_table_schema()


def _row_to_member(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {column: row.get(column) for column in _MEMBER_COLUMNS}


def _rows_to_members(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{column: row.get(column) for column in _MEMBER_COLUMNS} for row in rows]


def member_exists(email: str, *, admin_id: int) -> bool:
    query = (
        "SELECT 1 FROM members WHERE LOWER(email) = LOWER(%(email)s) "
        "AND admin_id = %(admin_id)s LIMIT 1"
    )
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"email": email, "admin_id": admin_id})
        return cur.fetchone() is not None


def get_member_by_id(member_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_MEMBER_COLUMNS)} FROM members "
        "WHERE member_id = %(member_id)s"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"member_id": member_id})
        row = cur.fetchone()
    return _row_to_member(row)


def get_member_by_email(email: str, *, admin_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    conditions = ["LOWER(email) = LOWER(%(email)s)"]
    params: Dict[str, Any] = {"email": email}
    if admin_id is not None:
        conditions.append("admin_id = %(admin_id)s")
        params["admin_id"] = admin_id

    query = (
        f"SELECT {', '.join(_MEMBER_COLUMNS)} FROM members "
        "WHERE " + " AND ".join(conditions) + " LIMIT 1"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    return _row_to_member(row)


def create_member(**fields: Any) -> Dict[str, Any]:
    columns = [
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
    ]
    column_clause = ", ".join(columns)
    placeholders = ", ".join(f"%({column})s" for column in columns)

    query = (
        f"INSERT INTO members ({column_clause}) "
        f"VALUES ({placeholders}) RETURNING {', '.join(_MEMBER_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, fields)
        row = cur.fetchone()

    member = _row_to_member(row)
    if member is None:
        raise RuntimeError("Failed to create member record")
    return member


def list_members(
    admin_id: int,
    *,
    work_type: Optional[str] = None,
    active_only: bool = False,
) -> List[Dict[str, Any]]:
    conditions = ["admin_id = %(admin_id)s"]
    params: Dict[str, Any] = {"admin_id": admin_id}

    if work_type:
        conditions.append("work_type = %(work_type)s")
        params["work_type"] = work_type

    if active_only:
        conditions.append("active = TRUE")

    where_clause = " AND ".join(conditions)

    query = (
        f"SELECT {', '.join(_MEMBER_COLUMNS)} FROM members "
        f"WHERE {where_clause} ORDER BY created_at DESC"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return _rows_to_members(rows)


def update_member(member_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_member_by_id(member_id)

    allowed_columns = {
        "name",
        "designation",
        "phone_number",
        "work_type",
        "active",
        "password",
        "last_login",
        "email",
    }
    invalid = set(fields.keys()) - allowed_columns
    if invalid:
        raise ValueError(f"Invalid member columns: {', '.join(sorted(invalid))}")

    assignments = [f"{column} = %({column})s" for column in fields.keys()]
    params = {"member_id": member_id, **fields}

    query = (
        f"UPDATE members SET {', '.join(assignments)} "
        "WHERE member_id = %(member_id)s RETURNING "
        + ", ".join(_MEMBER_COLUMNS)
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()

    return _row_to_member(row)

def delete_member(member_id: int) -> bool:
    query = "DELETE FROM members WHERE member_id = %(member_id)s"

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"member_id": member_id})
        return cur.rowcount > 0
