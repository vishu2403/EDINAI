"""Raw PostgreSQL helpers for admin management (admins table)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

_PACKAGE_COLUMNS = [
    "id",
    "name",
    "price",
    "duration_days",
    "video_limit",
    "max_quality",
    "max_minutes_per_lecture",
    "ai_videos_per_lecture",
    "topics_per_lecture",
    "extra_credit_price",
    "extra_ai_video_price",
    "discount_rate",
    "support_level",
    "features",
    "notes",
]


def _row_to_admin(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {key: row.get(key) for key in _ADMIN_COLUMNS}


def _rows_to_admins(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: row.get(key) for key in _ADMIN_COLUMNS} for row in rows]


def admin_exists_by_email(email: str, *, exclude_admin_id: Optional[int] = None) -> bool:
    query = "SELECT 1 FROM admins WHERE LOWER(email) = LOWER(%(email)s)"
    params: Dict[str, Any] = {"email": email}
    if exclude_admin_id is not None:
        query += " AND admin_id <> %(exclude_admin_id)s"
        params["exclude_admin_id"] = exclude_admin_id
    query += " LIMIT 1"

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
        return cur.fetchone() is not None


def create_admin(**fields: Any) -> Dict[str, Any]:
    columns = [
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
    ]
    if "admin_id" in fields:
        columns = ["admin_id", *columns]
    column_clause = ", ".join(columns)
    placeholders = ", ".join(f"%({column})s" for column in columns)

    query = (
        f"INSERT INTO admins ({column_clause}) "
        f"VALUES ({placeholders}) RETURNING {', '.join(_ADMIN_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, fields)
        row = cur.fetchone()

    admin = _row_to_admin(row)
    if admin is None:
        raise RuntimeError("Failed to create admin record")
    return admin


def list_admins(*, skip: int, limit: int, active_only: bool) -> Tuple[List[Dict[str, Any]], int]:
    base_condition = "WHERE active = TRUE" if active_only else ""

    list_query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM admins {base_condition} "
        "ORDER BY admin_id OFFSET %(skip)s LIMIT %(limit)s"
    )
    count_query = f"SELECT COUNT(*) FROM admins {base_condition}"

    params = {"skip": skip, "limit": limit}

    with get_pg_cursor() as cur:
        cur.execute(list_query, params)
        rows = cur.fetchall()

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(count_query)
        (total_count,) = cur.fetchone()

    return _rows_to_admins(rows), int(total_count)


def get_admin_by_id(admin_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM admins "
        "WHERE admin_id = %(admin_id)s"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        row = cur.fetchone()
    return _row_to_admin(row)


def admin_exists_by_id(admin_id: int) -> bool:
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(
            "SELECT 1 FROM admins WHERE admin_id = %(admin_id)s LIMIT 1",
            {"admin_id": admin_id},
        )
        return cur.fetchone() is not None


def update_admin(admin_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_admin_by_id(admin_id)

    allowed = {
        "name",
        "email",
        "package",
        "active",
        "expiry_date",
        "start_date",
        "has_inai_credentials",
        "is_super_admin",
        "password",
        "updated_at",
        "last_login",
    }
    invalid = set(fields.keys()) - allowed
    if invalid:
        raise ValueError(f"Invalid admin columns: {', '.join(invalid)}")

    assignments = [f"{column} = %({column})s" for column in fields.keys()]
    query = (
        f"UPDATE admins SET {', '.join(assignments)} "
        "WHERE admin_id = %(admin_id)s RETURNING " + ", ".join(_ADMIN_COLUMNS)
    )

    params = {"admin_id": admin_id, **fields}
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    return _row_to_admin(row)


def delete_admin(admin_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"DELETE FROM admins WHERE admin_id = %(admin_id)s RETURNING {', '.join(_ADMIN_COLUMNS)}"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        row = cur.fetchone()
    return _row_to_admin(row)


def extend_admin_subscription(admin_id: int, days: int) -> Optional[Dict[str, Any]]:
    query = (
        "UPDATE admins SET expiry_date = expiry_date + %(days)s * INTERVAL '1 day', updated_at = %(now)s "
        "WHERE admin_id = %(admin_id)s RETURNING " + ", ".join(_ADMIN_COLUMNS)
    )
    now = datetime.utcnow()
    params = {"days": days, "admin_id": admin_id, "now": now}

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()

    admin = _row_to_admin(row)
    if admin and admin["active"] is False and admin["expiry_date"] and admin["expiry_date"] >= now:
        admin = update_admin(admin_id, active=True)
    return admin


def list_all_admins() -> List[Dict[str, Any]]:
    query = f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM admins ORDER BY admin_id"
    with get_pg_cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return _rows_to_admins(rows)


def get_package_by_name(name: str) -> Optional[Dict[str, Any]]:
    query = f"SELECT {', '.join(_PACKAGE_COLUMNS)} FROM packages WHERE name = %(name)s"
    with get_pg_cursor() as cur:
        cur.execute(query, {"name": name})
        row = cur.fetchone()
    if row is None:
        return None
    return {key: row.get(key) for key in _PACKAGE_COLUMNS}


def package_exists(name: str) -> bool:
    query = "SELECT 1 FROM packages WHERE name = %(name)s LIMIT 1"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"name": name})
        return cur.fetchone() is not None


def is_super_admin(admin_id: int) -> bool:
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(
            "SELECT is_super_admin FROM admins WHERE admin_id = %(admin_id)s LIMIT 1",
            {"admin_id": admin_id},
        )
        row = cur.fetchone()
    return bool(row[0]) if row else False


def list_super_admin_ids() -> List[int]:
    query = "SELECT admin_id FROM admins WHERE is_super_admin = TRUE"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query)
        rows = cur.fetchall()

    admin_ids: List[int] = []
    for row in rows:
        admin_id = row[0] if isinstance(row, tuple) else row.get("admin_id")  # type: ignore[index]
        if admin_id is not None:
            admin_ids.append(int(admin_id))
    return admin_ids
