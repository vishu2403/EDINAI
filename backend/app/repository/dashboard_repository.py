"""Raw PostgreSQL helpers for dashboard metrics (ported from legacy app.dashboard)."""
from __future__ import annotations
from psycopg.errors import UndefinedColumn, UndefinedTable
from datetime import datetime
from typing import Any, Dict, Optional

from ..postgres import get_pg_cursor


_ADMIN_COLUMNS = [
    "admin_id",
    "name",
    "email",
    "package",
    "start_date",
    "expiry_date",
    "has_inai_credentials",
    "active",
    "is_super_admin",
    "created_at",
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


def _row_to_dict(row: Optional[Dict[str, Any]], columns: list[str]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {column: row.get(column) for column in columns}


def fetch_admin(admin_id: int) -> Optional[Dict[str, Any]]:
    # Try legacy admins table first
    query = (
        f"SELECT {', '.join(_ADMIN_COLUMNS)} FROM admins "
        "WHERE admin_id = %(admin_id)s"
    )
    with get_pg_cursor() as cur:
        try:
            cur.execute(query, {"admin_id": admin_id})
            row = cur.fetchone()
            if row:
                return _row_to_dict(row, _ADMIN_COLUMNS)
        except (UndefinedTable, UndefinedColumn):
            pass
    
    # Fall back to portal administrators table
    portal_columns = [
        "admin_aid",
        "full_name",
        "email",
        "package_plan",
        "validity",
        "inai_email",
        "inai_password_encrypted",
        "created_at",
        "updated_at",
    ]
    query = (
        f"SELECT {', '.join(portal_columns)} FROM administrators "
        "WHERE admin_aid = %(admin_id)s"
    )
    with get_pg_cursor() as cur:
        try:
            cur.execute(query, {"admin_id": admin_id})
            row = cur.fetchone()
        except (UndefinedTable, UndefinedColumn):
            return None
    
    if not row:
        return None
    
    # Map portal admin columns to legacy admin columns for compatibility
    return {
        "admin_id": row.get("admin_aid"),
        "name": row.get("full_name"),
        "email": row.get("email"),
        "package": row.get("package_plan"),
        "start_date": None,
        "expiry_date": None,
        "has_inai_credentials": bool(row.get("inai_email") and row.get("inai_password_encrypted")),
        "active": True,
        "is_super_admin": False,
        "created_at": row.get("created_at"),
        "last_login": None,
    }


def fetch_package(name: str) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_PACKAGE_COLUMNS)} FROM packages "
        "WHERE name = %(name)s"
    )
    with get_pg_cursor() as cur:
        try:
            cur.execute(query, {"name": name})
        except (UndefinedTable, UndefinedColumn):
            return None
        row = cur.fetchone()
    return _row_to_dict(row, _PACKAGE_COLUMNS)


def fetch_member(member_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_MEMBER_COLUMNS)} FROM members "
        "WHERE member_id = %(member_id)s"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"member_id": member_id})
        row = cur.fetchone()
    return _row_to_dict(row, _MEMBER_COLUMNS)


def count_members(admin_id: int, *, work_type: Optional[str] = None, active_only: bool = False) -> int:
    conditions = ["admin_id = %(admin_id)s"]
    params: Dict[str, Any] = {"admin_id": admin_id}

    if work_type:
        conditions.append("work_type = %(work_type)s")
        params["work_type"] = work_type

    if active_only:
        conditions.append("active = TRUE")

    query = "SELECT COUNT(*) FROM members WHERE " + " AND ".join(conditions)

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
        (count,) = cur.fetchone()
    return int(count)


def count_members_by_work_type(admin_id: int, *, active_only: bool = True) -> Dict[str, int]:
    params = {"admin_id": admin_id}
    query = (
        "SELECT work_type, COUNT(*) FROM members "
        "WHERE admin_id = %(admin_id)s"
    )
    if active_only:
        query += " AND active = TRUE"
    query += " GROUP BY work_type"

    counts = {"chapter": 0, "student": 0, "lecture": 0}
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
        for work_type, total in cur.fetchall():
            counts[str(work_type)] = int(total)
    return counts


def count_members_since(
    admin_id: int,
    *,
    since: datetime,
    work_type: Optional[str] = None,
    field: str = "last_login",
) -> int:
    if field not in {"last_login", "created_at"}:
        raise ValueError(f"Unsupported field for temporal count: {field}")

    conditions = ["admin_id = %(admin_id)s", f"{field} >= %(since)s"]
    params: Dict[str, Any] = {"admin_id": admin_id, "since": since}

    if work_type:
        conditions.append("work_type = %(work_type)s")
        params["work_type"] = work_type

    query = "SELECT COUNT(*) FROM members WHERE " + " AND ".join(conditions)

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
        (count,) = cur.fetchone()
    return int(count)

def count_admin_lectures(admin_id: int) -> int:
    query = "SELECT COUNT(*) FROM student_portal_videos WHERE admin_id = %(admin_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"admin_id": admin_id})
        (count,) = cur.fetchone()
    return int(count)

def recent_member_activity(admin_id: int, since: datetime) -> Dict[str, int]:
    return {
        "recent_logins": count_members_since(admin_id, since=since, field="last_login"),
        "new_members": count_members_since(admin_id, since=since, field="created_at"),
    }

def count_total_lectures(admin_id: int) -> int:
    query = "SELECT COUNT(*) FROM lecture_gen WHERE admin_id = %(admin_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"admin_id": admin_id})
        (count,) = cur.fetchone()
    return int(count)
def count_played_lectures(admin_id: int) -> int:
    query = (
        "SELECT COUNT(*) FROM lecture_gen "
        "WHERE admin_id = %(admin_id)s "
        "AND CAST(COALESCE(lecture_data->>'play_count', '0') AS INTEGER) > 0"
    )
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"admin_id": admin_id})
        (count,) = cur.fetchone()
    return int(count)


def dashboard_summary(admin_id: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_members": count_members(admin_id, active_only=False),
        "active_members": count_members(admin_id, active_only=True),
        "total_lecture": count_admin_lectures(admin_id),
    }

    admin = fetch_admin(admin_id)
    if admin and admin.get("expiry_date"):
        days_until_expiry = (admin["expiry_date"] - datetime.utcnow()).days
    else:
        days_until_expiry = 0

    summary.update(
        {
            "days_until_expiry": days_until_expiry,
            "subscription_status": "active" if days_until_expiry > 0 else "expired",
        }
    )

    return summary
