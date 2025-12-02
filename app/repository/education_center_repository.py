"""Education center persistence helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor

_COLUMNS: List[str] = [
    "id",
    "admin_id",
    "name",
    "upload_image",
    "center_photos",
    "logo",
    "other_activities",
    "created_at",
]


def _row_to_center(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {column: row.get(column) for column in _COLUMNS}


def get_center_by_admin(admin_id: int) -> Optional[Dict[str, Any]]:
    query = (
        f"SELECT {', '.join(_COLUMNS)} FROM education_centers "
        "WHERE admin_id = %(admin_id)s ORDER BY id DESC LIMIT 1"
    )
    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        row = cur.fetchone()
    return _row_to_center(row)


def create_center(**fields: Any) -> Dict[str, Any]:
    columns = [
        "admin_id",
        "name",
        "upload_image",
        "center_photos",
        "logo",
        "other_activities",
        "created_at",
    ]
    payload = {**fields}
    payload.setdefault("created_at", datetime.utcnow())

    column_clause = ", ".join(columns)
    placeholders = ", ".join(f"%({column})s" for column in columns)

    query = (
        f"INSERT INTO education_centers ({column_clause}) "
        f"VALUES ({placeholders}) RETURNING {', '.join(_COLUMNS)}"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, payload)
        row = cur.fetchone()

    center = _row_to_center(row)
    if center is None:
        raise RuntimeError("Failed to create education center")
    return center


def update_center(center_id: int, admin_id: int, **fields: Any) -> Optional[Dict[str, Any]]:
    if not fields:
        return get_center_by_admin(admin_id)

    allowed = {"name", "upload_image", "center_photos", "logo", "other_activities"}
    invalid = set(fields.keys()) - allowed
    if invalid:
        raise ValueError(f"Invalid education center columns: {', '.join(invalid)}")

    assignments = [f"{column} = %({column})s" for column in fields.keys()]
    params = {"center_id": center_id, "admin_id": admin_id, **fields}

    query = (
        f"UPDATE education_centers SET {', '.join(assignments)} "
        "WHERE id = %(center_id)s AND admin_id = %(admin_id)s RETURNING "
        + ", ".join(_COLUMNS)
    )

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()

    return _row_to_center(row)
