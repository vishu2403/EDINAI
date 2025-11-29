"""Database helpers for student roster uploads."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor


def _table_exists(table_name: str) -> bool:
    query = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %(table_name)s
        LIMIT 1
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"table_name": table_name})
        return cur.fetchone() is not None


def fetch_existing_enrollments(admin_id: int, enrollments: Iterable[str]) -> List[str]:
    """Return enrollment numbers that already exist for the admin."""

    enrollment_list = list(enrollments)
    if not enrollment_list:
        return []

    query = (
        "SELECT enrollment_number FROM student_roster_entries "
        "WHERE admin_id = %(admin_id)s AND enrollment_number = ANY(%(enrollments)s)"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id, "enrollments": enrollment_list})
        rows = cur.fetchall()

    return [row["enrollment_number"] for row in rows]


def insert_roster_entries(admin_id: int, entries: List[Dict[str, Any]]) -> None:
    """Insert roster entries for an admin."""

    if not entries:
        return

    insert_sql = (
        "INSERT INTO student_roster_entries "
        "(admin_id, enrollment_number, first_name, last_name, std, division, auto_password) "
        "VALUES (%(admin_id)s, %(enrollment_number)s, %(first_name)s, %(last_name)s, %(std)s, %(division)s, %(auto_password)s)"
    )

    payload = [
        {
            "admin_id": admin_id,
            "enrollment_number": entry["enrollment_number"],
            "first_name": entry["first_name"],
            "last_name": entry.get("last_name"),
            "std": entry["std"],
            "division": entry.get("division"),
            "auto_password": entry["auto_password"],
        }
        for entry in entries
    ]

    with get_pg_cursor(dict_rows=False) as cur:
        cur.executemany(insert_sql, payload)


def delete_roster_entry(admin_id: int, enrollment_number: str) -> bool:
    """Delete a single roster entry for the admin."""

    query = (
        "DELETE FROM student_roster_entries "
        "WHERE admin_id = %(admin_id)s AND enrollment_number = %(enrollment_number)s "
        "RETURNING enrollment_number"
    )

    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id, "enrollment_number": enrollment_number})
        return cur.fetchone() is not None


def fetch_roster_entries(admin_id: int) -> List[Dict[str, Any]]:
    """Return stored roster entries for an admin ordered by latest."""

    query = """
        SELECT enrollment_number, first_name, last_name, std, division, auto_password, created_at
        FROM student_roster_entries
        WHERE admin_id = %(admin_id)s
        ORDER BY created_at DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"admin_id": admin_id})
        return cur.fetchall()


def fetch_student_profiles(
    admin_id: int,
    *,
    class_filter: Optional[str] = None,
    division_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return roster entries enriched with profile details for the admin."""

    purchases_available = _table_exists("purchases")
    params: Dict[str, Any] = {"admin_id": admin_id}
    conditions = ["r.admin_id = %(admin_id)s"]

    if class_filter:
        params["class_filter"] = class_filter
        conditions.append("(r.std = %(class_filter)s OR p.class_stream = %(class_filter)s)")

    if division_filter:
        params["division_filter"] = division_filter
        conditions.append("(r.division = %(division_filter)s OR p.division = %(division_filter)s)")

    where_clause = " AND ".join(conditions)

    if purchases_available:
        query = """
            SELECT
                r.enrollment_number,
                r.first_name        AS roster_first_name,
                r.last_name         AS roster_last_name,
                r.std,
                r.division          AS roster_division,
                r.auto_password,
                r.created_at        AS roster_created_at,
                p.id                AS profile_id,
                p.first_name        AS profile_first_name,
                p.father_name,
                p.class_stream,
                p.division          AS profile_division,
                p.class_head,
                p.mobile_number,
                p.parents_number,
                p.email,
                p.photo_path,
                p.created_at        AS profile_created_at,
                (p.id IS NOT NULL)  AS profile_complete,
                COALESCE(
                    json_agg(
                        jsonb_build_object(
                            'transaction_id', pur.transaction_id,
                            'date', pur.date,
                            'time', pur.time,
                            'payment_method', pur.payment_method,
                            'transaction_type', pur.transaction_type,
                            'description', pur.description,
                            'amount', pur.amount,
                            'status', pur.status
                        )
                        ORDER BY pur.date DESC, pur.time DESC
                    ) FILTER (WHERE pur.transaction_id IS NOT NULL),
                    '[]'::json
                )                 AS purchases
            FROM student_roster_entries r
            LEFT JOIN student_profiles p ON p.enrollment_number = r.enrollment_number
            LEFT JOIN students s ON s.enrollment_number = r.enrollment_number
            LEFT JOIN purchases pur ON pur.sid = s.sid
            WHERE {where_clause}
            GROUP BY
                r.id,
                r.enrollment_number,
                r.first_name,
                r.last_name,
                r.std,
                r.division,
                r.auto_password,
                r.created_at,
                p.id,
                p.first_name,
                p.father_name,
                p.class_stream,
                p.division,
                p.class_head,
                p.mobile_number,
                p.parents_number,
                p.email,
                p.photo_path,
                p.created_at
            ORDER BY r.created_at DESC
        """
    else:
        query = """
            SELECT
                r.enrollment_number,
                r.first_name        AS roster_first_name,
                r.last_name         AS roster_last_name,
                r.std,
                r.division          AS roster_division,
                r.auto_password,
                r.created_at        AS roster_created_at,
                p.id                AS profile_id,
                p.first_name        AS profile_first_name,
                p.father_name,
                p.class_stream,
                p.division          AS profile_division,
                p.class_head,
                p.mobile_number,
                p.parents_number,
                p.email,
                p.photo_path,
                p.created_at        AS profile_created_at,
                (p.id IS NOT NULL)  AS profile_complete,
                '[]'::json          AS purchases
            FROM student_roster_entries r
            LEFT JOIN student_profiles p ON p.enrollment_number = r.enrollment_number
            WHERE {where_clause}
        """

    with get_pg_cursor() as cur:
        cur.execute(query.format(where_clause=where_clause), params)
        rows = cur.fetchall()

    for row in rows:
        purchases = row.get("purchases")
        if isinstance(purchases, str):
            try:
                row["purchases"] = json.loads(purchases)
            except json.JSONDecodeError:
                row["purchases"] = []
        elif purchases is None:
            row["purchases"] = []

    return rows
