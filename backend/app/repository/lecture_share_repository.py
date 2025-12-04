"""Repository helpers for recording lecture shares to students."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ..postgres import get_pg_cursor


def _ensure_table_exists() -> None:
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS lecture_shares (
            id SERIAL PRIMARY KEY,
            lecture_id TEXT NOT NULL,
            student_enrollment TEXT NOT NULL,
            std TEXT,
            division TEXT,
            subject TEXT,
            shared_by TEXT,
            share_message TEXT,
            shared_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_lecture_shares_lecture
        ON lecture_shares (lecture_id)
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(create_table_sql)
        cur.execute(create_index_sql)


_ensure_table_exists()


def record_shares(
    *,
    lecture_id: str,
    std: Optional[str],
    subject: Optional[str],
    division: Optional[str],
    shared_by: Optional[str],
    share_message: Optional[str],
    enrollments: Iterable[str],
) -> int:
    """Persist share records for each enrollment. Returns number of rows inserted."""

    rows: List[Dict[str, Optional[str]]] = []
    for enrollment in enrollments:
        enrollment_number = (enrollment or "").strip()
        if not enrollment_number:
            continue
        rows.append(
            {
                "lecture_id": lecture_id,
                "student_enrollment": enrollment_number,
                "std": std,
                "division": division,
                "subject": subject,
                "shared_by": shared_by,
                "share_message": share_message,
            }
        )

    if not rows:
        return 0

    insert_sql = """
        INSERT INTO lecture_shares (
            lecture_id,
            student_enrollment,
            std,
            division,
            subject,
            shared_by,
            share_message
        ) VALUES (
            %(lecture_id)s,
            %(student_enrollment)s,
            %(std)s,
            %(division)s,
            %(subject)s,
            %(shared_by)s,
            %(share_message)s
        )
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.executemany(insert_sql, rows)

    return len(rows)


def list_shares_for_lecture(lecture_id: str) -> List[Dict[str, object]]:
    """Return all share records for a lecture."""

    query = """
        SELECT
            id,
            lecture_id,
            student_enrollment,
            std,
            division,
            subject,
            shared_by,
            share_message,
            shared_at
        FROM lecture_shares
        WHERE lecture_id = %(lecture_id)s
        ORDER BY shared_at DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, {"lecture_id": lecture_id})
        return cur.fetchall()

def delete_shares_for_lecture(*, lecture_id: str) -> int:
    """Delete share records for a lecture and return removed row count."""

    query = """
        DELETE FROM lecture_shares
        WHERE lecture_id = %(lecture_id)s
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"lecture_id": lecture_id})
        return cur.rowcount or 0


def list_shared_lectures(
    *,
    admin_id: int,
    std: Optional[str] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Return lectures shared for the given admin with optional filters."""

    filters: List[str] = ["lg.lecture_shared IS TRUE"]
    params: Dict[str, object] = {"admin_id": admin_id}

    if std:
        filters.append("LOWER(lg.std) = LOWER(%(std)s)")
        params["std"] = std

    if subject:
        filters.append("LOWER(lg.subject) = LOWER(%(subject)s)")
        params["subject"] = subject

    where_clause = " AND ".join(["lg.admin_id = %(admin_id)s"] + filters)

    query = f"""
        SELECT
            lg.lecture_uid AS lecture_id,
            lg.lecture_title AS title,
            lg.std,
            lg.subject,
            lg.lecture_link AS lecture_url,
            MAX(ls.shared_at) AS last_shared_at,
            lg.updated_at AS lecture_updated_at
        FROM lecture_gen lg
        LEFT JOIN lecture_shares ls
            ON ls.lecture_id = lg.lecture_uid
        WHERE {where_clause}
        GROUP BY lg.lecture_uid, lg.lecture_title, lg.std, lg.subject, lg.lecture_link, lg.updated_at
        ORDER BY last_shared_at DESC NULLS LAST, lecture_updated_at DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()