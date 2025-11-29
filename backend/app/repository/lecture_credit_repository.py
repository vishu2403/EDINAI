"""Repository helpers for tracking lecture credit usage per admin."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from ..postgres import get_pg_cursor

_TABLE_NAME = "admin_lecture_usage"


def _ensure_table_exists() -> None:
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
            admin_id INTEGER PRIMARY KEY,
            credits_used INTEGER NOT NULL DEFAULT 0,
            overflow_attempts INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(create_table_sql)


_ensure_table_exists()


def get_usage(admin_id: int) -> Dict[str, int]:
    query = f"SELECT credits_used, overflow_attempts FROM {_TABLE_NAME} WHERE admin_id = %(admin_id)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"admin_id": admin_id})
        row = cur.fetchone()

    if not row:
        return {"credits_used": 0, "overflow_attempts": 0}

    credits_used, overflow_attempts = row
    return {
        "credits_used": int(credits_used or 0),
        "overflow_attempts": int(overflow_attempts or 0),
    }


def upsert_usage(admin_id: int, *, credits_delta: int = 0, overflow_delta: int = 0) -> None:
    if credits_delta == 0 and overflow_delta == 0:
        return

    query = f"""
        INSERT INTO {_TABLE_NAME} (admin_id, credits_used, overflow_attempts, updated_at)
        VALUES (%(admin_id)s, %(credits_used)s, %(overflow_attempts)s, %(updated_at)s)
        ON CONFLICT (admin_id)
        DO UPDATE SET
            credits_used = {_TABLE_NAME}.credits_used + %(credits_used)s,
            overflow_attempts = {_TABLE_NAME}.overflow_attempts + %(overflow_attempts)s,
            updated_at = %(updated_at)s
    """

    params = {
        "admin_id": admin_id,
        "credits_used": credits_delta,
        "overflow_attempts": overflow_delta,
        "updated_at": datetime.utcnow(),
    }

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)


def reset_usage(admin_id: int, *, credits_used: Optional[int] = None, overflow_attempts: Optional[int] = None) -> None:
    """Force-set credit usage values. Mainly for administrative tools/tests."""

    current = get_usage(admin_id)
    new_credits = credits_used if credits_used is not None else current["credits_used"]
    new_overflow = overflow_attempts if overflow_attempts is not None else current["overflow_attempts"]

    query = f"""
        INSERT INTO {_TABLE_NAME} (admin_id, credits_used, overflow_attempts, updated_at)
        VALUES (%(admin_id)s, %(credits_used)s, %(overflow_attempts)s, %(updated_at)s)
        ON CONFLICT (admin_id)
        DO UPDATE SET
            credits_used = EXCLUDED.credits_used,
            overflow_attempts = EXCLUDED.overflow_attempts,
            updated_at = EXCLUDED.updated_at
    """

    params = {
        "admin_id": admin_id,
        "credits_used": new_credits,
        "overflow_attempts": new_overflow,
        "updated_at": datetime.utcnow(),
    }

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, params)
