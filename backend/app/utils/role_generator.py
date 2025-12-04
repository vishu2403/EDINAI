"""Role ID generator utilities using psycopg connection."""
from __future__ import annotations

from typing import Literal

from ..postgres import get_pg_cursor

RoleWorkType = Literal["chapter", "student", "lecture"]

_PREFIX_MAP: dict[RoleWorkType, str] = {
    "chapter": "CM",
    "student": "SM",
    "lecture": "LM",
}


def _ensure_role_sequences_table() -> None:
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS role_sequences (
                work_type VARCHAR(32) PRIMARY KEY,
                last_sequence INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            "ALTER TABLE role_sequences ALTER COLUMN last_sequence SET NOT NULL"
        )
        cur.execute(
            "ALTER TABLE role_sequences ALTER COLUMN last_sequence SET DEFAULT 0"
        )


_ensure_role_sequences_table()


def _ensure_valid_work_type(work_type: RoleWorkType) -> str:
    if work_type not in _PREFIX_MAP:
        raise ValueError(f"Unsupported work type: {work_type}")
    return work_type


def generate_role_id(work_type: RoleWorkType) -> str:
    work_type_value = _ensure_valid_work_type(work_type)
    prefix = _PREFIX_MAP[work_type_value]

    query = (
        "INSERT INTO role_sequences (work_type, last_sequence) "
        "VALUES (%(work_type)s, 1) "
        "ON CONFLICT (work_type) DO UPDATE SET last_sequence = role_sequences.last_sequence + 1 "
        "RETURNING last_sequence"
    )

    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"work_type": work_type_value})
        (sequence,) = cur.fetchone()

    return f"{prefix}{sequence:03d}"


def get_next_role_id_preview(work_type: RoleWorkType) -> str:
    work_type_value = _ensure_valid_work_type(work_type)
    prefix = _PREFIX_MAP[work_type_value]

    query = "SELECT last_sequence FROM role_sequences WHERE work_type = %(work_type)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"work_type": work_type_value})
        row = cur.fetchone()

    next_sequence = (row[0] if row else 0) + 1
    return f"{prefix}{next_sequence:03d}"


def reset_role_sequence(work_type: RoleWorkType) -> bool:
    work_type_value = _ensure_valid_work_type(work_type)

    query = "UPDATE role_sequences SET last_sequence = 0 WHERE work_type = %(work_type)s"
    with get_pg_cursor(dict_rows=False) as cur:
        cur.execute(query, {"work_type": work_type_value})
        return cur.rowcount > 0