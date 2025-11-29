"""Database connection utilities for the modular example backend."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from psycopg import connect
from psycopg.rows import dict_row

from .config import settings


logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Iterator:
    """Yield a psycopg connection.

    Production code should switch to connection pooling; for this structure example
    we keep it minimal to highlight layering only.
    """

    conn = connect(settings.database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_pg_cursor(*, dict_rows: bool = True):
    """Yield a psycopg cursor with optional dict row factory."""

    with get_connection() as conn:
        cursor_factory = dict_row if dict_rows else None
        cur = None
        try:
            cur = conn.cursor(row_factory=cursor_factory)
            yield cur
        except Exception as exc:
            logger.exception("PostgreSQL cursor operation failed")
            raise
        finally:
            if cur is not None:
                cur.close()
