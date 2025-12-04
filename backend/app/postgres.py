"""Database connection utilities supporting psycopg (v3) and psycopg2 (v2)."""
from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Iterator
from sqlalchemy.engine import make_url

logger = logging.getLogger(__name__)
# Try psycopg (psycopg3) first, fall back to psycopg2 if not installed.
try:
    # psycopg v3
    from psycopg import connect  # type: ignore
    from psycopg.rows import dict_row as _psycopg3_dict_row  # type: ignore
    _DRIVER = "psycopg"
    _USE_PSYCOPG3 = True
    # Adapter for cursor factory name used later
    _DICT_ROW_FACTORY = _psycopg3_dict_row
except Exception:
    # fallback to psycopg2
    try:
        import psycopg2  # type: ignore
        from psycopg2 import connect  # type: ignore
        from psycopg2.extras import RealDictCursor as _psycopg2_dict_row  # type: ignore
        _DRIVER = "psycopg2"
        _USE_PSYCOPG3 = False
        _DICT_ROW_FACTORY = _psycopg2_dict_row
    except Exception:
        raise ImportError(
            "Neither 'psycopg' (psycopg3) nor 'psycopg2' is installed. "
            "Install one with `python -m pip install psycopg` or "
            "`python -m pip install psycopg2-binary`."
        )
from .config import settings

def _psycopg_conninfo() -> str:
    """Return a psycopg-friendly DSN string derived from DATABASE_URL."""

    raw_url = settings.database_url
    try:
        url_obj = make_url(raw_url)
    except Exception:
        return raw_url

    driver = url_obj.drivername
    if "+" in driver:
        driver = driver.split("+", 1)[0]
    if driver == "postgres":
        driver = "postgresql"

    if driver != url_obj.drivername:
        url_obj = url_obj.set(drivername=driver)

    return url_obj.render_as_string(hide_password=False)

@contextmanager
def get_connection() -> Iterator:
    """Yield a DB connection (psycopg or psycopg2)."""
    conn = connect(_psycopg_conninfo())
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
    """Yield a cursor; supports dict rows for both drivers."""
    with get_connection() as conn:
        cursor_factory = _DICT_ROW_FACTORY if dict_rows else None
        cur = None
        try:
            if _USE_PSYCOPG3:
                # psycopg v3 uses row_factory parameter
                cur = conn.cursor(row_factory=cursor_factory)  # type: ignore
            else:
                # psycopg2 uses cursor_factory parameter
                cur = conn.cursor(cursor_factory=cursor_factory)  # type: ignore
            yield cur
        except Exception:
            logger.exception("PostgreSQL cursor operation failed")
            raise
        finally:
            if cur is not None:
                cur.close()