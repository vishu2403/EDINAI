"""Database helpers for contact onboarding using raw PostgreSQL queries."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable, List, Optional

from ..postgres import get_pg_cursor
from ..schemas.contact_schema import ContactCreate

TABLE_NAME = "admin_portal_contacts"

_UPDATABLE_COLUMNS: set[str] = {
    "first_name",
    "last_name",
    "education_center_name",
    "address",
    "designation",
    "phone_number",
    "dob",
    "inai_email",
    "image_path",
    "center_photos",
    "logo_path",
    "other_activities_path",
    "inai_password_encrypted",
    "created_by",
}


def _row_to_contact(row: Optional[dict]) -> Optional[SimpleNamespace]:
    if row is None:
        return None
    return SimpleNamespace(**row)


def _fetchone(query: str, params: Iterable | dict) -> Optional[SimpleNamespace]:
    with get_pg_cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    return _row_to_contact(row)


def create_contact(
    contact_in: ContactCreate,
    *,
    admin_id: int,
    image_path: Optional[str] = None,
    center_photos: Optional[str] = None,
    logo_path: Optional[str] = None,
    other_activities_path: Optional[str] = None,
    inai_password_encrypted: Optional[str] = None,
    created_by: Optional[int] = None,
) -> Optional[SimpleNamespace]:
    params = {
        "admin_id": admin_id,
        "created_by": created_by,
        "first_name": contact_in.first_name,
        "last_name": contact_in.last_name,
        "education_center_name": contact_in.education_center_name,
        "address": contact_in.address,
        "designation": contact_in.designation,
        "phone_number": contact_in.phone_number,
        "dob": str(contact_in.dob) if getattr(contact_in, "dob", None) else None,
        "image_path": image_path,
        "center_photos": center_photos,
        "logo_path": logo_path,
        "other_activities_path": other_activities_path,
        "inai_email": contact_in.inai_email,
        "inai_password_encrypted": inai_password_encrypted,
    }

    insert_sql = f"""
        INSERT INTO {TABLE_NAME} (
            admin_id,
            created_by,
            first_name,
            last_name,
            education_center_name,
            address,
            designation,
            phone_number,
            dob,
            inai_email,
            image_path,
            center_photos,
            logo_path,
            other_activities_path,
            inai_password_encrypted
        )
        VALUES (
            %(admin_id)s,
            %(created_by)s,
            %(first_name)s,
            %(last_name)s,
            %(education_center_name)s,
            %(address)s,
            %(designation)s,
            %(phone_number)s,
            %(dob)s,
            %(inai_email)s,
            %(image_path)s,
            %(center_photos)s,
            %(logo_path)s,
            %(other_activities_path)s,
            %(inai_password_encrypted)s
        )
        RETURNING *
    """

    with get_pg_cursor() as cur:
        cur.execute(insert_sql, params)
        row = cur.fetchone()

    return _row_to_contact(row)


def get_contact_by_id(contact_id: int) -> Optional[SimpleNamespace]:
    select_sql = f"SELECT * FROM {TABLE_NAME} WHERE id = %s"
    return _fetchone(select_sql, (contact_id,))


def get_contact_by_admin(admin_id: int) -> Optional[SimpleNamespace]:
    select_sql = f"SELECT * FROM {TABLE_NAME} WHERE admin_id = %s ORDER BY id DESC LIMIT 1"
    return _fetchone(select_sql, (admin_id,))


def list_contacts() -> List[SimpleNamespace]:
    with get_pg_cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME} ORDER BY id")
        rows = cur.fetchall()
    return [_row_to_contact(row) for row in rows]


def update_contact_inai(
    contact_id: int,
    *,
    inai_email: Optional[str] = None,
    inai_password_encrypted: Optional[str] = None,
) -> Optional[SimpleNamespace]:
    fields: dict[str, Optional[str]] = {}
    if inai_email is not None:
        fields["inai_email"] = inai_email
    if inai_password_encrypted is not None:
        fields["inai_password_encrypted"] = inai_password_encrypted

    return update_contact(contact_id, **fields)


def update_contact(contact_id: int, **fields) -> Optional[SimpleNamespace]:
    if not fields:
        return get_contact_by_id(contact_id)

    invalid_columns = set(fields.keys()) - _UPDATABLE_COLUMNS
    if invalid_columns:
        raise ValueError(f"Invalid contact columns: {', '.join(invalid_columns)}")

    assignments = []
    params: dict[str, Optional[str]] = {"contact_id": contact_id}

    for index, (column, value) in enumerate(fields.items()):
        param_key = f"value_{index}"
        assignments.append(f"{column} = %({param_key})s")
        params[param_key] = value

    update_sql = (
        f"UPDATE {TABLE_NAME} SET " + ", ".join(assignments) + " WHERE id = %(contact_id)s RETURNING *"
    )

    with get_pg_cursor() as cur:
        cur.execute(update_sql, params)
        row = cur.fetchone()

    return _row_to_contact(row)
