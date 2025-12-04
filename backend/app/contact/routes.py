"""Routes for admin contact onboarding."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cryptography.fernet import Fernet
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from jose import JWTError, jwt

from ..config import settings
from ..contact import crud as contact_crud
from ..schemas import ContactCreate, ContactRead
from ..repository import registration_repository
from ..utils.session_store import valid_tokens

router = APIRouter(prefix="/contact", tags=["Contact Management"])

UPLOAD_BASE = Path("uploads/contact")
UPLOAD_BASE.mkdir(parents=True, exist_ok=True)


def _get_fernet() -> Optional[Fernet]:
    key = settings.fernet_key if hasattr(settings, "fernet_key") else None
    if not key:
        return None

    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as exc:  # pragma: no cover - invalid key
        print(f"⚠️ Invalid Fernet key: {exc}")
        return None


def _encrypt_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    fernet = _get_fernet()
    if not fernet:
        return value

    return fernet.encrypt(value.encode()).decode()


def _validate_phone_number(phone: Optional[str]) -> None:
    if phone and (not phone.isdigit() or len(phone) != 10):
        raise HTTPException(status_code=400, detail="Phone number must be exactly 10 digits")


async def _save_upload(file: UploadFile, subfolder: str) -> Optional[str]:
    if not file:
        return None

    folder = UPLOAD_BASE / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file.filename

    content = await file.read()
    with open(path, "wb") as buffer:
        buffer.write(content)

    return str(path).replace("\\", "/")


async def _save_multiple(files: Optional[List[UploadFile]], subfolder: str) -> Optional[str]:
    if not files:
        return None

    paths = []
    for file in files:
        saved = await _save_upload(file, subfolder)
        if saved:
            paths.append(saved)

    return ";".join(paths) if paths else None


def _validate_session(request: Request) -> int:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:  # pragma: no cover - invalid token
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    sub = payload.get("sub")
    user_type = payload.get("user_type")
    session_id = payload.get("session_id")

    if not sub or not user_type or not session_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    if user_type == "admin":
        admin_id = int(sub)
    else:
        admin_id = int(payload.get("admin_id", sub))

    if valid_tokens.get(f"admin_{admin_id}") != session_id:
        raise HTTPException(status_code=401, detail="Token expired or invalid. Please login again.")

    return admin_id


def _serialize_paths(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part for part in value.split(";") if part]


def _merge_paths(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    existing_paths = _serialize_paths(existing)
    new_paths = _serialize_paths(new)

    if not new_paths:
        return existing if existing_paths else None

    seen = set(existing_paths)
    merged: List[str] = existing_paths.copy()

    for path in new_paths:
        if path not in seen:
            merged.append(path)
            seen.add(path)

    if not merged:
        return None

    return ";".join(merged)


@router.post("/", response_model=ContactRead)
async def upsert_contact(
    request: Request,
    first_name: str = Form(...),
    last_name: Optional[str] = Form(None),
    education_center_name: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    designation: Optional[str] = Form(None),
    phone_number: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    inai_email: Optional[str] = Form(None),
    inai_password: Optional[str] = Form(None),
    image: Optional[List[UploadFile]] = File(None),
    center_photos: Optional[List[UploadFile]] = File(None),
    logo: Optional[List[UploadFile]] = File(None),
    other_activity: Optional[List[UploadFile]] = File(None),
):
    admin_id = _validate_session(request)

    admin = registration_repository.get_admin_by_id(admin_id)
    if admin is None:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    _validate_phone_number(phone_number)

    image_paths = await _save_multiple(image, "profile")
    center_photo_paths = await _save_multiple(center_photos, "center")
    logo_paths = await _save_multiple(logo, "logo")
    activity_paths = await _save_multiple(other_activity, "activity")

    encrypted_password = _encrypt_value(inai_password)

    contact = contact_crud.get_contact_by_admin(admin_id)

    if contact:
        updates = {
            "first_name": first_name,
            "last_name": last_name,
            "education_center_name": education_center_name,
            "address": address,
            "designation": designation,
            "phone_number": phone_number,
            "dob": dob,
            "inai_email": inai_email or contact.inai_email,
        }

        merged_image_paths = _merge_paths(contact.image_path, image_paths)
        if merged_image_paths is not None:
            updates["image_path"] = merged_image_paths

        merged_center_photos = _merge_paths(contact.center_photos, center_photo_paths)
        if merged_center_photos is not None:
            updates["center_photos"] = merged_center_photos

        merged_logo_paths = _merge_paths(contact.logo_path, logo_paths)
        if merged_logo_paths is not None:
            updates["logo_path"] = merged_logo_paths

        merged_activity_paths = _merge_paths(contact.other_activities_path, activity_paths)
        if merged_activity_paths is not None:
            updates["other_activities_path"] = merged_activity_paths

        if encrypted_password:
            updates["inai_password_encrypted"] = encrypted_password

        contact = contact_crud.update_contact(contact.id, **updates)
    else:
        contact = contact_crud.create_contact(
            ContactCreate(
                first_name=first_name,
                last_name=last_name,
                education_center_name=education_center_name,
                address=address,
                designation=designation,
                phone_number=phone_number,
                dob=None,
                inai_email=inai_email,
            ),
            admin_id=admin_id,
            created_by=admin_id,
            image_path=image_paths,
            center_photos=center_photo_paths,
            logo_path=logo_paths,
            other_activities_path=activity_paths,
            inai_password_encrypted=encrypted_password,
        )

    # CRITICAL: Update administrators table for login authentication!
    # The 3-step onboarding calls /contact/ NOT /complete-onboarding/
    if inai_password and inai_email:
        from ..services.registration_service import hash_password
        
        # Hash the password properly with bcrypt
        hashed_password = hash_password(inai_password)
        
        # Get admin's original registration email
        original_email = admin.get("email")
        
        if original_email:
            # Update administrators table with new password
            admin_in_auth = registration_repository.get_admin_by_email(original_email)
            if admin_in_auth:
                registration_repository.update_admin(
                    admin_in_auth["admin_aid"],
                    password=hashed_password,
                    inai_email=inai_email,
                    inai_password_encrypted=hashed_password,
                    updated_at=datetime.utcnow(),
                )
                print(f"✅ Updated password in administrators table for {original_email}")

    return ContactRead(
        id=contact.id,
        admin_id=contact.admin_id,
        first_name=contact.first_name,
        last_name=contact.last_name,
        education_center_name=contact.education_center_name,
        address=contact.address,
        designation=contact.designation,
        phone_number=contact.phone_number,
        dob=contact.dob,
        inai_email=contact.inai_email,
        image_path=_serialize_paths(contact.image_path),
        center_photos=_serialize_paths(contact.center_photos),
        logo_path=_serialize_paths(contact.logo_path),
        other_activities_path=_serialize_paths(contact.other_activities_path),
    )


@router.get("/{admin_id}")
def get_contact(admin_id: int):
    contact = contact_crud.get_contact_by_admin(admin_id)
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    return {
        "status": True,
        "message": "Contact fetched successfully",
        "admin_id": contact.admin_id,
        "contact": ContactRead(
            id=contact.id,
            admin_id=contact.admin_id,
            first_name=contact.first_name,
            last_name=contact.last_name,
            education_center_name=contact.education_center_name,
            address=contact.address,
            designation=contact.designation,
            phone_number=contact.phone_number,
            dob=contact.dob,
            inai_email=contact.inai_email,
            image_path=_serialize_paths(contact.image_path),
            center_photos=_serialize_paths(contact.center_photos),
            logo_path=_serialize_paths(contact.logo_path),
            other_activities_path=_serialize_paths(contact.other_activities_path),
        ).model_dump(),
    }