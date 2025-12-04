"""Routes to complete admin onboarding (contact + education center + files)."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import EmailStr
logger = logging.getLogger(__name__)

from ..contact import crud as contact_crud
# from ..repository import admin_management_repository as admin_repo
# from ..repository import education_center_repository
from ..repository import(
    admin_management_repository as admin_repo,
    education_center_repository,
    registration_repository
)
from ..schemas import ResponseBase
from ..schemas.contact_schema import ContactCreate, ContactResponse
from ..schemas.education_center_schema import CompleteOnboardingResponse, EducationCenterResponse
from ..services.registration_service import hash_password
from ..utils.dependencies import onboarding_completed_required, onboarding_required, create_access_token
from ..utils.file_handler import (
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    DEFAULT_ALLOWED_EXTENSIONS,
    DEFAULT_ALLOWED_TYPES,
    save_multiple_files,
    save_uploaded_file,
)

router = APIRouter(prefix="/complete-onboarding", tags=["Complete Onboarding"])


def _merge_and_dump(existing: Optional[str], new_items: List[str]) -> Optional[str]:
    if not new_items:
        return None
    try:
        merged = json.loads(existing) if existing else []
    except json.JSONDecodeError:
        merged = []
    merged.extend(new_items)
    return json.dumps(merged)


def _ensure_list(value: Optional[Any]) -> Optional[List[str]]:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [value]
    return [value]


@router.post("/", response_model=ResponseBase)
async def complete_onboarding(
    current_user: dict = Depends(onboarding_required),
    first_name: str = Form(...),
    last_name: str = Form(...),
    address: str = Form(...),
    designation: str = Form(...),
    phone_number: str = Form(...),
    date_of_birth: str = Form(...),
    education_center_name: str = Form(...),
    inai_email: EmailStr = Form(...),
    inai_password: str = Form(...),
    upload_image: Optional[UploadFile] = File(None),
    logo: Optional[UploadFile] = File(None),
    center_photos: Optional[List[UploadFile]] = File(None),
    other_activities: Optional[List[UploadFile]] = File(None),
) -> ResponseBase:
    admin_id = current_user["id"]
    admin = admin_repo.get_admin_by_id(admin_id)
    if admin is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")


    logger.info(f"========== ONBOARDING STARTED ==========")
    logger.info(f"Admin ID: {admin_id}")
    logger.info(f"Admin original email: {admin.get('email')}")
    logger.info(f"INAI email from form: {inai_email}")
    logger.info(f"INAI password (first 5 chars): {inai_password[:5]}...")

    dob_value = date_of_birth.strip()
    try:
        dob_date = datetime.strptime(dob_value, "%d-%m-%Y").date()
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Date of birth must be DD-MM-YYYY") from None
    dob_iso = dob_date.isoformat()

    new_email = inai_email.lower().strip()
    if admin_repo.admin_exists_by_email(new_email, exclude_admin_id=admin_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An account with this INAI email already exists.")

    hashed_password = hash_password(inai_password)
    logger.info(f"Password hashed: {hashed_password[:30]}...")

    existing_contact = contact_crud.get_contact_by_admin(admin_id)
    contact_payload = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "education_center_name": education_center_name.strip(),
        "address": address.strip(),
        "designation": designation.strip(),
        "phone_number": phone_number.strip(),
        "dob": dob_iso,
        "inai_email": new_email,
    }

    contact_file_values: Dict[str, Optional[str]] = {}

    upload_image_path: Optional[str] = None
    if upload_image is not None:
        upload_info = await save_uploaded_file(upload_image, subfolder=f"education_center/admin_{admin_id}/profile")
        upload_image_path = upload_info["file_path"]
        contact_file_values["image_path"] = upload_image_path

    logo_path: Optional[str] = None
    if logo is not None:
        logo_info = await save_uploaded_file(logo, subfolder=f"education_center/admin_{admin_id}/logo")
        logo_path = logo_info["file_path"]
        contact_file_values["logo_path"] = logo_path

    center_photos_value: Optional[str] = None
    if center_photos:
        saved_photos = await save_multiple_files(center_photos, subfolder=f"education_center/admin_{admin_id}/center_photos")
        center_photos_value = json.dumps([info["file_path"] for info in saved_photos])
        contact_file_values["center_photos"] = center_photos_value

    other_activities_value: Optional[str] = None
    if other_activities:
        doc_extensions = DEFAULT_ALLOWED_EXTENSIONS | ALLOWED_PDF_EXTENSIONS
        doc_types = DEFAULT_ALLOWED_TYPES | ALLOWED_PDF_TYPES
        saved_activities = await save_multiple_files(
            other_activities,
            subfolder=f"education_center/admin_{admin_id}/other_activities",
            allowed_extensions=doc_extensions,
            allowed_types=doc_types,
        )
        other_activities_value = json.dumps([info["file_path"] for info in saved_activities])
    if other_activities_value is not None:
        contact_file_values["other_activities_path"] = other_activities_value

    if existing_contact:
        contact_update_payload = {**contact_payload}
        # Always update the encrypted password when completing onboarding
        contact_update_payload["inai_password_encrypted"] = hashed_password
        for key, value in contact_file_values.items():
            if value is not None:
                contact_update_payload[key] = value
        contact = contact_crud.update_contact(existing_contact.id, **contact_update_payload)
    else:
        contact = contact_crud.create_contact(
            contact_in=ContactCreate(
                first_name=first_name.strip(),
                last_name=last_name.strip(),
                education_center_name=education_center_name.strip(),
                address=address.strip(),
                designation=designation.strip(),
                phone_number=phone_number.strip(),
                dob=dob_date,
                inai_email=new_email,
            ),
            admin_id=admin_id,
            created_by=admin_id,
            inai_password_encrypted=hashed_password,
            image_path=contact_file_values.get("image_path"),
            center_photos=contact_file_values.get("center_photos"),
            logo_path=contact_file_values.get("logo_path"),
            other_activities_path=contact_file_values.get("other_activities_path"),
        )

    existing_center = education_center_repository.get_center_by_admin(admin_id)

    if existing_center:
        center_update_payload: Dict[str, Optional[str]] = {"name": education_center_name.strip()}
        if upload_image_path is not None:
            center_update_payload["upload_image"] = upload_image_path
        if center_photos_value is not None:
            center_update_payload["center_photos"] = center_photos_value
        if logo_path is not None:
            center_update_payload["logo"] = logo_path
        if other_activities_value is not None:
            center_update_payload["other_activities"] = other_activities_value

        education_center = education_center_repository.update_center(
            existing_center["id"],
            admin_id,
            **center_update_payload,
        )
    else:
        center_payload = {
            "name": education_center_name.strip(),
            "upload_image": upload_image_path,
            "center_photos": center_photos_value,
            "logo": logo_path,
            "other_activities": other_activities_value,
        }
        education_center = education_center_repository.create_center(admin_id=admin_id, **center_payload)

    # registration_repository.update_admin(
    #     admin_id,
    #     password=hashed_password,
    #     inai_email=new_email,
    #     inai_password_encrypted=hashed_password,
    #     updated_at=datetime.utcnow(),
    # )
    updated_admin = admin_repo.update_admin(
        admin_id,
        email=new_email,
        password=hashed_password,
        has_inai_credentials=True,
        updated_at=datetime.utcnow(),
    )
    if updated_admin is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update admin credentials")

     # CRITICAL: Also update the administrators table (used for login authentication)
    # The system has two admin tables - we must keep both in sync
    # IMPORTANT: We need to look up the admin in the administrators table using their
    # ORIGINAL registration email (admin["email"]), NOT the new INAI email,
    # because they might be different!
    original_admin_email = admin.get("email")
    
    logger.info(f"========== ADMINISTRATORS TABLE UPDATE ==========")
    logger.info(f"admin object keys: {list(admin.keys()) if isinstance(admin, dict) else 'not a dict'}")
    logger.info(f"admin.get('email'): {original_admin_email}")
    logger.info(f"new_email (INAI): {new_email}")
    
    if original_admin_email:
        logger.info(f"Looking for admin in administrators table with ORIGINAL email: {original_admin_email}")
        admin_in_auth_table = registration_repository.get_admin_by_email(original_admin_email)
        
        if admin_in_auth_table:
            logger.info(f"✅ Found admin in administrators table:")
            logger.info(f"   admin_aid: {admin_in_auth_table['admin_aid']}")
            logger.info(f"   email: {admin_in_auth_table['email']}")
            logger.info(f"   OLD password (first 30): {admin_in_auth_table.get('password', 'N/A')[:30]}...")
            logger.info(f"   NEW password (first 30): {hashed_password[:30]}...")
            
            logger.info(f"Updating administrators table...")
            updated_auth_admin = registration_repository.update_admin(
                admin_in_auth_table["admin_aid"],
                password=hashed_password,
                inai_email=new_email,
                inai_password_encrypted=hashed_password,
                updated_at=datetime.utcnow(),
            )
            
            if updated_auth_admin:
                # Verify the update by re-querying
                verified_admin = registration_repository.get_admin_by_email(original_admin_email)
                if verified_admin and verified_admin.get('password') == hashed_password:
                    logger.info(f"✅✅✅ VERIFIED: Password successfully updated in administrators table!")
                    logger.info(f"   New password hash (first 30): {verified_admin.get('password', '')[:30]}...")
                    logger.info(f"   Admin can now login with:")
                    logger.info(f"   📧 Email: {original_admin_email}")
                    logger.info(f"   🔑 Password: <the new password from onboarding form>")
                else:
                    logger.error(f"❌ VERIFICATION FAILED: Password in DB doesn't match what we set!")
                    logger.error(f"   Expected: {hashed_password[:30]}...")
                    logger.error(f"   Got: {verified_admin.get('password', 'N/A')[:30] if verified_admin else 'admin not found'}...")
            else:
                logger.error(f"❌ Failed to update administrators table for admin_aid={admin_in_auth_table['admin_aid']}")
        else:
            logger.warning(f"⚠️ Admin NOT FOUND in administrators table with email: {original_admin_email}")
            logger.warning(f"This admin will NOT be able to login! They need to be registered via /registration/register first.")
            logger.warning(f"Attempted lookup with: {original_admin_email}")
    else:
        logger.error(f"❌ admin.email is missing! Cannot sync with administrators table.")
        logger.error(f"admin object: {admin}")

    token_data = {
        "role": "admin",
        "id": updated_admin["admin_id"],
        "package": updated_admin.get("package"),
        "has_inai_credentials": True,
        "is_super_admin": updated_admin.get("is_super_admin"),
    }
    new_token = create_access_token(token_data)

    education_center_payload = {
        "id": education_center["id"],
        "admin_id": education_center["admin_id"],
        "name": education_center["name"],
        "upload_image": education_center.get("upload_image"),
        "center_photos": _ensure_list(education_center.get("center_photos")),
        "logo": education_center.get("logo"),
        "other_activities": _ensure_list(education_center.get("other_activities")),
        "created_at": education_center["created_at"],
    }

    response_payload = CompleteOnboardingResponse(
        contact=ContactResponse(
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
            created_at=datetime.utcnow(),
        ),
        education_center=EducationCenterResponse(**education_center_payload),
        onboarding_completed=True,
        new_token=new_token,
        admin_info={
            "admin_id": updated_admin["admin_id"],
            "name": updated_admin["name"],
            "email": updated_admin["email"],
            "package": updated_admin.get("package"),
            "has_inai_credentials": True,
        },
    )

    return ResponseBase(
        status=True,
        message="Onboarding completed successfully!",
        data=response_payload.model_dump(exclude_none=True),
    )
