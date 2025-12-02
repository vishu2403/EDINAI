"""Dashboard domain services."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ..repository import dashboard_repository, lecture_credit_repository
from ..schemas import AdminResponse, MemberResponse, PackageResponse
from ..plan_limits import PLAN_CREDIT_LIMITS


def _serialize_admin(admin: Optional[dict]) -> Optional[dict]:
    return AdminResponse(**admin).model_dump() if admin else None


def _serialize_member(member: Optional[dict]) -> Optional[dict]:
    return MemberResponse(**member).model_dump() if member else None


def _serialize_package(package: Optional[dict]) -> Optional[dict]:
    return PackageResponse(**package).model_dump() if package else None


def _aware_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_plan_label(raw_label: Optional[str]) -> Optional[str]:
    if not raw_label:
        return None

    label = raw_label.strip().lower()
    if not label:
        return None

    label = re.sub(r"[^0-9a-z]", "", label)
    for token in ("plan", "package", "credits"):
        label = label.replace(token, "")

    if "k" not in label:
        numeric = re.search(r"(\d+)", raw_label.lower())
        if numeric:
            multiplier = "k" if "k" in raw_label.lower() else ""
            label = f"{numeric.group(1)}{multiplier}" or label

    return label or None


def _compute_expiry(admin: dict, package: Optional[dict]) -> Optional[datetime]:
    expiry = admin.get("expiry_date")
    if expiry:
        return expiry if expiry.tzinfo else expiry.replace(tzinfo=timezone.utc)

    start_date = admin.get("start_date")
    if not start_date or not package:
        return None

    duration = package.get("duration_days")
    if not duration:
        return None

    start_dt = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    return start_dt + timedelta(days=int(duration))


def get_admin_dashboard(admin_id: int) -> Dict[str, object]:
    admin = dashboard_repository.fetch_admin(admin_id)
    if not admin:
        raise ValueError("Admin not found")

    package = (
        dashboard_repository.fetch_package(admin.get("package"))
        if admin.get("package")
        else None
    )

    plan_label = _normalize_plan_label(admin.get("package"))
    plan_credit_total = PLAN_CREDIT_LIMITS.get(plan_label)
    credit_usage = lecture_credit_repository.get_usage(admin_id)
    credit_remaining = (
        max(plan_credit_total - credit_usage["credits_used"], 0)
        if plan_credit_total is not None
        else None
    )
    post_limit_generated = (
        credit_usage["credits_used"] - plan_credit_total
        if plan_credit_total is not None and credit_usage["credits_used"] > plan_credit_total
        else 0
    )

    total_members = dashboard_repository.count_members(admin_id, active_only=False)
    active_members = dashboard_repository.count_members(admin_id, active_only=True)
    member_counts = dashboard_repository.count_members_by_work_type(admin_id)

    computed_expiry = _compute_expiry(admin, package)
    if computed_expiry:
        delta = computed_expiry - _aware_now()
        days_until_expiry = max(delta.days, 0)
        expiry_iso = computed_expiry.isoformat()
    else:
        days_until_expiry = 0
        expiry_iso = None

    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_activity = dashboard_repository.recent_member_activity(admin_id, week_ago)

    package_usage: Dict[str, object] = {}
    if package:
        limit = package["video_limit"] if package["video_limit"] > 0 else "unlimited"
        usage_percentage = (
            active_members / package["video_limit"] * 100
            if isinstance(limit, (int, float)) and package["video_limit"] > 0
            else 0
        )
        package_usage = {
            "members_used": active_members,
            "members_limit": limit,
            "usage_percentage": usage_percentage,
        }

    return {
        "admin_info": _serialize_admin(admin),
        "package_info": _serialize_package(package),
        "member_statistics": {
            "total_members": total_members,
            "active_members": active_members,
            "inactive_members": total_members - active_members,
            "chapter_members": member_counts.get("chapter", 0),
            "student_members": member_counts.get("student", 0),
            "lecture_members": member_counts.get("lecture", 0),
        },
        "account_status": {
            "days_until_expiry": days_until_expiry,
            "is_expired": days_until_expiry < 0,
            "expiry_date": expiry_iso,
            "subscription_status": "active" if days_until_expiry > 0 else "expired",
        },
        "lecture_credits": {
            "plan_label": plan_label,
            "total": plan_credit_total,
            "used": credit_usage["credits_used"],
            "remaining": credit_remaining,
            "post_limit_generated": post_limit_generated,
            "overflow_attempts": credit_usage["overflow_attempts"],
        },
        "recent_activity": recent_activity,
        "package_usage": package_usage,
    }


def get_member_dashboard(*, member_id: int, admin_id: int, work_type: str) -> Dict[str, object]:
    member = dashboard_repository.fetch_member(member_id)
    if not member:
        raise ValueError("Member not found")

    admin = dashboard_repository.fetch_admin(admin_id)
    package = (
        dashboard_repository.fetch_package(admin.get("package"))
        if admin and admin.get("package")
        else None
    )

    if work_type == "chapter":
        total_students = dashboard_repository.count_members(admin_id, work_type="student", active_only=True)
        total_lectures = dashboard_repository.count_members(admin_id, work_type="lecture", active_only=True)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_student_activity = dashboard_repository.count_members_since(
            admin_id,
            since=week_ago,
            work_type="student",
            field="last_login",
        )
        payload = {
            "chapter_metrics": {
                "total_students": total_students,
                "total_lectures": total_lectures,
                "recent_student_activity": recent_student_activity,
            }
        }
    elif work_type == "student":
        total_chapters = dashboard_repository.count_members(admin_id, work_type="chapter", active_only=True)
        total_lectures = dashboard_repository.count_members(admin_id, work_type="lecture", active_only=True)
        payload = {
            "student_metrics": {
                "total_chapters": total_chapters,
                "total_lectures": total_lectures,
                "progress": {
                    "completed_lectures": 0,
                    "total_available_lectures": total_lectures,
                    "progress_percentage": 0,
                    "last_activity": member["last_login"].isoformat() if member.get("last_login") else None,
                },
            }
        }
    else:  # lecture
        total_students = dashboard_repository.count_members(admin_id, work_type="student", active_only=True)
        total_chapters = dashboard_repository.count_members(admin_id, work_type="chapter", active_only=True)
        payload = {
            "lecture_metrics": {
                "total_students": total_students,
                "total_chapters": total_chapters,
                "performance": {
                    "total_lectures_created": 0,
                    "total_students_enrolled": total_students,
                    "average_completion_rate": 0,
                    "total_watch_time": 0,
                    "last_lecture_created": None,
                },
                "package_limits": (
                    {
                        "max_minutes_per_lecture": package["max_minutes_per_lecture"],
                        "ai_videos_per_lecture": package["ai_videos_per_lecture"],
                        "topics_per_lecture": package["topics_per_lecture"],
                        "max_quality": package["max_quality"],
                    }
                    if package
                    else {}
                ),
            }
        }

    plan_label = _normalize_plan_label(admin.get("package") or admin.get("package_plan"))
    return {
        "member_info": _serialize_member(member),
        "admin_info": {
            "name": admin["name"],
            "email": admin["email"],
            "plan_label": plan_label or admin.get("package") or admin.get("package_plan"),
        }
        if admin
        else None,
        "package_info": _serialize_package(package),
        "work_type": work_type,
        **payload,
    }


def get_summary(admin_id: int) -> Dict[str, object]:
    summary = dashboard_repository.dashboard_summary(admin_id)
    summary["alerts"] = []

    if summary["days_until_expiry"] <= 30:
        summary["alerts"].append(
            {
                "type": "warning",
                "message": f"Subscription expires in {summary['days_until_expiry']} days",
            }
        )

    if summary["total_members"] == 0:
        summary["alerts"].append(
            {
                "type": "info",
                "message": "No members added yet. Add your first team member!",
            }
        )

    return summary