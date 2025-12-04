"""Centralized plan-based limits for suggestions, lecture durations, and credits."""
from __future__ import annotations

from typing import Optional, Set

PLAN_SUGGESTION_LIMITS = {
    "20k": 2,
    "50k": 5,
    "100k": 8,
}

PLAN_DURATION_LIMITS = {
    "20k": 30,
    "50k": 45,
    "100k": 60,
}

PLAN_CREDIT_LIMITS = {
    "20k": 45,
    "50k": 120,
    "100k": 180,
}


def _known_plans() -> Set[str]:
    return set(PLAN_SUGGESTION_LIMITS) | set(PLAN_DURATION_LIMITS) | set(PLAN_CREDIT_LIMITS)


def normalize_plan_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    label = str(value).strip().lower()
    return label if label in _known_plans() else None
