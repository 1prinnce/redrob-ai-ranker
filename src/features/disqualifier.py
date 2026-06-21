"""Disqualifier and penalty rules for candidate career backgrounds."""

from collections.abc import Mapping
from typing import Any

from src.utils.config import CONSULTING_FIRMS, CONSULTING_PENALTY, RESEARCH_PENALTY


CONSULTING_FIRM_NAMES: frozenset[str] = frozenset(company.casefold() for company in CONSULTING_FIRMS)
RESEARCH_INDUSTRIES: frozenset[str] = frozenset({"research", "academia"})


def disqualifier_penalty(candidate: Mapping[str, Any]) -> float:
    """Return a configured penalty for consulting-only or research-only profiles."""
    history = _career_history(candidate)
    if not history:
        return 0.0

    companies = [_company_name(job) for job in history if _company_name(job)]
    has_consulting = any(company in CONSULTING_FIRM_NAMES for company in companies)
    has_product = any(company not in CONSULTING_FIRM_NAMES for company in companies)

    if companies and all(company in CONSULTING_FIRM_NAMES for company in companies):
        return float(CONSULTING_PENALTY)

    if _current_industry(history) in RESEARCH_INDUSTRIES:
        return float(RESEARCH_PENALTY)

    if has_consulting and has_product:
        return 0.0

    return 0.0


def _career_history(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return mapping-like career history entries."""
    history = candidate.get("career_history", [])
    if not isinstance(history, list):
        return []
    return [job for job in history if isinstance(job, Mapping)]


def _company_name(job: Mapping[str, Any]) -> str:
    """Return a normalized company name from a career-history entry."""
    for key in ("company", "company_name", "employer", "organization"):
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return ""


def _current_industry(history: list[Mapping[str, Any]]) -> str:
    """Return the normalized industry from the current or most recent role."""
    current_job = next((job for job in history if _is_current(job)), history[0])
    value = current_job.get("industry", "")
    return value.strip().casefold() if isinstance(value, str) else ""


def _is_current(job: Mapping[str, Any]) -> bool:
    """Return whether a career-history entry appears to be the current role."""
    value = job.get("is_current", job.get("current"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "y", "1", "current", "present"}
    return False
