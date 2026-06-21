"""Career-history feature extraction for candidate scoring."""

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import Any

from src.utils.config import BAD_INDUSTRIES


BAD_INDUSTRY_NAMES: frozenset[str] = frozenset(industry.casefold() for industry in BAD_INDUSTRIES)
ML_SYSTEM_KEYWORDS: tuple[str, ...] = (
    "embedding",
    "retrieval",
    "ranking",
    "vector",
    "recommendation",
)
DESCRIPTION_FIELDS: tuple[str, ...] = (
    "description",
    "responsibilities",
    "achievements",
    "summary",
    "title",
)


def career_quality(candidate: Mapping[str, Any]) -> float:
    """Score non-penalized industry experience, normalized to 60 months."""
    try:
        months = sum(
            _duration_months(job)
            for job in _career_history(candidate)
            if _industry_name(job) not in BAD_INDUSTRY_NAMES
        )
        return _clamp(months / 60.0)
    except Exception:
        return 0.0


def continuity_score(candidate: Mapping[str, Any]) -> float:
    """Score career continuity by penalizing more than three hops in five years."""
    try:
        history = _career_history(candidate)
        if not history:
            return 0.0

        cutoff = date.today() - timedelta(days=365 * 5)
        recent_starts = [_start_date(job) for job in history]
        recent_hops = sum(start is not None and start >= cutoff for start in recent_starts)
        if recent_hops == 0 and all(start is None for start in recent_starts):
            recent_hops = max(0, len(history) - 1)

        excess_hops = max(0, recent_hops - 3)
        return _clamp(1.0 - (excess_hops * 0.25))
    except Exception:
        return 0.0


def has_shipped_ml_system(candidate: Mapping[str, Any]) -> bool:
    """Return whether career descriptions indicate shipped ML retrieval systems."""
    try:
        for job in _career_history(candidate):
            text = " ".join(_text_values(job, DESCRIPTION_FIELDS)).casefold()
            if any(keyword in text for keyword in ML_SYSTEM_KEYWORDS):
                return True
    except Exception:
        return False
    return False


def combined_career_score(candidate: Mapping[str, Any]) -> float:
    """Combine quality, continuity, and shipped-system signals into one score."""
    try:
        shipped_score = 1.0 if has_shipped_ml_system(candidate) else 0.0
        score = (
            0.50 * career_quality(candidate)
            + 0.30 * continuity_score(candidate)
            + 0.20 * shipped_score
        )
        return _clamp(score)
    except Exception:
        return 0.0


def _career_history(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return career history entries that are mapping-like records."""
    history = candidate.get("career_history", [])
    if not isinstance(history, list):
        return []
    return [job for job in history if isinstance(job, Mapping)]


def _duration_months(job: Mapping[str, Any]) -> float:
    """Return a non-negative duration in months from a job record."""
    value = job.get("duration_months", 0)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _industry_name(job: Mapping[str, Any]) -> str:
    """Return a normalized industry name from a job record."""
    value = job.get("industry", "")
    return value.casefold() if isinstance(value, str) else ""


def _start_date(job: Mapping[str, Any]) -> date | None:
    """Parse a start date from common career-history fields."""
    for key in ("start_date", "started_at", "start", "from"):
        parsed = _parse_date(job.get(key))
        if parsed is not None:
            return parsed

    year = job.get("start_year")
    if isinstance(year, int):
        month = job.get("start_month", 1)
        return date(year, month if isinstance(month, int) and 1 <= month <= 12 else 1, 1)
    return None


def _parse_date(value: Any) -> date | None:
    """Parse a date value using common resume date formats."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    raw_value = value.strip()
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%b %Y", "%B %Y", "%Y"):
        try:
            return datetime.strptime(raw_value, date_format).date()
        except ValueError:
            continue
    return None


def _text_values(job: Mapping[str, Any], fields: Iterable[str]) -> list[str]:
    """Collect string values from selected fields, including simple string lists."""
    values: list[str] = []
    for field in fields:
        value = job.get(field)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return values


def _clamp(value: float) -> float:
    """Clamp a score to the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, value))
