"""Skill-strength feature scoring for candidate ranking."""

from collections.abc import Iterable, Mapping
from typing import Any

from src.jd.constants import PENALIZED_PRIMARY_SKILLS
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
PROFICIENCY_SCORES: dict[str, float] = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.9,
    "expert": 1.0,
}
PENALIZED_SKILL_NAMES: frozenset[str] = frozenset(
    skill.casefold() for skill in PENALIZED_PRIMARY_SKILLS
)


def skill_score(skills: Iterable[Any] | None) -> float:
    """Return a normalized 0-1 score from proficiency, duration, and endorsements."""
    valid_scores: list[float] = []
    malformed_count = 0

    for skill in _iter_skills(skills):
        if not isinstance(skill, Mapping):
            malformed_count += 1
            continue

        valid_scores.append(
            _clamp(
                0.50 * _proficiency_score(skill.get("proficiency"))
                + 0.30 * _duration_score(skill.get("duration_months"))
                + 0.20 * _endorsement_score(skill.get("endorsements"))
            )
        )

    if malformed_count:
        LOGGER.debug("Ignored %d malformed skill records.", malformed_count)
    if not valid_scores:
        return 0.0

    return _clamp(sum(valid_scores) / len(valid_scores))


def has_penalized_primary_skill(skills: Iterable[Any] | None) -> bool:
    """Return True when the top three duration-ranked skills are penalized skills."""
    skill_records = [skill for skill in _iter_skills(skills) if isinstance(skill, Mapping)]
    if len(skill_records) < 3:
        return False

    top_skills = sorted(skill_records, key=_safe_duration_months, reverse=True)[:3]
    penalized_count = sum(_skill_name(skill) in PENALIZED_SKILL_NAMES for skill in top_skills)
    return penalized_count >= 2


def _iter_skills(skills: Iterable[Any] | None) -> list[Any]:
    """Return skill items from iterable inputs and reject malformed containers safely."""
    if skills is None:
        return []
    if isinstance(skills, str | bytes):
        LOGGER.debug("Skill list was provided as a scalar string.")
        return []
    try:
        return list(skills)
    except TypeError:
        LOGGER.debug("Skill input is not iterable: %r", type(skills).__name__)
        return []


def _proficiency_score(value: Any) -> float:
    """Map proficiency labels or numeric values into a 0-1 score."""
    if _is_missing(value):
        return 0.0
    if isinstance(value, str):
        return PROFICIENCY_SCORES.get(value.strip().casefold(), 0.0)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return _clamp(float(value))
    return 0.0


def _duration_score(value: Any) -> float:
    """Return duration score capped at 36 months."""
    return _clamp(_safe_duration_months_value(value) / 36.0)


def _endorsement_score(value: Any) -> float:
    """Return endorsement score capped at 50 endorsements."""
    count = _to_float(value)
    return _clamp((count or 0.0) / 50.0)


def _safe_duration_months(skill: Mapping[str, Any]) -> float:
    """Return a safe duration value for sorting skill records."""
    return _safe_duration_months_value(skill.get("duration_months"))


def _safe_duration_months_value(value: Any) -> float:
    """Return non-negative duration months from a malformed-safe value."""
    months = _to_float(value)
    return max(0.0, months or 0.0)


def _skill_name(skill: Mapping[str, Any]) -> str:
    """Return a normalized skill name from common record fields."""
    for key in ("name", "skill", "skill_name", "title"):
        value = skill.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return ""


def _to_float(value: Any) -> float | None:
    """Convert numeric-like values to float while treating sentinels as missing."""
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _is_missing(value: Any) -> bool:
    """Return whether a value is null, blank, or a known missing sentinel."""
    return value is None or value == -1 or (isinstance(value, str) and value.strip() in {"", "-1"})


def _clamp(value: float) -> float:
    """Clamp a score to the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, value))
