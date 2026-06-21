"""Trust-and-safety scoring for suspicious or inflated candidate profiles."""

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from typing import Any

from src.utils.config import HONEYPOT_PENALTY
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

CRITICAL_FIELDS: tuple[str, ...] = ("profile", "career_history", "skills")
DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m",
    "%Y/%m",
    "%b %Y",
    "%B %Y",
    "%Y",
)
EXPERT_NO_EXPERIENCE_THRESHOLD: int = 5
LOW_DURATION_MONTHS: float = 6.0


def honeypot_score(candidate: Mapping[str, Any] | None) -> float:
    """Return a bounded penalty for unrealistic, inconsistent, or malformed profiles."""
    if not isinstance(candidate, Mapping):
        LOGGER.debug("Honeypot check failed: candidate is not a mapping.")
        return float(HONEYPOT_PENALTY)

    penalty = 0.0
    triggered_checks: list[str] = []

    if _expert_without_experience_count(candidate) > EXPERT_NO_EXPERIENCE_THRESHOLD:
        penalty += HONEYPOT_PENALTY * 0.20
        triggered_checks.append("expert_with_no_experience")

    if _has_invalid_timeline(candidate):
        penalty += HONEYPOT_PENALTY * 0.25
        triggered_checks.append("invalid_career_timeline")

    if _has_experience_inconsistency(candidate):
        penalty += HONEYPOT_PENALTY * 0.20
        triggered_checks.append("experience_inconsistency")

    if _has_unrealistic_skill_profile(candidate):
        penalty += HONEYPOT_PENALTY * 0.20
        triggered_checks.append("unrealistic_skill_profile")

    if _has_data_integrity_issues(candidate):
        penalty += HONEYPOT_PENALTY * 0.15
        triggered_checks.append("data_integrity_issues")

    final_penalty = min(float(HONEYPOT_PENALTY), float(penalty))
    if triggered_checks:
        LOGGER.debug(
            "Honeypot checks triggered",
            extra={"checks": triggered_checks, "penalty": round(final_penalty, 4)},
        )
    return final_penalty


def _expert_without_experience_count(candidate: Mapping[str, Any]) -> int:
    """Count expert skills with explicitly zero duration."""
    count = 0
    for skill in _skill_records(candidate):
        if _proficiency(skill) == "expert" and _duration_months(skill.get("duration_months")) == 0:
            count += 1
    return count


def _has_invalid_timeline(candidate: Mapping[str, Any]) -> bool:
    """Return whether career dates contain future starts or impossible overlaps."""
    today = date.today()
    periods: list[tuple[date, date]] = []

    for job in _career_records(candidate):
        start = _job_start_date(job)
        end = _job_end_date(job)
        if start is not None and start > today:
            return True
        if start is not None and end is not None and start > end:
            return True
        if start is not None:
            periods.append((start, end or today))

    return _has_impossible_overlap(periods)


def _has_experience_inconsistency(candidate: Mapping[str, Any]) -> bool:
    """Return whether reported experience greatly exceeds derived career duration."""
    reported_years = _reported_years_of_experience(candidate)
    if reported_years is None or reported_years <= 0:
        return False

    total_months = _total_career_duration_months(candidate)
    reported_months = reported_years * 12.0
    if total_months <= 0:
        return reported_months >= 24.0

    exceeds_by_two_years = reported_months > total_months + 24.0
    exceeds_by_ratio = reported_months > total_months * 1.5
    return exceeds_by_two_years and exceeds_by_ratio


def _has_unrealistic_skill_profile(candidate: Mapping[str, Any]) -> bool:
    """Return whether expert skills are implausibly numerous with very low duration."""
    skills = _skill_records(candidate)
    if not skills:
        return False

    expert_skills = [skill for skill in skills if _proficiency(skill) == "expert"]
    low_duration_experts = [
        skill
        for skill in expert_skills
        if (_duration_months(skill.get("duration_months")) or 0.0) <= LOW_DURATION_MONTHS
    ]

    if len(low_duration_experts) > 10:
        return True
    if len(skills) >= 12 and len(low_duration_experts) / len(skills) > 0.60:
        return True
    return len(expert_skills) > 15 and len(low_duration_experts) / len(expert_skills) >= 0.50


def _has_data_integrity_issues(candidate: Mapping[str, Any]) -> bool:
    """Return whether critical fields are missing or record containers are malformed."""
    for field in CRITICAL_FIELDS:
        if _is_missing(candidate.get(field)):
            return True

    career_history = candidate.get("career_history")
    skills = candidate.get("skills")
    if not isinstance(career_history, list) or not isinstance(skills, list):
        return True
    if any(not isinstance(job, Mapping) for job in career_history):
        return True
    return any(not isinstance(skill, Mapping | str) for skill in skills)


def _skill_records(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return mapping-shaped skill records only."""
    skills = candidate.get("skills", [])
    if not isinstance(skills, Iterable) or isinstance(skills, str | bytes):
        return []
    return [skill for skill in skills if isinstance(skill, Mapping)]


def _career_records(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return mapping-shaped career-history records only."""
    career_history = candidate.get("career_history", [])
    if not isinstance(career_history, list):
        return []
    return [job for job in career_history if isinstance(job, Mapping)]


def _job_start_date(job: Mapping[str, Any]) -> date | None:
    """Parse a start date from common career-history fields."""
    return _first_date(job, ("start_date", "started_at", "start", "from"))


def _job_end_date(job: Mapping[str, Any]) -> date | None:
    """Parse an end date from common career-history fields."""
    return _first_date(job, ("end_date", "ended_at", "end", "to", "until"))


def _first_date(job: Mapping[str, Any], keys: tuple[str, ...]) -> date | None:
    """Return the first parseable date from a job record."""
    for key in keys:
        parsed = _parse_date(job.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_date(value: Any) -> date | None:
    """Parse common resume date values into dates."""
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    if raw_value.casefold() in {"present", "current", "now"}:
        return date.today()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(raw_value, date_format).date()
        except ValueError:
            continue
    return None


def _has_impossible_overlap(periods: list[tuple[date, date]]) -> bool:
    """Detect implausible overlaps across employment periods."""
    if len(periods) < 2:
        return False

    sorted_periods = sorted(periods)
    for index, (start, end) in enumerate(sorted_periods):
        for other_start, other_end in sorted_periods[index + 1 :]:
            if other_start > end:
                break
            overlap_days = (min(end, other_end) - other_start).days
            if overlap_days > 365:
                return True

    events: list[tuple[date, int]] = []
    for start, end in sorted_periods:
        events.append((start, 1))
        events.append((end + timedelta(days=1), -1))

    active_periods = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active_periods += delta
        if active_periods > 2:
            return True
    return False


def _reported_years_of_experience(candidate: Mapping[str, Any]) -> float | None:
    """Return reported years of experience from candidate or profile fields."""
    profile = candidate.get("profile", {})
    sources = [candidate]
    if isinstance(profile, Mapping):
        sources.append(profile)

    for source in sources:
        for key in ("years_of_experience", "total_experience_years", "experience_years", "yoe"):
            years = _to_float(source.get(key))
            if years is not None:
                return max(0.0, years)
    return None


def _total_career_duration_months(candidate: Mapping[str, Any]) -> float:
    """Return total career duration in months, using merged dates before raw durations."""
    periods: list[tuple[date, date]] = []
    duration_months = 0.0
    today = date.today()

    for job in _career_records(candidate):
        start = _job_start_date(job)
        end = _job_end_date(job) or today
        if start is not None and start <= end:
            periods.append((start, end))
        else:
            duration_months += _duration_months(job.get("duration_months")) or 0.0

    if periods:
        return _merged_duration_months(periods) + duration_months
    return duration_months


def _merged_duration_months(periods: list[tuple[date, date]]) -> float:
    """Return total covered months after merging overlapping periods."""
    merged: list[tuple[date, date]] = []
    for start, end in sorted(periods):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    total_days = sum((end - start).days for start, end in merged)
    return total_days / 30.4375


def _proficiency(skill: Mapping[str, Any]) -> str:
    """Return normalized proficiency text for a skill record."""
    value = skill.get("proficiency")
    return value.strip().casefold() if isinstance(value, str) else ""


def _duration_months(value: Any) -> float | None:
    """Return non-negative duration months while preserving missing as None."""
    duration = _to_float(value)
    return max(0.0, duration) if duration is not None else None


def _to_float(value: Any) -> float | None:
    """Convert numeric-like values to float while treating null sentinels as missing."""
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
    """Return whether a value is null, blank, empty, or a known sentinel."""
    if value is None or value == -1:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "-1"}
    if isinstance(value, list | tuple | dict | set):
        return len(value) == 0
    return False
