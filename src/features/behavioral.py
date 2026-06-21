"""Behavioral signal scoring for Redrob candidate ranking."""

import datetime
import math

from src.utils import config


def recency_score(last_active_date: object) -> float:
    """Return an exponential activity-recency score using the configured half-life."""
    active_date = _parse_date(last_active_date)
    if active_date is None:
        return 0.0

    age_days = max(0, (datetime.date.today() - active_date).days)
    half_life = max(1, config.RECENCY_HALFLIFE_DAYS)
    return _clamp(math.exp(-math.log(2) * age_days / half_life))


def notice_score(notice_period_days: object) -> float:
    """Return an availability score based on configured notice-period thresholds."""
    days = _to_float(notice_period_days)
    if days is None:
        return 0.0

    thresholds = config.NOTICE_PERIOD_THRESHOLDS
    immediate = thresholds["immediate_days"]
    preferred = thresholds["preferred_max_days"]
    acceptable = thresholds["acceptable_max_days"]
    high_risk = thresholds["high_risk_min_days"]

    if days <= immediate:
        return 1.0
    if days <= preferred:
        return 0.85
    if days <= acceptable:
        return 0.60
    if days < high_risk:
        return 0.30
    return 0.10


def behavioral_score(redrob_signals: dict[str, object] | None) -> float:
    """Combine Redrob behavioral signals into a bounded score from 0 to 1."""
    if not isinstance(redrob_signals, dict):
        return 0.0

    open_to_work = _flag_score(redrob_signals.get("open_to_work_flag"))
    active_recency = recency_score(redrob_signals.get("last_active_date"))
    response_rate = _rate_score(redrob_signals.get("recruiter_response_rate"))
    notice = notice_score(redrob_signals.get("notice_period_days"))

    return _clamp(
        0.30 * open_to_work
        + 0.30 * active_recency
        + 0.25 * response_rate
        + 0.15 * notice
    )


def _parse_date(value: object) -> datetime.date | None:
    """Parse supported date-like values while treating null sentinels as missing."""
    if _is_missing(value):
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(raw_value, date_format).date()
        except ValueError:
            continue
    try:
        return datetime.datetime.fromisoformat(raw_value).date()
    except ValueError:
        return None


def _to_float(value: object) -> float | None:
    """Convert numeric values to float while treating null sentinels as missing."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"immediate", "now", "none"}:
            return 0.0
        digits = "".join(character for character in lowered if character.isdigit() or character == ".")
        if digits:
            try:
                return float(digits)
            except ValueError:
                return None
    return None


def _flag_score(value: object) -> float:
    """Return a conservative binary score for a flag-like signal."""
    if _is_missing(value):
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return 1.0 if value > 0 else 0.0
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "y", "1", "open", "available"}:
            return 1.0
        if lowered in {"false", "no", "n", "0", "closed", "unavailable"}:
            return 0.0
    return 0.0


def _rate_score(value: object) -> float:
    """Normalize a response-rate value expressed as either 0-1 or 0-100."""
    rate = _to_float(value)
    if rate is None:
        return 0.0
    if rate > 1:
        rate /= 100.0
    return _clamp(rate)


def _is_missing(value: object) -> bool:
    """Return whether a value is null, blank, or a known missing-value sentinel."""
    return value is None or value == -1 or (isinstance(value, str) and value.strip() in {"", "-1"})


def _clamp(value: float) -> float:
    """Clamp a score to the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, value))
