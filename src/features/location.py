"""Location and availability feature scoring."""

from collections.abc import Mapping
from typing import Any

from src.features.behavioral import notice_score
from src.utils import config


PREFERRED_CITY_NAMES: frozenset[str] = frozenset(city.casefold() for city in config.PREFERRED_CITIES)
INDIA_COUNTRY_NAMES: frozenset[str] = frozenset({"india", "in", "bharat"})


def geo_score(profile: Mapping[str, Any] | None) -> float:
    """Score a candidate location based on preferred cities and relocation fit."""
    if not isinstance(profile, Mapping):
        return 0.50

    city = _string_value(profile, "city", "current_city", "location_city")
    country = _string_value(profile, "country", "current_country", "location_country")
    location = _string_value(profile, "location", "current_location")

    normalized_city = city.casefold()
    normalized_country = country.casefold()
    normalized_location = location.casefold()

    if normalized_city in PREFERRED_CITY_NAMES or normalized_location in PREFERRED_CITY_NAMES:
        return 1.0
    if _is_india(normalized_country, normalized_location):
        return 0.85
    if _willing_to_relocate(profile):
        return 0.70
    return 0.50


def location_score(
    profile: Mapping[str, Any] | None,
    redrob_signals: Mapping[str, Any] | None,
) -> float:
    """Combine geographic fit and notice-period availability into one score."""
    signals = redrob_signals if isinstance(redrob_signals, Mapping) else {}
    notice = notice_score(signals.get("notice_period_days"))
    return _clamp((0.60 * geo_score(profile)) + (0.40 * notice))


def _string_value(mapping: Mapping[str, Any], *keys: str) -> str:
    """Return the first non-empty string value from a mapping."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_india(country: str, location: str) -> bool:
    """Return whether normalized country or location text indicates India."""
    if country in INDIA_COUNTRY_NAMES:
        return True
    return any(token in location for token in INDIA_COUNTRY_NAMES)


def _willing_to_relocate(profile: Mapping[str, Any]) -> bool:
    """Return whether a profile contains an affirmative relocation signal."""
    value = profile.get("willing_to_relocate", profile.get("relocation_open"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "y", "1", "open", "willing"}
    if isinstance(value, int | float):
        return value > 0
    return False


def _clamp(value: float) -> float:
    """Clamp a score to the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, value))
