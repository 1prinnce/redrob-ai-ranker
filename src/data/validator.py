"""Validation helpers for candidate records."""

from collections.abc import Iterable
from typing import Any

from src.data.loader import validate_candidate_id
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)
REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "candidate_id",
        "profile",
        "career_history",
        "education",
        "skills",
        "certifications",
        "languages",
        "redrob_signals",
    }
)


def validate_schema(candidate_dict: dict[str, Any]) -> tuple[bool, str]:
    """Validate that a candidate record contains all required top-level keys."""
    missing_keys = sorted(REQUIRED_KEYS.difference(candidate_dict))
    if missing_keys:
        return False, f"Missing required keys: {', '.join(missing_keys)}"

    return True, ""


def validate_dataset(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates with valid IDs and required top-level schema keys."""
    valid_candidates: list[dict[str, Any]] = []
    invalid_count = 0

    for index, candidate in enumerate(candidates, start=1):
        is_valid_schema, error_message = validate_schema(candidate)
        if not is_valid_schema:
            invalid_count += 1
            LOGGER.error("Invalid candidate at row %d: %s", index, error_message)
            continue

        candidate_id = candidate["candidate_id"]
        if not validate_candidate_id(candidate_id):
            invalid_count += 1
            LOGGER.error("Invalid candidate_id at row %d: %r", index, candidate_id)
            continue

        valid_candidates.append(candidate)

    LOGGER.info(
        "Dataset validation complete: valid=%d invalid=%d",
        len(valid_candidates),
        invalid_count,
    )
    return valid_candidates
