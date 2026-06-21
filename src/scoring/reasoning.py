"""Human-readable candidate ranking explanations.

The functions in this module convert structured candidate signals into concise,
fact-grounded recruiter explanations. They deliberately avoid filling in missing
data: a field is mentioned only when it is present and safely parseable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import logging
from numbers import Integral, Real
import random
import re
from typing import Any, Final

import numpy as np


LOGGER = logging.getLogger(__name__)

HIGH_CONFIDENCE_RANK_MAX: Final[int] = 10
BALANCED_RANK_MAX: Final[int] = 50
EXPECTED_TOP_RANK_MAX: Final[int] = 100

HIGH_NOTICE_PERIOD_DAYS: Final[float] = 60.0
PREFERRED_NOTICE_PERIOD_DAYS: Final[float] = 30.0
LOW_RECRUITER_RESPONSE_RATE: Final[float] = 0.30
HIGH_RECRUITER_RESPONSE_RATE: Final[float] = 0.70
EXPERIENCE_STRENGTH_YEARS: Final[float] = 5.0

SCORE_DECIMAL_PLACES: Final[int] = 3
PERCENT_DECIMAL_PLACES: Final[int] = 0
YEARS_DECIMAL_PLACES: Final[int] = 1
MAX_TEXT_FIELD_CHARS: Final[int] = 90

DEFAULT_VARIANCE_SAMPLE_SIZE: Final[int] = 25
VARIANCE_SAMPLE_SEED: Final[int] = 7411
MIN_REASONING_SAMPLE_COUNT: Final[int] = 2
MIN_UNIQUE_REASONING_COUNT: Final[int] = 2
MIN_UNIQUE_REASONING_RATIO: Final[float] = 0.50

CURRENT_TITLE_KEYS: Final[tuple[str, ...]] = ("current_title",)
CURRENT_COMPANY_KEYS: Final[tuple[str, ...]] = ("current_company",)
YEARS_EXPERIENCE_KEYS: Final[tuple[str, ...]] = (
    "years_of_experience",
    "total_experience_years",
    "experience_years",
    "yoe",
)
RECRUITER_RESPONSE_RATE_KEYS: Final[tuple[str, ...]] = ("recruiter_response_rate",)
NOTICE_PERIOD_KEYS: Final[tuple[str, ...]] = ("notice_period_days",)

CURRENT_JOB_TITLE_KEYS: Final[tuple[str, ...]] = ("title", "job_title", "role", "position")
CURRENT_JOB_COMPANY_KEYS: Final[tuple[str, ...]] = (
    "company",
    "company_name",
    "employer",
    "organization",
)
CURRENT_JOB_FLAGS: Final[tuple[str, ...]] = ("is_current", "current")
CURRENT_JOB_END_KEYS: Final[tuple[str, ...]] = ("end_date", "ended_at", "to")
CURRENT_JOB_END_SENTINELS: Final[frozenset[str]] = frozenset({"present", "current", "now"})

WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"-?\d+(?:\.\d+)?")

STRONG_OPENERS: Final[tuple[str, ...]] = (
    "Strong high-confidence match",
    "Clear top-tier candidate",
    "Highly competitive profile",
)
BALANCED_OPENERS: Final[tuple[str, ...]] = (
    "Solid candidate",
    "Competitive profile",
    "Promising fit",
)
CAUTIOUS_OPENERS: Final[tuple[str, ...]] = (
    "Potential fit worth review",
    "Candidate merits a closer look",
    "Possible fit with selected strengths",
)


def generate_reasoning(candidate: Mapping[str, Any], rank: int, score: float) -> str:
    """Generate a concise recruiter-facing explanation for one ranked candidate.

    The explanation uses only structured facts present on the candidate record,
    primarily ``current_title``, ``current_company``, ``years_of_experience``,
    ``recruiter_response_rate``, and ``notice_period_days``. Availability and
    outreach concerns are included only when the corresponding values are
    present and cross configured risk thresholds.

    Args:
        candidate: Candidate record containing profile and Redrob signal data.
        rank: One-based rank position in the candidate list.
        score: Final normalized ranking score for the candidate.

    Returns:
        A short explanation suitable for recruiter review.

    Raises:
        TypeError: If ``candidate``, ``rank``, or ``score`` has an unsupported
            type.
        ValueError: If ``rank`` is not positive or ``score`` is non-finite.
    """
    if not isinstance(candidate, Mapping):
        raise TypeError(f"candidate must be a mapping, got {type(candidate).__name__}.")

    clean_rank = _validate_rank(rank)
    clean_score = _validate_score(score)
    profile = _as_mapping(candidate.get("profile"))
    signals = _as_mapping(candidate.get("redrob_signals"))

    current_title = _current_title(candidate, profile)
    current_company = _current_company(candidate, profile)
    years_of_experience = _first_number(candidate, profile, keys=YEARS_EXPERIENCE_KEYS)
    recruiter_response_rate = _response_rate(
        _first_value(signals, candidate, profile, keys=RECRUITER_RESPONSE_RATE_KEYS)
    )
    notice_period_days = _non_negative_number(
        _first_value(signals, candidate, profile, keys=NOTICE_PERIOD_KEYS)
    )

    opener = _rank_opener(clean_rank)
    facts = _fact_phrases(
        current_title=current_title,
        current_company=current_company,
        years_of_experience=years_of_experience,
    )
    signals_text = _signal_text(
        score=clean_score,
        recruiter_response_rate=recruiter_response_rate,
        notice_period_days=notice_period_days,
    )
    strengths = _strength_phrases(
        current_title=current_title,
        current_company=current_company,
        years_of_experience=years_of_experience,
        recruiter_response_rate=recruiter_response_rate,
        notice_period_days=notice_period_days,
        rank=clean_rank,
    )
    concerns = _concern_phrases(
        recruiter_response_rate=recruiter_response_rate,
        notice_period_days=notice_period_days,
    )

    sentences = [
        f"Rank {clean_rank}: {opener} with score {_format_score(clean_score)}.",
    ]
    if facts:
        sentences.append("Profile signals show " + ", ".join(facts) + ".")
    if signals_text:
        sentences.append(signals_text)
    if strengths:
        sentences.append("Strengths: " + ", ".join(strengths) + ".")
    if concerns:
        sentences.append("Concerns: " + "; ".join(concerns) + ".")

    reasoning = " ".join(sentences)
    LOGGER.debug(
        "Generated candidate reasoning",
        extra={
            "rank": clean_rank,
            "score": clean_score,
            "has_concerns": bool(concerns),
            "has_profile_facts": bool(facts),
        },
    )
    return reasoning


def validate_reasoning_variance(reasonings: Sequence[str]) -> None:
    """Validate that generated explanations have enough textual variance.

    A deterministic random sample is used so validation is reproducible in CI
    and submission pipelines. The sampled explanations are normalized for
    whitespace before comparison, then checked for a minimum number and ratio of
    unique strings.

    Args:
        reasonings: Sequence of generated explanation strings.

    Raises:
        TypeError: If ``reasonings`` is not a sequence of strings.
        ValueError: If there are too few explanations, blank explanations, or
            sampled variance falls below configured thresholds.
    """
    clean_reasonings = _validate_reasonings(reasonings)
    sample_count = min(DEFAULT_VARIANCE_SAMPLE_SIZE, len(clean_reasonings))
    sampler = random.Random(VARIANCE_SAMPLE_SEED)
    sampled_reasonings = sampler.sample(clean_reasonings, sample_count)

    unique_reasonings = set(sampled_reasonings)
    unique_count = len(unique_reasonings)
    unique_ratio = unique_count / sample_count

    if unique_count < MIN_UNIQUE_REASONING_COUNT or unique_ratio < MIN_UNIQUE_REASONING_RATIO:
        raise ValueError(
            "reasoning variance is too low: "
            f"{unique_count} unique explanation(s) in a sample of {sample_count} "
            f"(unique_ratio={unique_ratio:.2f}, required>={MIN_UNIQUE_REASONING_RATIO:.2f})."
        )

    LOGGER.debug(
        "Validated reasoning variance",
        extra={
            "count": len(clean_reasonings),
            "sample_count": sample_count,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
        },
    )


def _validate_rank(rank: int) -> int:
    """Validate and return a positive one-based rank."""
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise TypeError(f"rank must be an integer, got {type(rank).__name__}.")
    clean_rank = int(rank)
    if clean_rank < 1:
        raise ValueError(f"rank must be positive, got {clean_rank}.")
    return clean_rank


def _validate_score(score: float) -> float:
    """Validate and return a finite ranking score."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError(f"score must be a finite real number, got {type(score).__name__}.")
    clean_score = float(score)
    if not np.isfinite(clean_score):
        raise ValueError(f"score must be finite, got {clean_score!r}.")
    return clean_score


def _rank_opener(rank: int) -> str:
    """Return deterministic wording for the rank band."""
    if rank <= HIGH_CONFIDENCE_RANK_MAX:
        return STRONG_OPENERS[(rank - 1) % len(STRONG_OPENERS)]
    if rank <= BALANCED_RANK_MAX:
        return BALANCED_OPENERS[(rank - HIGH_CONFIDENCE_RANK_MAX - 1) % len(BALANCED_OPENERS)]
    return CAUTIOUS_OPENERS[(rank - BALANCED_RANK_MAX - 1) % len(CAUTIOUS_OPENERS)]


def _fact_phrases(
    *,
    current_title: str,
    current_company: str,
    years_of_experience: float | None,
) -> list[str]:
    """Build factual profile phrases from available fields."""
    facts: list[str] = []
    if current_title and current_company:
        facts.append(f"current role as {current_title} at {current_company}")
    elif current_title:
        facts.append(f"current role as {current_title}")
    elif current_company:
        facts.append(f"current company {current_company}")

    if years_of_experience is not None:
        facts.append(f"{_format_years(years_of_experience)} of experience")
    return facts


def _signal_text(
    *,
    score: float,
    recruiter_response_rate: float | None,
    notice_period_days: float | None,
) -> str:
    """Build the ranking-signal sentence from present values."""
    signals = [f"normalized ranking score {_format_score(score)}"]

    if recruiter_response_rate is not None:
        signals.append(f"{_format_percent(recruiter_response_rate)} recruiter response rate")
    if notice_period_days is not None:
        signals.append(_format_notice_period(notice_period_days))

    if len(signals) == 1:
        return ""
    return "Ranking signals include " + ", ".join(signals) + "."


def _strength_phrases(
    *,
    current_title: str,
    current_company: str,
    years_of_experience: float | None,
    recruiter_response_rate: float | None,
    notice_period_days: float | None,
    rank: int,
) -> list[str]:
    """Return recruiter-facing strengths grounded in available fields."""
    strengths: list[str] = []
    if current_title and current_company:
        strengths.append("clear current-role context")
    elif current_title:
        strengths.append("relevant current-title signal")
    elif current_company:
        strengths.append("current-company context")

    if years_of_experience is not None and years_of_experience >= EXPERIENCE_STRENGTH_YEARS:
        strengths.append("meaningful experience depth")
    if recruiter_response_rate is not None and recruiter_response_rate >= HIGH_RECRUITER_RESPONSE_RATE:
        strengths.append("strong recruiter responsiveness")
    if notice_period_days is not None and notice_period_days <= PREFERRED_NOTICE_PERIOD_DAYS:
        strengths.append("favorable availability")

    if not strengths and rank <= EXPECTED_TOP_RANK_MAX:
        strengths.append("ranking model placed the profile inside the top candidate set")
    return strengths


def _concern_phrases(
    *,
    recruiter_response_rate: float | None,
    notice_period_days: float | None,
) -> list[str]:
    """Return concern phrases for present signals that cross risk thresholds."""
    concerns: list[str] = []
    if notice_period_days is not None and notice_period_days > HIGH_NOTICE_PERIOD_DAYS:
        concerns.append(f"{_format_notice_period(notice_period_days)} may slow availability")
    if recruiter_response_rate is not None and recruiter_response_rate < LOW_RECRUITER_RESPONSE_RATE:
        concerns.append(f"{_format_percent(recruiter_response_rate)} recruiter response rate may reduce outreach reliability")
    return concerns


def _current_title(candidate: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    """Return the current title if explicitly available or derivable from a current job."""
    title = _first_text(candidate, profile, keys=CURRENT_TITLE_KEYS)
    if title:
        return title

    current_job = _current_job(candidate)
    return _first_text(current_job, keys=CURRENT_JOB_TITLE_KEYS) if current_job else ""


def _current_company(candidate: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    """Return the current company if explicitly available or derivable from a current job."""
    company = _first_text(candidate, profile, keys=CURRENT_COMPANY_KEYS)
    if company:
        return company

    current_job = _current_job(candidate)
    return _first_text(current_job, keys=CURRENT_JOB_COMPANY_KEYS) if current_job else ""


def _current_job(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first career-history record that explicitly appears current."""
    history = candidate.get("career_history")
    if not isinstance(history, Iterable) or isinstance(history, (str, bytes, Mapping)):
        return {}

    for job in history:
        if isinstance(job, Mapping) and _is_current_job(job):
            return job
    return {}


def _is_current_job(job: Mapping[str, Any]) -> bool:
    """Return whether a career-history record explicitly represents the current job."""
    for key in CURRENT_JOB_FLAGS:
        if _truthy(job.get(key)):
            return True
    for key in CURRENT_JOB_END_KEYS:
        value = job.get(key)
        if isinstance(value, str) and value.strip().casefold() in CURRENT_JOB_END_SENTINELS:
            return True
    return False


def _first_text(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty sanitized text from the given mappings."""
    value = _first_value(*mappings, keys=keys)
    return _text_value(value)


def _first_number(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first non-negative numeric value from the given mappings."""
    return _non_negative_number(_first_value(*mappings, keys=keys))


def _first_value(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> object:
    """Return the first non-missing value for any key in any mapping."""
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        for key in keys:
            value = mapping.get(key)
            if not _is_missing(value):
                return value
    return None


def _response_rate(value: object) -> float | None:
    """Parse a recruiter response rate expressed as either 0-1 or 0-100."""
    rate = _non_negative_number(value)
    if rate is None:
        return None
    if rate > 1.0:
        rate /= 100.0
    return min(rate, 1.0)


def _non_negative_number(value: object) -> float | None:
    """Convert a present scalar value to a non-negative finite float."""
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) and numeric_value >= 0.0 else None
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"immediate", "now", "none", "no notice"}:
            return 0.0
        match = NUMBER_RE.search(lowered)
        if match is None:
            return None
        try:
            numeric_value = float(match.group(0))
        except ValueError:
            return None
        return numeric_value if np.isfinite(numeric_value) and numeric_value >= 0.0 else None
    return None


def _format_score(score: float) -> str:
    """Format a model score with a stable number of decimal places."""
    return f"{score:.{SCORE_DECIMAL_PLACES}f}"


def _format_percent(rate: float) -> str:
    """Format a normalized rate as a recruiter-friendly percentage."""
    return f"{rate * 100:.{PERCENT_DECIMAL_PLACES}f}%"


def _format_years(years: float) -> str:
    """Format years of experience without unnecessary decimals."""
    if years.is_integer():
        return f"{int(years)} years"
    return f"{years:.{YEARS_DECIMAL_PLACES}f} years"


def _format_notice_period(days: float) -> str:
    """Format notice period as immediate availability or whole days."""
    if days <= 0.0:
        return "immediate availability"
    rounded_days = int(round(days))
    return f"{rounded_days}-day notice period"


def _validate_reasonings(reasonings: Sequence[str]) -> list[str]:
    """Validate explanation strings and return normalized text."""
    if isinstance(reasonings, (str, bytes)):
        raise TypeError("reasonings must be a sequence of strings, not a scalar string.")
    if not isinstance(reasonings, Sequence):
        raise TypeError(f"reasonings must be a sequence of strings, got {type(reasonings).__name__}.")
    if len(reasonings) < MIN_REASONING_SAMPLE_COUNT:
        raise ValueError(
            f"at least {MIN_REASONING_SAMPLE_COUNT} reasonings are required to validate variance; "
            f"got {len(reasonings)}."
        )

    clean_reasonings: list[str] = []
    for index, reasoning in enumerate(reasonings):
        if not isinstance(reasoning, str):
            raise TypeError(
                "reasonings must contain only strings; "
                f"reasonings[{index}] has type {type(reasoning).__name__}."
            )
        clean_reasoning = _normalize_text(reasoning, max_chars=None)
        if not clean_reasoning:
            raise ValueError(f"reasonings[{index}] must not be blank.")
        clean_reasonings.append(clean_reasoning)
    return clean_reasonings


def _text_value(value: object) -> str:
    """Return sanitized display text for a scalar field."""
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return _normalize_text(value, max_chars=MAX_TEXT_FIELD_CHARS)
    if isinstance(value, Real) and not isinstance(value, bool):
        return _normalize_text(str(value), max_chars=MAX_TEXT_FIELD_CHARS)
    return ""


def _normalize_text(text: str, max_chars: int | None) -> str:
    """Normalize whitespace and remove control characters from display text."""
    clean_text = WHITESPACE_RE.sub(" ", CONTROL_RE.sub(" ", text)).strip()
    if max_chars is None or len(clean_text) <= max_chars:
        return clean_text
    return clean_text[:max_chars].rsplit(" ", 1)[0].strip() or clean_text[:max_chars].strip()


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping for malformed data."""
    return value if isinstance(value, Mapping) else {}


def _truthy(value: object) -> bool:
    """Return whether a loose value represents true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return float(value) > 0.0
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "y", "1", "current", "present"}
    return False


def _is_missing(value: object) -> bool:
    """Return whether a value should be treated as absent."""
    return value is None or value == -1 or (isinstance(value, str) and value.strip() in {"", "-1"})


__all__ = [
    "BALANCED_RANK_MAX",
    "HIGH_CONFIDENCE_RANK_MAX",
    "HIGH_NOTICE_PERIOD_DAYS",
    "LOW_RECRUITER_RESPONSE_RATE",
    "generate_reasoning",
    "validate_reasoning_variance",
]
