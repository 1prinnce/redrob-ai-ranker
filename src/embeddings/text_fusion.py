"""Text-fusion utilities for sentence-transformer embedding inputs."""

from collections.abc import Iterable, Mapping
from typing import Any
import re


MAX_TEXT_CHARS: int = 2000
MAX_SKILLS: int = 15
MAX_JOBS: int = 3

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
YEAR_RE = re.compile(r"(19|20)\d{2}")

HEADLINE_KEYS: tuple[str, ...] = ("headline", "professional_headline", "current_title", "title")
SUMMARY_KEYS: tuple[str, ...] = ("summary", "professional_summary", "profile_summary", "about", "bio")
TITLE_KEYS: tuple[str, ...] = ("title", "job_title", "role", "position")
COMPANY_KEYS: tuple[str, ...] = ("company", "company_name", "employer", "organization")
DESCRIPTION_KEYS: tuple[str, ...] = ("description", "job_description", "summary", "impact", "achievements")
RESPONSIBILITY_KEYS: tuple[str, ...] = ("responsibilities", "responsibility", "duties", "projects")
NAME_KEYS: tuple[str, ...] = ("name", "full_name", "candidate_name")
LOCATION_KEYS: tuple[str, ...] = (
    "location",
    "current_location",
    "city",
    "current_city",
    "state",
    "current_state",
    "country",
    "current_country",
)
ITEM_NAME_KEYS: tuple[str, ...] = (
    "name",
    "title",
    "skill",
    "skill_name",
    "language",
    "language_name",
    "certification",
    "certification_name",
    "text",
    "value",
)


def candidate_text(candidate: Mapping[str, Any] | None) -> str:
    """Build a deterministic, weighted embedding document from a candidate profile."""
    if not isinstance(candidate, Mapping):
        return ""

    profile = _as_mapping(candidate.get("profile"))
    forbidden = _forbidden_values(candidate, profile)
    blocks: list[str] = []

    headline = _first_text(candidate, profile, keys=HEADLINE_KEYS)
    if headline:
        blocks.append(_repeat_text("Headline: " + headline, times=2))

    summary = _first_text(candidate, profile, keys=SUMMARY_KEYS)
    if summary:
        blocks.append("Summary: " + summary)

    for job in _recent_jobs(candidate):
        job_block = _job_text(job)
        if job_block:
            blocks.append(job_block)

    skills = _unique(_extract_names(candidate.get("skills", profile.get("skills"))))[:MAX_SKILLS]
    if skills:
        blocks.append("Skills: " + ", ".join(skills))

    certifications = _unique(_extract_names(candidate.get("certifications", profile.get("certifications"))))
    if certifications:
        blocks.append("Certifications: " + ", ".join(certifications))

    languages = _unique(_extract_names(candidate.get("languages", profile.get("languages"))))
    if languages:
        blocks.append("Languages: " + ", ".join(languages))

    cleaned_blocks = _dedupe_blocks(_clean_block(block, forbidden) for block in blocks)
    return _cap_text(_normalize_whitespace(" ".join(cleaned_blocks)), MAX_TEXT_CHARS)


def jd_text(jd_string: str) -> str:
    """Clean and de-duplicate a job description for embedding generation."""
    if not isinstance(jd_string, str):
        return ""

    sections = _split_sections(jd_string)
    cleaned_sections = _dedupe_blocks(_clean_block(section, forbidden_values=set()) for section in sections)
    return _normalize_whitespace(" ".join(cleaned_sections))


def batch_candidate_texts(candidates: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return embedding text for each candidate using an efficient list comprehension."""
    return [candidate_text(candidate) for candidate in candidates]


def _job_text(job: Mapping[str, Any]) -> str:
    """Return a weighted text block for a single career-history item."""
    title = _first_mapping_text(job, TITLE_KEYS)
    company = _first_mapping_text(job, COMPANY_KEYS)
    responsibilities = _first_mapping_text(job, RESPONSIBILITY_KEYS)
    description = _first_mapping_text(job, DESCRIPTION_KEYS)

    parts: list[str] = []
    if title:
        parts.append("Title: " + title)
    if company:
        parts.append("Company: " + company)
    if responsibilities:
        parts.append("Responsibilities: " + responsibilities)
    if description:
        parts.append(_repeat_text("Description: " + description, times=2))
    elif responsibilities:
        parts.append("Description: " + responsibilities)

    return ". ".join(parts)


def _recent_jobs(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the top most-recent career-history records in deterministic order."""
    history = candidate.get("career_history", [])
    if not isinstance(history, list | tuple):
        return []

    jobs = [(index, job) for index, job in enumerate(history) if isinstance(job, Mapping)]
    ranked = sorted(jobs, key=lambda item: _job_sort_key(item[0], item[1]), reverse=True)
    return [job for _, job in ranked[:MAX_JOBS]]


def _job_sort_key(index: int, job: Mapping[str, Any]) -> tuple[int, tuple[int, int, int], tuple[int, int, int], int]:
    """Return a stable recency key for career-history sorting."""
    current = 1 if _truthy(job.get("is_current", job.get("current"))) else 0
    end_date = _date_key(job.get("end_date", job.get("ended_at", job.get("to"))))
    start_date = _date_key(job.get("start_date", job.get("started_at", job.get("from"))))
    return current, end_date, start_date, -index


def _date_key(value: object) -> tuple[int, int, int]:
    """Parse a lightweight sortable date key from common resume date values."""
    if _is_missing(value):
        return 0, 0, 0
    if hasattr(value, "year"):
        return int(getattr(value, "year", 0)), int(getattr(value, "month", 1)), int(getattr(value, "day", 1))
    if isinstance(value, int):
        return value, 12, 31
    if not isinstance(value, str):
        return 0, 0, 0

    lowered = value.strip().casefold()
    if lowered in {"present", "current", "now"}:
        return 9999, 12, 31
    year_match = YEAR_RE.search(lowered)
    if year_match is None:
        return 0, 0, 0
    year = int(year_match.group(0))
    month = _month_number(lowered)
    return year, month, 31


def _month_number(text: str) -> int:
    """Return a best-effort month number from text, defaulting to year-end."""
    month_names = (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    )
    for index, name in enumerate(month_names, start=1):
        if name in text:
            return index
    numeric_month = re.search(r"(?:^|\D)(1[0-2]|0?[1-9])(?:\D|$)", text)
    return int(numeric_month.group(1)) if numeric_month else 12


def _first_text(primary: Mapping[str, Any], secondary: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first text value found in either primary or secondary mapping."""
    return _first_mapping_text(primary, keys) or _first_mapping_text(secondary, keys)


def _first_mapping_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first normalized text field found for the provided keys."""
    for key in keys:
        text = _value_to_text(mapping.get(key))
        if text:
            return text
    return ""


def _value_to_text(value: object) -> str:
    """Convert common profile values into normalized plain text."""
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        return _normalize_whitespace(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list | tuple):
        return _normalize_whitespace(" ".join(_value_to_text(item) for item in value))
    if isinstance(value, Mapping):
        return _first_mapping_text(value, ITEM_NAME_KEYS)
    return ""


def _extract_names(value: object) -> list[str]:
    """Extract ordered display names from strings, lists, or mapping-shaped records."""
    if _is_missing(value):
        return []
    if isinstance(value, str):
        return [_normalize_whitespace(value)]
    if isinstance(value, list | tuple):
        names: list[str] = []
        for item in value:
            names.extend(_extract_names(item))
        return names
    if isinstance(value, Mapping):
        named_value = _first_mapping_text(value, ITEM_NAME_KEYS)
        if named_value:
            return [named_value]
        names = []
        for nested_value in value.values():
            names.extend(_extract_names(nested_value))
        return names
    return []


def _forbidden_values(candidate: Mapping[str, Any], profile: Mapping[str, Any]) -> set[str]:
    """Collect exact candidate name and location values to remove from embedding text."""
    forbidden: set[str] = set()
    for source in (candidate, profile):
        for key in NAME_KEYS:
            for clean_value in _string_values(source.get(key)):
                if len(clean_value) < 3:
                    continue
                forbidden.add(clean_value)
                forbidden.update(part for part in clean_value.split() if len(part) >= 3)
        for key in LOCATION_KEYS:
            for clean_value in _string_values(source.get(key)):
                if len(clean_value) >= 3:
                    forbidden.add(clean_value)

    first_name = _value_to_text(profile.get("first_name"))
    last_name = _value_to_text(profile.get("last_name"))
    full_name = _normalize_whitespace(f"{first_name} {last_name}")
    if len(full_name) >= 3:
        forbidden.add(full_name)
    for name_part in (first_name, last_name):
        if len(name_part) >= 3:
            forbidden.add(name_part)
    return forbidden


def _clean_block(text: str, forbidden_values: set[str]) -> str:
    """Normalize one block and remove exact forbidden values."""
    clean_text = _normalize_whitespace(text)
    for value in sorted(forbidden_values, key=len, reverse=True):
        clean_text = re.sub(re.escape(value), " ", clean_text, flags=re.IGNORECASE)
    return _normalize_whitespace(clean_text)


def _string_values(value: object) -> list[str]:
    """Return normalized strings contained in a scalar, list, or mapping."""
    if _is_missing(value):
        return []
    if isinstance(value, str):
        return [_normalize_whitespace(value)]
    if isinstance(value, list | tuple):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    if isinstance(value, Mapping):
        values = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    return []


def _split_sections(text: str) -> list[str]:
    """Split raw text into candidate sections suitable for exact de-duplication."""
    normalized = _to_utf8_text(text).replace("\r\n", "\n").replace("\r", "\n")
    return [section for section in re.split(r"\n+", normalized) if section.strip()]


def _dedupe_blocks(blocks: Iterable[str]) -> list[str]:
    """Remove duplicate normalized blocks while preserving first occurrence order."""
    seen: set[str] = set()
    unique_blocks: list[str] = []
    for block in blocks:
        normalized = _normalize_whitespace(block)
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique_blocks.append(normalized)
    return unique_blocks


def _repeat_text(text: str, times: int) -> str:
    """Repeat a text fragment a fixed number of times for embedding weighting."""
    clean_text = _normalize_whitespace(text)
    return " ".join(clean_text for _ in range(max(1, times)))


def _unique(values: Iterable[str]) -> list[str]:
    """Return ordered unique normalized values."""
    return _dedupe_blocks(values)


def _normalize_whitespace(text: str) -> str:
    """Return clean UTF-8 text with collapsed whitespace and no control characters."""
    utf8_text = _to_utf8_text(text)
    return WHITESPACE_RE.sub(" ", CONTROL_RE.sub(" ", utf8_text)).strip()


def _to_utf8_text(text: str) -> str:
    """Round-trip text through UTF-8 to drop invalid surrogate data."""
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping for malformed data."""
    return value if isinstance(value, Mapping) else {}


def _truthy(value: object) -> bool:
    """Return whether a loose profile value represents true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value > 0
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "y", "1", "current", "present"}
    return False


def _is_missing(value: object) -> bool:
    """Return whether a value should be treated as absent."""
    return value is None or value == -1 or (isinstance(value, str) and value.strip() in {"", "-1"})


def _cap_text(text: str, max_chars: int) -> str:
    """Cap text length while preferring to end on a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated if len(truncated) >= int(max_chars * 0.8) else text[:max_chars].strip()
