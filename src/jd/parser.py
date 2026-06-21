"""Deterministic job-description parsing for retrieval and ranking signals.

The parser uses regex-driven extraction only. It is designed for production
ranking pipelines where the same messy job description must always produce the
same structured intent signals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import logging
import re
from typing import Final, TypedDict

from src.jd.constants import (
    DISQUALIFIER_KEYWORDS,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    PENALIZED_PRIMARY_SKILLS,
)


LOGGER = logging.getLogger(__name__)

ExperienceRange = tuple[float | None, float | None]


class ParsedJD(TypedDict):
    """Structured job-description intent used by ranking components."""

    must_haves: list[str]
    nice_to_haves: list[str]
    disqualifiers: list[str]
    preferred_locations: list[str]
    experience_range: ExperienceRange
    domain_focus: list[str]
    raw_requirements: list[str]


class JDComplexity(TypedDict):
    """Summary complexity features derived from a job description."""

    complexity_score: float
    skill_count: int
    experience_requirements: int
    domain_count: int


@dataclass(frozen=True)
class PatternSpec:
    """Canonical label and regex patterns used for ordered extraction."""

    label: str
    patterns: tuple[re.Pattern[str], ...]
    order: int


@dataclass(frozen=True)
class ExperienceMatch:
    """One normalized experience expression found in a job description."""

    minimum: float
    maximum: float | None
    start: int
    end: int
    text: str


MAX_REQUIREMENT_LINE_CHARS: Final[int] = 360
MIN_REQUIREMENT_LINE_CHARS: Final[int] = 3

SKILL_COMPLEXITY_CAP: Final[int] = 10
EXPERIENCE_COMPLEXITY_CAP: Final[int] = 3
DOMAIN_COMPLEXITY_CAP: Final[int] = 4
SKILL_COMPLEXITY_WEIGHT: Final[float] = 0.45
EXPERIENCE_COMPLEXITY_WEIGHT: Final[float] = 0.20
DOMAIN_COMPLEXITY_WEIGHT: Final[float] = 0.35
COMPLEXITY_DECIMAL_PLACES: Final[int] = 4

MUST_HAVE_SYNONYMS: Final[Mapping[str, tuple[str, ...]]] = {
    "embedding retrieval": (
        "embedding retrieval",
        "embedding based retrieval",
        "embedding-based retrieval",
        "dense retrieval",
        "semantic retrieval",
        "vector retrieval",
    ),
    "vector database": (
        "vector database",
        "vector databases",
        "vector db",
        "vector dbs",
        "vector store",
        "vector stores",
        "vector index",
        "vectordb",
    ),
    "python": ("python", "python3"),
    "evaluation framework": (
        "evaluation framework",
        "eval framework",
        "evaluation pipeline",
        "evaluation pipelines",
        "offline evaluation",
        "ranking evaluation",
        "search evaluation",
    ),
    "ndcg": ("ndcg", "normalized discounted cumulative gain"),
    "ranking system": (
        "ranking system",
        "ranking systems",
        "ranker",
        "learning to rank",
        "ltr",
        "ranking model",
        "ranking models",
    ),
    "sentence transformers": (
        "sentence transformers",
        "sentence-transformers",
        "sentence transformer",
        "sbert",
        "sentence embedding model",
    ),
    "faiss": ("faiss", "facebook ai similarity search"),
    "pinecone": ("pinecone",),
}

NICE_TO_HAVE_SYNONYMS: Final[Mapping[str, tuple[str, ...]]] = {
    "semantic search": ("semantic search", "neural search", "vector search"),
    "hybrid search": ("hybrid search", "keyword and vector search", "lexical and semantic search"),
    "reranking": ("reranking", "re ranking", "re-ranking", "reranker", "re-ranker"),
    "cross encoder": ("cross encoder", "cross-encoder", "crossencoder"),
    "bi encoder": ("bi encoder", "bi-encoder", "biencoder", "dual encoder", "dual-encoder"),
    "ann search": ("ann search", "approximate nearest neighbor", "approximate nearest neighbour"),
    "hnsw": ("hnsw", "hierarchical navigable small world"),
    "bm25": ("bm25", "okapi bm25"),
    "recall evaluation": ("recall evaluation", "recall@k", "recall at k"),
    "mrr": ("mrr", "mean reciprocal rank"),
    "information retrieval": ("information retrieval", "ir systems", "ir system"),
    "rag": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "langchain": ("langchain", "lang chain"),
    "llamaindex": ("llamaindex", "llama index"),
    "elasticsearch": ("elasticsearch", "elastic search"),
    "opensearch": ("opensearch", "open search"),
    "weaviate": ("weaviate",),
    "milvus": ("milvus",),
    "qdrant": ("qdrant",),
}

LOCATION_SYNONYMS: Final[Mapping[str, tuple[str, ...]]] = {
    "Bengaluru": ("bengaluru", "bangalore"),
    "Pune": ("pune",),
    "Hyderabad": ("hyderabad",),
    "Mumbai": ("mumbai",),
    "Delhi": ("delhi", "new delhi", "delhi ncr", "delhi-ncr"),
    "Noida": ("noida",),
    "Gurugram": ("gurugram", "gurgaon"),
    "Remote": ("remote", "work from home", "wfh"),
    "Hybrid": ("hybrid", "hybrid work", "hybrid mode"),
}

DOMAIN_SYNONYMS: Final[Mapping[str, tuple[str, ...]]] = {
    "Retrieval Systems": (
        "retrieval systems",
        "retrieval system",
        "embedding retrieval",
        "dense retrieval",
        "semantic retrieval",
        "vector retrieval",
    ),
    "Search": (
        "search",
        "semantic search",
        "hybrid search",
        "search relevance",
        "search engine",
        "full text search",
        "keyword search",
    ),
    "Ranking": (
        "ranking",
        "ranking systems",
        "ranking system",
        "learning to rank",
        "ltr",
        "reranking",
        "re-ranking",
        "ranker",
    ),
    "Recommendation Systems": (
        "recommendation systems",
        "recommendation system",
        "recommender systems",
        "recommender system",
        "recsys",
        "personalization",
    ),
    "Vector Databases": (
        "vector databases",
        "vector database",
        "vector db",
        "vector store",
        "vector stores",
        "vector index",
        "faiss",
        "pinecone",
        "milvus",
        "qdrant",
        "weaviate",
    ),
    "LLM Applications": (
        "llm applications",
        "llm application",
        "large language models",
        "large language model",
        "generative ai",
        "chatbot",
        "ai agents",
        "agentic",
    ),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "AI Infrastructure": (
        "ai infrastructure",
        "ml infrastructure",
        "ml platform",
        "model serving",
        "inference platform",
        "feature store",
        "mlops",
    ),
}

DISQUALIFIER_SYNONYMS: Final[Mapping[str, tuple[str, ...]]] = {
    "consulting-only": (
        "consulting-only",
        "consulting only",
        "pure consulting",
        "staff augmentation",
        "body shopping",
        "outsourcing delivery",
        "implementation partner",
    ),
    "research-only": (
        "research-only",
        "research only",
        "publication-focused",
        "publication focused",
        "phd researcher",
        "postdoctoral researcher",
        "academic researcher",
    ),
    "academia-only": (
        "academia-only",
        "academia only",
        "faculty",
        "professor",
        "lecturer",
        "university researcher",
    ),
}

KEEP_SECTION_HEADINGS: Final[frozenset[str]] = frozenset(
    {
        "requirements",
        "requirement",
        "required skills",
        "must have",
        "must haves",
        "minimum qualifications",
        "basic qualifications",
        "qualifications",
        "preferred qualifications",
        "nice to have",
        "nice to haves",
        "responsibilities",
        "role responsibilities",
        "what you will do",
        "what youll do",
        "what you'll do",
        "you will",
        "skills",
        "technical skills",
        "experience",
        "about the role",
        "role overview",
    }
)

SKIP_SECTION_HEADINGS: Final[frozenset[str]] = frozenset(
    {
        "about us",
        "about the company",
        "company overview",
        "who we are",
        "why join us",
        "why join",
        "benefits",
        "perks",
        "compensation",
        "diversity",
        "diversity and inclusion",
        "equal opportunity",
        "eeo",
        "legal",
        "legal disclaimer",
        "privacy",
        "privacy notice",
        "application process",
    }
)

BOILERPLATE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bequal opportunity employer\b", re.IGNORECASE),
    re.compile(r"\ball qualified applicants\b", re.IGNORECASE),
    re.compile(r"\bdiversity (?:and|&) inclusion\b", re.IGNORECASE),
    re.compile(r"\bwe (?:celebrate|value|embrace) diversity\b", re.IGNORECASE),
    re.compile(r"\breasonable accommodation\b", re.IGNORECASE),
    re.compile(r"\bprivacy policy\b", re.IGNORECASE),
    re.compile(r"\blegal disclaimer\b", re.IGNORECASE),
    re.compile(r"\bbackground check\b", re.IGNORECASE),
    re.compile(r"\bbenefits include\b", re.IGNORECASE),
    re.compile(r"\bhealth insurance\b", re.IGNORECASE),
    re.compile(r"\bfast[-\s]?growing\b", re.IGNORECASE),
    re.compile(r"\bventure[-\s]?backed\b", re.IGNORECASE),
    re.compile(r"\bjoin our (?:team|mission|journey)\b", re.IGNORECASE),
    re.compile(r"\bfounded in \d{4}\b", re.IGNORECASE),
)

REQUIREMENT_SIGNAL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?:must|required|requirement|qualification|experience|expertise)\b", re.IGNORECASE),
    re.compile(r"\b(?:build|design|develop|deploy|scale|optimi[sz]e|evaluate|own|lead)\b", re.IGNORECASE),
    re.compile(r"\b(?:responsible for|you will|work on|familiarity with|proficiency in)\b", re.IGNORECASE),
)

CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
BULLET_RE: Final[re.Pattern[str]] = re.compile(r"[•●▪◦]")
LEADING_BULLET_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:[-*?]+|\d+[.)])\s+")
SENTENCE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9])")
HEADING_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(r"^\s*([^:]{2,80}):\s*(.*)$")
NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

EXPERIENCE_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<min>\d{1,2}(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
    r"(?P<max>\d{1,2}(?:\.\d+)?)\s*(?:\+?\s*)?"
    r"(?:years?|yrs?)\b(?:\s+(?:of\s+)?(?:experience|exp))?",
    re.IGNORECASE,
)
EXPERIENCE_PLUS_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<min>\d{1,2}(?:\.\d+)?)\s*\+\s*"
    r"(?:years?|yrs?)\b(?:\s+(?:of\s+)?(?:experience|exp))?",
    re.IGNORECASE,
)
EXPERIENCE_MINIMUM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:minimum|min\.?|at\s+least|more\s+than|over)\s+(?:of\s+)?"
    r"(?P<min>\d{1,2}(?:\.\d+)?)\s*\+?\s*"
    r"(?:years?|yrs?)\b(?:\s+(?:of\s+)?(?:experience|exp))?",
    re.IGNORECASE,
)
EXPERIENCE_EXACT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<min>\d{1,2}(?:\.\d+)?)\s*(?:years?|yrs?)\s+"
    r"(?:of\s+)?(?:relevant\s+)?(?:professional\s+)?(?:work\s+)?(?:experience|exp)\b",
    re.IGNORECASE,
)
EXPERIENCE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:experience|exp)\s*(?:of|:)?\s*"
    r"(?P<min>\d{1,2}(?:\.\d+)?)\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

EXPERIENCE_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("range", EXPERIENCE_RANGE_RE),
    ("minimum", EXPERIENCE_MINIMUM_RE),
    ("plus", EXPERIENCE_PLUS_RE),
    ("exact", EXPERIENCE_EXACT_RE),
    ("exact", EXPERIENCE_PREFIX_RE),
)

MUST_HAVE_SPECS: Final[tuple[PatternSpec, ...]] = ()
NICE_TO_HAVE_SPECS: Final[tuple[PatternSpec, ...]] = ()
LOCATION_SPECS: Final[tuple[PatternSpec, ...]] = ()
DOMAIN_SPECS: Final[tuple[PatternSpec, ...]] = ()
DISQUALIFIER_SPECS: Final[tuple[PatternSpec, ...]] = ()
TECHNICAL_SIGNAL_PATTERNS: Final[tuple[re.Pattern[str], ...]] = ()


def parse_jd(jd_text: str) -> ParsedJD:
    """Parse an unstructured job description into ranking intent signals.

    Args:
        jd_text: Raw job description text from a recruiter, ATS, or generated
            source.

    Returns:
        A dictionary containing must-have skills, nice-to-have skills,
        disqualifying signals, preferred locations, normalized experience range,
        ordered domain focus, and cleaned raw requirement lines.

    Raises:
        TypeError: If ``jd_text`` is not a string.
    """
    normalized_text = _validate_and_normalize_jd_text(jd_text)
    if not normalized_text:
        return _empty_parsed_jd()

    experience_matches = _extract_experience_matches(normalized_text)
    parsed: ParsedJD = {
        "must_haves": _extract_ordered_labels(normalized_text, _must_have_specs()),
        "nice_to_haves": _extract_ordered_labels(normalized_text, _nice_to_have_specs()),
        "disqualifiers": _extract_disqualifiers(normalized_text),
        "preferred_locations": _extract_ordered_labels(normalized_text, _location_specs()),
        "experience_range": _experience_range(experience_matches),
        "domain_focus": _extract_ordered_labels(normalized_text, _domain_specs()),
        "raw_requirements": _extract_requirement_lines(jd_text),
    }

    LOGGER.debug(
        "Parsed job description",
        extra={
            "must_have_count": len(parsed["must_haves"]),
            "nice_to_have_count": len(parsed["nice_to_haves"]),
            "domain_count": len(parsed["domain_focus"]),
            "experience_requirements": len(experience_matches),
        },
    )
    return parsed


def extract_requirements_text(jd_text: str) -> str:
    """Return dense requirements text optimized for embedding generation.

    The output removes common boilerplate, company marketing, legal language,
    diversity statements, and repeated sections while preserving technical
    requirements, responsibilities, skills, and experience signals.

    Args:
        jd_text: Raw job description text.

    Returns:
        Cleaned, semantically dense text suitable for sentence-transformer
        embedding input.

    Raises:
        TypeError: If ``jd_text`` is not a string.
    """
    _validate_and_normalize_jd_text(jd_text)
    requirement_lines = _extract_requirement_lines(jd_text)
    if not requirement_lines:
        requirement_lines = _fallback_non_boilerplate_lines(jd_text)

    dense_text = _normalize_whitespace(". ".join(_strip_terminal_punctuation(line) for line in requirement_lines))
    LOGGER.debug(
        "Extracted requirements text",
        extra={"line_count": len(requirement_lines), "char_count": len(dense_text)},
    )
    return dense_text


def score_jd_complexity(jd_text: str) -> JDComplexity:
    """Compute lightweight complexity features from a job description.

    Args:
        jd_text: Raw job description text.

    Returns:
        Dictionary with a normalized complexity score and the component counts
        used to compute it.

    Raises:
        TypeError: If ``jd_text`` is not a string.
    """
    normalized_text = _validate_and_normalize_jd_text(jd_text)
    if not normalized_text:
        return {
            "complexity_score": 0.0,
            "skill_count": 0,
            "experience_requirements": 0,
            "domain_count": 0,
        }

    must_haves = _extract_ordered_labels(normalized_text, _must_have_specs())
    nice_to_haves = _extract_ordered_labels(normalized_text, _nice_to_have_specs())
    experience_count = len(_extract_experience_matches(normalized_text))
    domain_count = len(_extract_ordered_labels(normalized_text, _domain_specs()))
    skill_count = len(must_haves) + len(nice_to_haves)

    skill_component = min(skill_count / SKILL_COMPLEXITY_CAP, 1.0) * SKILL_COMPLEXITY_WEIGHT
    experience_component = (
        min(experience_count / EXPERIENCE_COMPLEXITY_CAP, 1.0) * EXPERIENCE_COMPLEXITY_WEIGHT
    )
    domain_component = min(domain_count / DOMAIN_COMPLEXITY_CAP, 1.0) * DOMAIN_COMPLEXITY_WEIGHT
    complexity_score = round(
        skill_component + experience_component + domain_component,
        COMPLEXITY_DECIMAL_PLACES,
    )

    complexity: JDComplexity = {
        "complexity_score": complexity_score,
        "skill_count": skill_count,
        "experience_requirements": experience_count,
        "domain_count": domain_count,
    }
    LOGGER.debug("Scored JD complexity", extra=complexity)
    return complexity


def _empty_parsed_jd() -> ParsedJD:
    """Return an empty parse result with a stable schema."""
    return {
        "must_haves": [],
        "nice_to_haves": [],
        "disqualifiers": [],
        "preferred_locations": [],
        "experience_range": (None, None),
        "domain_focus": [],
        "raw_requirements": [],
    }


def _validate_and_normalize_jd_text(jd_text: str) -> str:
    """Validate raw JD text and return a normalized string for matching."""
    if not isinstance(jd_text, str):
        raise TypeError(f"jd_text must be a string, got {type(jd_text).__name__}.")
    return _normalize_whitespace(jd_text)


def _must_have_specs() -> tuple[PatternSpec, ...]:
    """Return compiled must-have skill patterns."""
    global MUST_HAVE_SPECS
    if not MUST_HAVE_SPECS:
        MUST_HAVE_SPECS = _compile_specs(MUST_HAVE_SKILLS, MUST_HAVE_SYNONYMS)
    return MUST_HAVE_SPECS


def _nice_to_have_specs() -> tuple[PatternSpec, ...]:
    """Return compiled nice-to-have skill patterns."""
    global NICE_TO_HAVE_SPECS
    if not NICE_TO_HAVE_SPECS:
        NICE_TO_HAVE_SPECS = _compile_specs(NICE_TO_HAVE_SKILLS, NICE_TO_HAVE_SYNONYMS)
    return NICE_TO_HAVE_SPECS


def _location_specs() -> tuple[PatternSpec, ...]:
    """Return compiled preferred-location patterns."""
    global LOCATION_SPECS
    if not LOCATION_SPECS:
        LOCATION_SPECS = _compile_specs(tuple(LOCATION_SYNONYMS), LOCATION_SYNONYMS)
    return LOCATION_SPECS


def _domain_specs() -> tuple[PatternSpec, ...]:
    """Return compiled domain-focus patterns."""
    global DOMAIN_SPECS
    if not DOMAIN_SPECS:
        DOMAIN_SPECS = _compile_specs(tuple(DOMAIN_SYNONYMS), DOMAIN_SYNONYMS)
    return DOMAIN_SPECS


def _disqualifier_specs() -> tuple[PatternSpec, ...]:
    """Return compiled disqualifier signal patterns."""
    global DISQUALIFIER_SPECS
    if not DISQUALIFIER_SPECS:
        merged: dict[str, tuple[str, ...]] = {label: synonyms for label, synonyms in DISQUALIFIER_SYNONYMS.items()}
        for keyword in DISQUALIFIER_KEYWORDS:
            canonical = _canonical_disqualifier(keyword)
            merged.setdefault(canonical, ())
            merged[canonical] = tuple(dict.fromkeys((*merged[canonical], keyword)))
        for domain in PENALIZED_PRIMARY_SKILLS:
            label = f"unrelated domain: {domain}"
            merged[label] = (domain,)
        DISQUALIFIER_SPECS = _compile_specs(tuple(merged), merged)
    return DISQUALIFIER_SPECS


def _technical_signal_patterns() -> tuple[re.Pattern[str], ...]:
    """Return compiled patterns that indicate a technical requirement line."""
    global TECHNICAL_SIGNAL_PATTERNS
    if not TECHNICAL_SIGNAL_PATTERNS:
        patterns: list[re.Pattern[str]] = []
        for specs in (
            _must_have_specs(),
            _nice_to_have_specs(),
            _domain_specs(),
            _location_specs(),
        ):
            for spec in specs:
                patterns.extend(spec.patterns)
        TECHNICAL_SIGNAL_PATTERNS = tuple(patterns)
    return TECHNICAL_SIGNAL_PATTERNS


def _compile_specs(labels: Sequence[str], synonym_map: Mapping[str, tuple[str, ...]]) -> tuple[PatternSpec, ...]:
    """Compile canonical labels and synonyms into deterministic pattern specs."""
    specs: list[PatternSpec] = []
    for order, label in enumerate(labels):
        phrases = tuple(dict.fromkeys((label, *synonym_map.get(label, ()))))
        patterns = tuple(_compile_phrase_pattern(phrase) for phrase in phrases if phrase.strip())
        specs.append(PatternSpec(label=label, patterns=patterns, order=order))
    return tuple(specs)


def _compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Compile a phrase pattern tolerant to whitespace, slashes, and hyphens."""
    normalized_phrase = phrase.casefold().strip()
    if normalized_phrase == "hybrid":
        return re.compile(r"(?<![a-z0-9])hybrid(?![\s\-/_]+search)(?![a-z0-9])", re.IGNORECASE)
    if normalized_phrase == "remote":
        return re.compile(r"(?<![a-z0-9])remote(?![\s\-/_]+sensing)(?![a-z0-9])", re.IGNORECASE)

    tokens = re.findall(r"[a-z0-9]+", phrase.casefold())
    if not tokens:
        return re.compile(r"a\A")

    separator = r"[\s\-/_@]*" if len(tokens) == 1 else r"[\s\-/_@]+"
    body = separator.join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def _extract_ordered_labels(text: str, specs: Sequence[PatternSpec]) -> list[str]:
    """Extract canonical labels sorted by first occurrence in text."""
    first_matches: dict[str, tuple[int, int]] = {}
    for spec in specs:
        first_start: int | None = None
        for pattern in spec.patterns:
            match = pattern.search(text)
            if match is not None and (first_start is None or match.start() < first_start):
                first_start = match.start()
        if first_start is not None:
            first_matches[spec.label] = (first_start, spec.order)

    return [
        label
        for label, _ in sorted(
            first_matches.items(),
            key=lambda item: (item[1][0], item[1][1], item[0].casefold()),
        )
    ]


def _extract_disqualifiers(text: str) -> list[str]:
    """Extract disqualifying signals, including unrelated-domain focus."""
    return _extract_ordered_labels(text, _disqualifier_specs())


def _canonical_disqualifier(keyword: str) -> str:
    """Map known disqualifier keywords to stable high-level labels."""
    normalized = _normalize_key(keyword)
    if normalized in {
        "consulting only",
        "staff augmentation",
        "body shopping",
        "outsourcing delivery",
        "implementation partner",
    }:
        return "consulting-only"
    if normalized in {
        "research only",
        "publication focused",
        "phd researcher",
        "postdoctoral researcher",
        "academic researcher",
    }:
        return "research-only"
    if normalized in {"faculty", "professor", "lecturer", "academia only"}:
        return "academia-only"
    return keyword


def _extract_experience_matches(text: str) -> list[ExperienceMatch]:
    """Extract unique experience expressions from normalized text."""
    matches: list[ExperienceMatch] = []
    occupied_spans: list[tuple[int, int]] = []

    for kind, pattern in EXPERIENCE_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if _overlaps(span, occupied_spans):
                continue

            minimum = _safe_float(match.group("min"))
            if minimum is None:
                continue

            maximum: float | None = None
            if kind == "range":
                maximum = _safe_float(match.group("max"))
                if maximum is None:
                    continue
                if maximum < minimum:
                    minimum, maximum = maximum, minimum

            matches.append(
                ExperienceMatch(
                    minimum=minimum,
                    maximum=maximum,
                    start=span[0],
                    end=span[1],
                    text=_normalize_whitespace(match.group(0)),
                )
            )
            occupied_spans.append(span)

    matches.sort(key=lambda item: (item.start, item.end))
    return _dedupe_experience_matches(matches)


def _experience_range(matches: Sequence[ExperienceMatch]) -> ExperienceRange:
    """Convert matched experience expressions into one normalized range."""
    if not matches:
        return (None, None)

    minimum = min(match.minimum for match in matches)
    explicit_maximums = [match.maximum for match in matches if match.maximum is not None]
    maximum = max(explicit_maximums) if explicit_maximums else None
    return (_normalize_year_number(minimum), _normalize_year_number(maximum))


def _dedupe_experience_matches(matches: Sequence[ExperienceMatch]) -> list[ExperienceMatch]:
    """Remove repeated experience expressions while preserving order."""
    seen: set[str] = set()
    unique_matches: list[ExperienceMatch] = []
    for match in matches:
        key = match.text.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_matches.append(match)
    return unique_matches


def _overlaps(span: tuple[int, int], occupied_spans: Sequence[tuple[int, int]]) -> bool:
    """Return whether a span overlaps any previously accepted span."""
    start, end = span
    return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_spans)


def _extract_requirement_lines(jd_text: str) -> list[str]:
    """Extract cleaned technical requirement and responsibility lines."""
    lines = _split_candidate_lines(jd_text)
    requirement_lines: list[str] = []
    seen: set[str] = set()
    active_keep_section = False
    active_skip_section = False

    for raw_line in lines:
        line = _clean_line(raw_line)
        if len(line) < MIN_REQUIREMENT_LINE_CHARS:
            continue

        heading, content = _split_heading_prefix(line)
        heading_kind = _heading_kind(heading) if heading else ""

        if heading_kind == "skip":
            active_keep_section = False
            active_skip_section = True
            if not content:
                continue
            line = content
        elif heading_kind == "keep":
            active_keep_section = True
            active_skip_section = False
            if not content:
                continue
            line = content
        elif _is_standalone_heading(line):
            active_keep_section = heading_kind == "keep"
            active_skip_section = heading_kind == "skip"
            continue

        if _is_boilerplate_line(line):
            continue
        if active_skip_section and not _is_requirement_like(line):
            continue
        if not active_keep_section and not _is_requirement_like(line):
            continue

        _append_unique_line(requirement_lines, seen, line)

    return requirement_lines


def _fallback_non_boilerplate_lines(jd_text: str) -> list[str]:
    """Return cleaned non-boilerplate lines when no requirement section is found."""
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in _split_candidate_lines(jd_text):
        line = _clean_line(raw_line)
        if len(line) < MIN_REQUIREMENT_LINE_CHARS or _is_boilerplate_line(line):
            continue
        heading, content = _split_heading_prefix(line)
        if _heading_kind(heading) == "skip":
            continue
        _append_unique_line(lines, seen, content or line)
    return lines


def _split_candidate_lines(jd_text: str) -> list[str]:
    """Split messy JD text into candidate requirement lines."""
    text = CONTROL_RE.sub(" ", jd_text).replace("\r\n", "\n").replace("\r", "\n")
    text = BULLET_RE.sub("\n", text)
    candidate_lines: list[str] = []
    for block in text.split("\n"):
        clean_block = _clean_line(block)
        if not clean_block:
            continue
        split_parts = (
            SENTENCE_SPLIT_RE.split(clean_block)
            if len(clean_block) > MAX_REQUIREMENT_LINE_CHARS
            else [clean_block]
        )
        candidate_lines.extend(part for part in split_parts if part.strip())
    return candidate_lines


def _clean_line(line: str) -> str:
    """Normalize one JD line and remove leading bullet syntax."""
    return _normalize_whitespace(LEADING_BULLET_RE.sub("", line))


def _split_heading_prefix(line: str) -> tuple[str, str]:
    """Split a section heading prefix from inline content when present."""
    match = HEADING_SEPARATOR_RE.match(line)
    if match is None:
        return "", ""
    heading = _normalize_key(match.group(1))
    content = _normalize_whitespace(match.group(2))
    return heading, content


def _heading_kind(heading: str) -> str:
    """Classify a normalized section heading as keep, skip, or unknown."""
    if not heading:
        return ""
    if heading in KEEP_SECTION_HEADINGS:
        return "keep"
    if heading in SKIP_SECTION_HEADINGS:
        return "skip"
    if any(skip_heading in heading for skip_heading in SKIP_SECTION_HEADINGS):
        return "skip"
    if any(keep_heading in heading for keep_heading in KEEP_SECTION_HEADINGS):
        return "keep"
    return ""


def _is_standalone_heading(line: str) -> bool:
    """Return whether a line looks like a standalone section heading."""
    normalized = _normalize_key(line)
    return len(line) <= 80 and normalized in KEEP_SECTION_HEADINGS.union(SKIP_SECTION_HEADINGS)


def _is_boilerplate_line(line: str) -> bool:
    """Return whether a line is boilerplate or company marketing text."""
    normalized = _normalize_key(line)
    if normalized in SKIP_SECTION_HEADINGS:
        return True
    return any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)


def _is_requirement_like(line: str) -> bool:
    """Return whether a line carries a technical requirement signal."""
    if any(pattern.search(line) for pattern in REQUIREMENT_SIGNAL_PATTERNS):
        return True
    if any(pattern.search(line) for pattern in _technical_signal_patterns()):
        return True
    return bool(_extract_experience_matches(line))


def _append_unique_line(lines: list[str], seen: set[str], line: str) -> None:
    """Append a line once using a normalized de-duplication key."""
    clean_line = _normalize_whitespace(line)
    key = _normalize_key(clean_line)
    if not key or key in seen:
        return
    seen.add(key)
    lines.append(clean_line)


def _safe_float(value: str | None) -> float | None:
    """Parse a finite non-negative year value."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _normalize_year_number(value: float | None) -> float | None:
    """Return years as an int-like float when possible."""
    if value is None:
        return None
    return float(int(value)) if value.is_integer() else value


def _strip_terminal_punctuation(line: str) -> str:
    """Strip punctuation that would duplicate sentence separators."""
    return line.strip().rstrip(".;")


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace after removing control characters."""
    return WHITESPACE_RE.sub(" ", CONTROL_RE.sub(" ", text)).strip()


def _normalize_key(text: str) -> str:
    """Normalize text for stable matching and de-duplication."""
    return NON_ALNUM_RE.sub(" ", text.casefold()).strip()


__all__ = [
    "JDComplexity",
    "ParsedJD",
    "extract_requirements_text",
    "parse_jd",
    "score_jd_complexity",
]
