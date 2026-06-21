"""Final candidate ranking score composition."""

from collections.abc import Mapping
from typing import Any
import logging

import numpy as np

from src.features import behavioral, career, disqualifier, honeypot, location, skills
from src.utils import config


LOGGER = logging.getLogger(__name__)

JD_MATCH_WEIGHT: float = 0.35
CAREER_WEIGHT: float = 0.25
BEHAVIORAL_WEIGHT: float = 0.15
LOCATION_WEIGHT: float = 0.10
SKILLS_WEIGHT: float = 0.15
FALLBACK_SCORE: float = 0.10
EPSILON: float = 1e-12


def cosine_similarity_score(jd_embedding: np.ndarray, candidate_embedding: np.ndarray) -> float:
    """Return normalized dot-product similarity clamped to the [0, 1] score range."""
    jd_vector = _validate_vector(jd_embedding, "jd_embedding")
    candidate_vector = _validate_vector(candidate_embedding, "candidate_embedding")
    if jd_vector.shape != candidate_vector.shape:
        raise ValueError(
            f"Embedding dimensions must match, got {jd_vector.shape[0]} and {candidate_vector.shape[0]}."
        )

    jd_norm = float(np.linalg.norm(jd_vector))
    candidate_norm = float(np.linalg.norm(candidate_vector))
    if jd_norm <= EPSILON or candidate_norm <= EPSILON:
        return 0.0

    cosine = float(np.dot(jd_vector, candidate_vector) / (jd_norm * candidate_norm))
    return _clamp(cosine)


def score_breakdown(
    candidate: Mapping[str, Any],
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
) -> dict[str, float]:
    """Return all component scores, penalties, and final score for one candidate."""
    scores = _component_scores(candidate, jd_embedding, candidate_embedding)
    base_score = _base_score(scores)
    final_score = _apply_penalties(
        base_score,
        scores["honeypot_penalty"],
        scores["disqualifier_penalty"],
    )
    return {
        "jd_match": scores["jd_match"],
        "career": scores["career"],
        "behavioral": scores["behavioral"],
        "location": scores["location"],
        "skills": scores["skills"],
        "honeypot_penalty": scores["honeypot_penalty"],
        "disqualifier_penalty": scores["disqualifier_penalty"],
        "base_score": base_score,
        "final_score": final_score,
    }


def score_candidate(
    candidate: Mapping[str, Any],
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
) -> float:
    """Compute a bounded final ranking score for one candidate."""
    scores = _component_scores(candidate, jd_embedding, candidate_embedding)
    return _apply_penalties(
        _base_score(scores),
        scores["honeypot_penalty"],
        scores["disqualifier_penalty"],
    )


def safe_score(
    candidate: Mapping[str, Any],
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    idx: int,
) -> float:
    """Score one candidate without allowing errors to crash the ranking pipeline."""
    try:
        return score_candidate(candidate, jd_embedding, candidate_embedding)
    except Exception as exc:
        LOGGER.exception("Candidate scoring failed at index %d: %s", idx, exc)
        return FALLBACK_SCORE


def _component_scores(
    candidate: Mapping[str, Any],
    jd_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
) -> dict[str, float]:
    """Compute bounded feature scores and penalties for a candidate."""
    if not isinstance(candidate, Mapping):
        raise TypeError(f"candidate must be a mapping, got {type(candidate).__name__}.")

    profile = candidate.get("profile")
    redrob_signals = candidate.get("redrob_signals")
    candidate_skills = candidate.get("skills")

    return {
        "jd_match": cosine_similarity_score(jd_embedding, candidate_embedding),
        "career": _clamp(career.combined_career_score(candidate)),
        "behavioral": _clamp(behavioral.behavioral_score(redrob_signals)),
        "location": _clamp(location.location_score(profile, redrob_signals)),
        "skills": _clamp(skills.skill_score(candidate_skills)),
        "honeypot_penalty": _clamp_penalty(honeypot.honeypot_score(candidate), config.HONEYPOT_PENALTY),
        "disqualifier_penalty": _clamp_penalty(disqualifier.disqualifier_penalty(candidate), 1.0),
    }


def _base_score(scores: Mapping[str, float]) -> float:
    """Combine non-penalty ranking signals with named weights."""
    return _clamp(
        JD_MATCH_WEIGHT * scores["jd_match"]
        + CAREER_WEIGHT * scores["career"]
        + BEHAVIORAL_WEIGHT * scores["behavioral"]
        + LOCATION_WEIGHT * scores["location"]
        + SKILLS_WEIGHT * scores["skills"]
    )


def _apply_penalties(base_score: float, honeypot_penalty: float, disqualifier_penalty: float) -> float:
    """Apply multiplicative penalties and clamp the final score to [0, 1]."""
    return _clamp(base_score * (1.0 - honeypot_penalty) * (1.0 - disqualifier_penalty))


def _validate_vector(vector: np.ndarray, name: str) -> np.ndarray:
    """Validate an embedding vector and return it as a one-dimensional float array."""
    if not isinstance(vector, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(vector).__name__}.")
    if vector.ndim == 2 and 1 in vector.shape:
        vector = vector.reshape(-1)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}.")
    if vector.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(vector.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values, got dtype {vector.dtype}.")

    clean_vector = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(clean_vector).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return clean_vector


def _clamp_penalty(value: float, max_penalty: float) -> float:
    """Clamp a penalty to a configured non-negative maximum."""
    return max(0.0, min(float(max_penalty), float(value)))


def _clamp(value: float) -> float:
    """Clamp a score to the inclusive 0.0 to 1.0 range."""
    return max(0.0, min(1.0, float(value)))
