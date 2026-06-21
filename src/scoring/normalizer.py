"""Score normalization and ranking integrity helpers.

This module is intentionally small and deterministic: it validates candidate
score arrays, normalizes raw ranking signals into the submission score band,
and selects a stable top-N list suitable for final output generation.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from numbers import Integral, Real
from typing import Final

import numpy as np
import numpy.typing as npt


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_MIN: Final[float] = 0.40
DEFAULT_OUTPUT_MAX: Final[float] = 1.00
DEFAULT_TOP_N: Final[int] = 100
IDENTICAL_SCORE_FILL_FRACTION: Final[float] = 0.50
MIN_VALID_N: Final[int] = 1


def normalize_scores(
    scores: np.ndarray,
    output_min: float = DEFAULT_OUTPUT_MIN,
    output_max: float = DEFAULT_OUTPUT_MAX,
) -> npt.NDArray[np.float32]:
    """Normalize raw ranking scores into a bounded float32 score range.

    Scores are min-max normalized into ``[output_min, output_max]``. The
    transform is monotonic, so higher raw scores never become lower normalized
    scores. If every score is identical, all scores are assigned the midpoint
    of the output range to avoid divide-by-zero while preserving equality.

    Args:
        scores: One-dimensional numpy array of finite numeric scores.
        output_min: Inclusive lower bound for normalized scores.
        output_max: Inclusive upper bound for normalized scores.

    Returns:
        A one-dimensional ``np.float32`` array clipped to the requested bounds.

    Raises:
        TypeError: If inputs have unsupported types.
        ValueError: If scores are empty, non-finite, non-numeric, not
            one-dimensional, or if output bounds are invalid.
    """
    score_values = _validate_score_array(scores)
    lower, upper = _validate_output_bounds(output_min, output_max)

    score_min = float(np.min(score_values))
    score_max = float(np.max(score_values))
    output_range = upper - lower

    if score_min == score_max:
        fill_value = lower + (output_range * IDENTICAL_SCORE_FILL_FRACTION)
        normalized = np.full(score_values.shape, fill_value, dtype=np.float32)
        LOGGER.debug(
            "Normalized identical scores to midpoint",
            extra={"count": int(score_values.size), "value": float(fill_value)},
        )
        return normalized

    # Scale by the largest absolute endpoint before min-max normalization. This
    # keeps extreme-but-finite ranges from overflowing during subtraction.
    scale_base = max(abs(score_min), abs(score_max), 1.0)
    scaled_scores = score_values / scale_base
    scaled_min = score_min / scale_base
    scaled_max = score_max / scale_base
    scaled_range = scaled_max - scaled_min

    if scaled_range == 0.0 or not np.isfinite(scaled_range):
        raise ValueError(
            "scores cannot be normalized because the finite score range "
            f"collapsed numerically: min={score_min!r}, max={score_max!r}."
        )

    normalized64 = ((scaled_scores - scaled_min) / scaled_range) * output_range + lower
    np.clip(normalized64, lower, upper, out=normalized64)

    normalized = normalized64.astype(np.float32, copy=False)
    np.clip(normalized, np.float32(lower), np.float32(upper), out=normalized)

    LOGGER.debug(
        "Normalized scores",
        extra={
            "count": int(score_values.size),
            "input_min": score_min,
            "input_max": score_max,
            "output_min": lower,
            "output_max": upper,
        },
    )
    return normalized


def validate_monotonic(scores: np.ndarray) -> None:
    """Validate that scores are sorted in non-increasing order.

    Args:
        scores: One-dimensional numpy array of finite numeric scores.

    Raises:
        TypeError: If ``scores`` is not a supported numpy numeric array.
        ValueError: If the array is empty, contains non-finite values, or any
            ``scores[i] < scores[i + 1]`` violation is found.
    """
    score_values = _validate_score_array(scores)
    if score_values.size < 2:
        return

    violations = np.flatnonzero(score_values[:-1] < score_values[1:])
    if violations.size:
        index = int(violations[0])
        current_value = float(score_values[index])
        next_value = float(score_values[index + 1])
        raise ValueError(
            "scores must be monotonically non-increasing; "
            f"failure at index {index}: scores[{index}]={current_value!r} "
            f"< scores[{index + 1}]={next_value!r}."
        )

    LOGGER.debug("Validated monotonic score ordering", extra={"count": int(score_values.size)})


def select_top_n(
    scores: np.ndarray,
    candidate_ids: Sequence[str],
    n: int = DEFAULT_TOP_N,
) -> list[tuple[str, float]]:
    """Return the top-N candidates sorted by score descending.

    Ties are stable: candidates with equal scores retain their original input
    order. This gives deterministic output without requiring a secondary
    business-specific tie-breaker.

    Args:
        scores: One-dimensional numpy array of finite numeric scores.
        candidate_ids: Candidate IDs aligned one-to-one with ``scores``.
        n: Maximum number of candidates to return. If ``n`` exceeds the number
            of candidates, all candidates are returned.

    Returns:
        A list of ``(candidate_id, score)`` tuples sorted descending by score.

    Raises:
        TypeError: If inputs have unsupported types.
        ValueError: If inputs are empty, lengths differ, IDs are duplicated,
            IDs are not strings, scores are non-finite, or ``n`` is invalid.
    """
    top_n = _validate_top_n(n)
    ids, score_values = _validate_ranking_inputs(candidate_ids, scores)

    selected_count = min(top_n, score_values.size)
    sorted_indices = np.argsort(-score_values, kind="stable")[:selected_count]
    top_candidates = [(ids[int(index)], float(score_values[int(index)])) for index in sorted_indices]

    LOGGER.debug(
        "Selected top candidates",
        extra={"requested_n": top_n, "returned_n": len(top_candidates), "pool_size": len(ids)},
    )
    return top_candidates


def validate_ranking(candidate_ids: Sequence[str], scores: np.ndarray) -> None:
    """Validate candidate IDs and score arrays before ranking submission.

    Checks include non-empty inputs, equal ID/score lengths, string candidate
    IDs, duplicate IDs, one-dimensional numeric scores, and finite score values
    with no NaN or infinity.

    Args:
        candidate_ids: Candidate IDs aligned one-to-one with ``scores``.
        scores: One-dimensional numpy array of finite numeric scores.

    Raises:
        TypeError: If inputs have unsupported types.
        ValueError: If any ranking integrity check fails.
    """
    ids, _ = _validate_ranking_inputs(candidate_ids, scores)
    LOGGER.debug("Validated ranking inputs", extra={"count": len(ids)})


def rank_statistics(scores: np.ndarray) -> dict[str, int | float]:
    """Return summary statistics for a finite numeric score array.

    Args:
        scores: One-dimensional numpy array of finite numeric scores.

    Returns:
        Dictionary containing ``count``, ``min``, ``max``, ``mean``, and
        population ``std`` as Python scalar values.

    Raises:
        TypeError: If ``scores`` is not a supported numpy numeric array.
        ValueError: If scores are empty, non-finite, or not one-dimensional.
    """
    score_values = _validate_score_array(scores)
    stats: dict[str, int | float] = {
        "count": int(score_values.size),
        "min": float(np.min(score_values)),
        "max": float(np.max(score_values)),
        "mean": float(np.mean(score_values, dtype=np.float64)),
        "std": float(np.std(score_values, dtype=np.float64)),
    }

    LOGGER.debug("Computed rank statistics", extra=stats)
    return stats


def _validate_score_array(scores: np.ndarray, name: str = "scores") -> npt.NDArray[np.float64]:
    """Validate and return scores as a float64 working array."""
    if not isinstance(scores, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(scores).__name__}.")
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {scores.shape}.")
    if scores.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.issubdtype(scores.dtype, np.bool_) or not np.issubdtype(scores.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values, got dtype {scores.dtype}.")

    score_values = np.asarray(scores, dtype=np.float64)

    nan_indices = np.flatnonzero(np.isnan(score_values))
    if nan_indices.size:
        index = int(nan_indices[0])
        raise ValueError(f"{name} contains NaN at index {index}.")

    inf_indices = np.flatnonzero(np.isinf(score_values))
    if inf_indices.size:
        index = int(inf_indices[0])
        raise ValueError(f"{name} contains infinity at index {index}: {scores[index]!r}.")

    return score_values


def _validate_output_bounds(output_min: float, output_max: float) -> tuple[float, float]:
    """Validate normalization output bounds."""
    lower = _validate_real_number(output_min, "output_min")
    upper = _validate_real_number(output_max, "output_max")
    if lower >= upper:
        raise ValueError(f"output_min must be less than output_max, got {lower!r} >= {upper!r}.")
    return lower, upper


def _validate_real_number(value: float, name: str) -> float:
    """Validate a finite real-valued scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, got {type(value).__name__}.")

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite, got {numeric_value!r}.")
    return numeric_value


def _validate_candidate_ids(candidate_ids: Sequence[str]) -> list[str]:
    """Validate candidate IDs and return a list preserving input order."""
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence of strings, not a scalar string.")

    if isinstance(candidate_ids, np.ndarray):
        if candidate_ids.ndim != 1:
            raise ValueError(f"candidate_ids must be one-dimensional, got shape {candidate_ids.shape}.")
        ids = candidate_ids.tolist()
    elif isinstance(candidate_ids, Sequence):
        ids = list(candidate_ids)
    else:
        raise TypeError(f"candidate_ids must be a sequence of strings, got {type(candidate_ids).__name__}.")

    if not ids:
        raise ValueError("candidate_ids must not be empty.")

    first_seen_index: dict[str, int] = {}
    for index, candidate_id in enumerate(ids):
        if not isinstance(candidate_id, str):
            raise TypeError(
                "candidate_ids must contain only strings; "
                f"candidate_ids[{index}] has type {type(candidate_id).__name__}."
            )
        if candidate_id in first_seen_index:
            raise ValueError(
                "candidate_ids must not contain duplicates; "
                f"{candidate_id!r} appears at indices {first_seen_index[candidate_id]} and {index}."
            )
        first_seen_index[candidate_id] = index

    return ids


def _validate_ranking_inputs(
    candidate_ids: Sequence[str],
    scores: np.ndarray,
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Validate aligned candidate IDs and scores."""
    ids = _validate_candidate_ids(candidate_ids)
    score_values = _validate_score_array(scores)

    if len(ids) != score_values.size:
        raise ValueError(
            "candidate_ids and scores must have equal lengths; "
            f"got {len(ids)} candidate IDs and {score_values.size} scores."
        )

    return ids, score_values


def _validate_top_n(n: int) -> int:
    """Validate top-N selection size."""
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError(f"n must be an integer, got {type(n).__name__}.")
    top_n = int(n)
    if top_n < MIN_VALID_N:
        raise ValueError(f"n must be at least {MIN_VALID_N}, got {top_n}.")
    return top_n


__all__ = [
    "DEFAULT_OUTPUT_MAX",
    "DEFAULT_OUTPUT_MIN",
    "DEFAULT_TOP_N",
    "normalize_scores",
    "rank_statistics",
    "select_top_n",
    "validate_monotonic",
    "validate_ranking",
]
