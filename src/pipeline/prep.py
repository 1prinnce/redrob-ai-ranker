"""Offline preprocessing pipeline for candidate ranking artifacts.

This script validates candidate profiles, computes static ranking features,
generates candidate embeddings, builds a FAISS index, and writes the metadata
needed by the online ranking pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import argparse
import json
from numbers import Real
import pathlib
import pickle
import sys
import time
from typing import Any, Final, TypedDict

import numpy as np
import numpy.typing as npt


if __package__ in {None, ""}:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.data.loader import stream_candidates, validate_candidate_id
from src.data.validator import validate_schema
from src.embeddings import encoder
from src.embeddings import index as faiss_index
from src.embeddings.text_fusion import candidate_text
from src.features import behavioral, career, disqualifier, honeypot, location, skills
from src.utils import config
from src.utils.logger import get_logger, tqdm_loader


LOGGER = get_logger(__name__)

DEFAULT_MODEL_NAME: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE: Final[int] = encoder.DEFAULT_BATCH_SIZE

EMBEDDINGS_FILENAME: Final[str] = "candidate_embeddings.npy"
FEATURES_FILENAME: Final[str] = "candidate_features.npz"
METADATA_FILENAME: Final[str] = "candidate_meta.pkl"
FAISS_INDEX_FILENAME: Final[str] = "faiss.index"
MANIFEST_FILENAME: Final[str] = "prep_manifest.json"

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "career_score",
    "behavioral_score",
    "location_score",
    "skill_score",
    "honeypot_penalty",
    "disqualifier_penalty",
)
METADATA_FIELDS: Final[tuple[str, ...]] = (
    "candidate_id",
    "current_title",
    "current_company",
    "years_of_experience",
    "notice_period_days",
    "recruiter_response_rate",
)
YEARS_EXPERIENCE_KEYS: Final[tuple[str, ...]] = (
    "years_of_experience",
    "total_experience_years",
    "experience_years",
    "yoe",
)
TITLE_KEYS: Final[tuple[str, ...]] = ("current_title", "title", "job_title", "role", "position")
COMPANY_KEYS: Final[tuple[str, ...]] = (
    "current_company",
    "company",
    "company_name",
    "employer",
    "organization",
)
CURRENT_JOB_FLAGS: Final[tuple[str, ...]] = ("is_current", "current")
CURRENT_JOB_END_KEYS: Final[tuple[str, ...]] = ("end_date", "ended_at", "to", "until")
CURRENT_JOB_END_SENTINELS: Final[frozenset[str]] = frozenset({"present", "current", "now"})
MISSING_SENTINELS: Final[frozenset[str]] = frozenset({"", "-1", "none", "null", "nan"})


class PrepManifest(TypedDict):
    """Manifest written after a successful offline preprocessing run."""

    timestamp: str
    candidate_count: int
    embedding_dimension: int
    model_name: str
    embedding_backend: str
    prep_duration_seconds: float
    stage_durations_seconds: dict[str, float]
    artifacts: dict[str, str]
    feature_names: list[str]
    metadata_fields: list[str]


class CandidateMetaCache(TypedDict):
    """Columnar candidate metadata cache aligned to embedding row order."""

    candidate_id: list[str]
    current_title: list[str | None]
    current_company: list[str | None]
    years_of_experience: list[float | None]
    notice_period_days: list[float | None]
    recruiter_response_rate: list[float | None]


@dataclass(frozen=True)
class PrepArtifacts:
    """Resolved output artifact paths for the preprocessing run."""

    embeddings: pathlib.Path
    features: pathlib.Path
    metadata: pathlib.Path
    faiss_index: pathlib.Path
    manifest: pathlib.Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline preprocessing pipeline from command-line arguments.

    Args:
        argv: Optional argument vector used by tests. ``None`` reads from
            ``sys.argv`` through ``argparse``.

    Returns:
        Process-style exit code: ``0`` on success, ``1`` on failure.
    """
    args = _parse_args(argv)
    candidates_path = args.candidates.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    artifacts = _artifact_paths(out_dir)
    stage_durations: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    try:
        _validate_cli_inputs(candidates_path, args.batch_size)
        out_dir.mkdir(parents=True, exist_ok=True)

        with _stage_timer("load_validate", stage_durations):
            candidates = _load_valid_candidates(candidates_path)

        with _stage_timer("feature_computation", stage_durations):
            feature_arrays = _compute_feature_arrays(candidates)
            _save_feature_arrays(feature_arrays, artifacts.features)

        with _stage_timer("text_fusion", stage_durations):
            candidate_texts = _generate_candidate_texts(candidates)
            metadata_cache = _build_metadata_cache(candidates)

        with _stage_timer("embedding_generation", stage_durations):
            model = encoder.load_model(args.model_name)
            embeddings = encoder.encode_texts(candidate_texts, model=model, batch_size=args.batch_size)
            encoder.save_embeddings(embeddings, artifacts.embeddings)

        with _stage_timer("faiss_build", stage_durations):
            retrieval_index = faiss_index.build_index(embeddings)
            faiss_index.save_index(retrieval_index, artifacts.faiss_index)

        with _stage_timer("metadata_cache", stage_durations):
            _save_pickle(metadata_cache, artifacts.metadata)

        with _stage_timer("manifest", stage_durations):
            manifest = _build_manifest(
                artifacts=artifacts,
                candidate_count=len(candidate_texts),
                embeddings=embeddings,
                model=model,
                stage_durations=stage_durations,
                prep_duration_seconds=time.perf_counter() - pipeline_start,
            )
            _save_json(manifest, artifacts.manifest)

        LOGGER.info(
            "Offline preprocessing complete",
            extra={
                "candidate_count": len(candidate_texts),
                "embedding_shape": tuple(int(value) for value in embeddings.shape),
                "out_dir": str(out_dir),
            },
        )
        return 0
    except Exception as exc:
        LOGGER.exception("Offline preprocessing failed: %s", exc)
        return 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse preprocessing CLI arguments."""
    parser = argparse.ArgumentParser(description="Build offline ranking artifacts for candidate retrieval.")
    parser.add_argument(
        "--candidates",
        required=True,
        type=pathlib.Path,
        help="Path to candidates.jsonl or candidates.jsonl.gz.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=pathlib.Path,
        help="Directory where artifacts will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model name or path. Default: {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int,
        help=f"Embedding batch size. Default: {DEFAULT_BATCH_SIZE}.",
    )
    return parser.parse_args(argv)


def _validate_cli_inputs(candidates_path: pathlib.Path, batch_size: int) -> None:
    """Validate CLI inputs before running expensive stages."""
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidate input file does not exist: {candidates_path}")
    if not candidates_path.is_file():
        raise ValueError(f"Candidate input path is not a file: {candidates_path}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than zero, got {batch_size}.")


def _artifact_paths(out_dir: pathlib.Path) -> PrepArtifacts:
    """Return all artifact paths for an output directory."""
    return PrepArtifacts(
        embeddings=out_dir / EMBEDDINGS_FILENAME,
        features=out_dir / FEATURES_FILENAME,
        metadata=out_dir / METADATA_FILENAME,
        faiss_index=out_dir / FAISS_INDEX_FILENAME,
        manifest=out_dir / MANIFEST_FILENAME,
    )


@contextmanager
def _stage_timer(stage_name: str, stage_durations: dict[str, float]) -> Iterator[None]:
    """Log and record elapsed wall time for one pipeline stage."""
    LOGGER.info("Starting stage: %s", stage_name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        stage_durations[stage_name] = elapsed
        LOGGER.exception("Stage failed: %s", stage_name, extra={"duration_seconds": round(elapsed, 3)})
        raise
    else:
        elapsed = time.perf_counter() - start
        stage_durations[stage_name] = elapsed
        LOGGER.info("Finished stage: %s", stage_name, extra={"duration_seconds": round(elapsed, 3)})


def _load_valid_candidates(candidates_path: pathlib.Path) -> list[dict[str, Any]]:
    """Stream, validate, and return candidate records in input order."""
    valid_candidates: list[dict[str, Any]] = []
    invalid_count = 0
    total_count = 0
    seen_candidate_ids: set[str] = set()

    for total_count, candidate in enumerate(stream_candidates(candidates_path), start=1):
        is_valid_schema, error_message = validate_schema(candidate)
        if not is_valid_schema:
            invalid_count += 1
            LOGGER.error("Invalid candidate at row %d: %s", total_count, error_message)
            continue

        candidate_id = candidate.get("candidate_id")
        if not validate_candidate_id(candidate_id):
            invalid_count += 1
            LOGGER.error("Invalid candidate_id at row %d: %r", total_count, candidate_id)
            continue
        if candidate_id in seen_candidate_ids:
            invalid_count += 1
            LOGGER.error("Duplicate candidate_id at row %d: %r", total_count, candidate_id)
            continue

        seen_candidate_ids.add(candidate_id)
        valid_candidates.append(candidate)

    if not valid_candidates:
        raise ValueError(f"No valid candidate records found in {candidates_path}.")

    LOGGER.info(
        "Candidate load and validation complete",
        extra={
            "input_rows": total_count,
            "valid_candidates": len(valid_candidates),
            "invalid_candidates": invalid_count,
        },
    )
    return valid_candidates


def _compute_feature_arrays(candidates: Sequence[Mapping[str, Any]]) -> dict[str, npt.NDArray[np.float32]]:
    """Compute static candidate feature arrays aligned to candidate order."""
    count = len(candidates)
    arrays = {name: np.empty(count, dtype=np.float32) for name in FEATURE_NAMES}

    for index, candidate in enumerate(tqdm_loader(candidates, total=count, desc="Computing features")):
        candidate_id = _candidate_id(candidate, fallback_index=index)
        profile = _as_mapping(candidate.get("profile"))
        redrob_signals = _as_mapping(candidate.get("redrob_signals"))

        arrays["career_score"][index] = _safe_feature(
            "career_score",
            candidate_id,
            lambda: career.combined_career_score(candidate),
            default=0.0,
        )
        arrays["behavioral_score"][index] = _safe_feature(
            "behavioral_score",
            candidate_id,
            lambda: behavioral.behavioral_score(dict(redrob_signals)),
            default=0.0,
        )
        arrays["location_score"][index] = _safe_feature(
            "location_score",
            candidate_id,
            lambda: location.location_score(profile, redrob_signals),
            default=0.0,
        )
        arrays["skill_score"][index] = _safe_feature(
            "skill_score",
            candidate_id,
            lambda: skills.skill_score(candidate.get("skills")),
            default=0.0,
        )
        arrays["honeypot_penalty"][index] = _safe_feature(
            "honeypot_penalty",
            candidate_id,
            lambda: honeypot.honeypot_score(candidate),
            default=float(config.HONEYPOT_PENALTY),
        )
        arrays["disqualifier_penalty"][index] = _safe_feature(
            "disqualifier_penalty",
            candidate_id,
            lambda: disqualifier.disqualifier_penalty(candidate),
            default=0.0,
        )

    return arrays


def _safe_feature(
    feature_name: str,
    candidate_id: str,
    scorer: Callable[[], float],
    default: float,
) -> np.float32:
    """Compute one bounded feature value without breaking the whole prep run."""
    try:
        value = float(scorer())
    except Exception as exc:
        LOGGER.exception(
            "Feature computation failed",
            extra={"feature_name": feature_name, "candidate_id": candidate_id, "error": str(exc)},
        )
        value = default

    if not np.isfinite(value):
        LOGGER.warning(
            "Feature computation returned non-finite value",
            extra={"feature_name": feature_name, "candidate_id": candidate_id, "value": value},
        )
        value = default

    return np.float32(_clamp(value))


def _save_feature_arrays(features: Mapping[str, npt.NDArray[np.float32]], path: pathlib.Path) -> None:
    """Persist feature arrays to an uncompressed NPZ for fast online loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing feature arrays: {', '.join(missing)}")

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez(
            handle,
            **{name: _validate_feature_array(features[name], name) for name in FEATURE_NAMES},
            feature_names=np.asarray(FEATURE_NAMES),
        )
    temp_path.replace(path)
    LOGGER.info("Saved candidate features", extra={"path": str(path), "feature_count": len(FEATURE_NAMES)})


def _validate_feature_array(values: npt.NDArray[np.float32], name: str) -> npt.NDArray[np.float32]:
    """Validate one feature array before persistence."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN values.")
    if np.isinf(array).any():
        raise ValueError(f"{name} contains infinite values.")
    return array


def _generate_candidate_texts(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    """Generate deterministic embedding text for every valid candidate."""
    count = len(candidates)
    texts: list[str] = []
    texts_append = texts.append
    for candidate in tqdm_loader(candidates, total=count, desc="Generating candidate text"):
        texts_append(candidate_text(candidate))

    LOGGER.info("Generated candidate texts", extra={"candidate_count": len(texts)})
    return texts


def _build_metadata_cache(candidates: Sequence[Mapping[str, Any]]) -> CandidateMetaCache:
    """Build a compact columnar metadata cache aligned with embedding rows."""
    cache: CandidateMetaCache = {
        "candidate_id": [],
        "current_title": [],
        "current_company": [],
        "years_of_experience": [],
        "notice_period_days": [],
        "recruiter_response_rate": [],
    }

    for index, candidate in enumerate(tqdm_loader(candidates, total=len(candidates), desc="Caching metadata")):
        profile = _as_mapping(candidate.get("profile"))
        signals = _as_mapping(candidate.get("redrob_signals"))
        current_job = _current_job(candidate)

        cache["candidate_id"].append(_candidate_id(candidate, fallback_index=index))
        cache["current_title"].append(
            _first_text(candidate, profile, current_job, keys=TITLE_KEYS)
        )
        cache["current_company"].append(
            _first_text(candidate, profile, current_job, keys=COMPANY_KEYS)
        )
        cache["years_of_experience"].append(
            _first_number(candidate, profile, keys=YEARS_EXPERIENCE_KEYS)
        )
        cache["notice_period_days"].append(
            _non_negative_number(_first_value(signals, candidate, profile, keys=("notice_period_days",)))
        )
        cache["recruiter_response_rate"].append(
            _response_rate(_first_value(signals, candidate, profile, keys=("recruiter_response_rate",)))
        )

    _validate_metadata_cache(cache)
    LOGGER.info("Built candidate metadata cache", extra={"candidate_count": len(cache["candidate_id"])})
    return cache


def _validate_metadata_cache(cache: CandidateMetaCache) -> None:
    """Validate metadata cache field alignment and duplicate IDs."""
    lengths = {field: len(cache[field]) for field in METADATA_FIELDS}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Metadata fields have inconsistent lengths: {lengths}")

    candidate_ids = cache["candidate_id"]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Metadata cache contains duplicate candidate IDs.")


def _save_pickle(value: Any, path: pathlib.Path) -> None:
    """Persist a pickle artifact through a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(path)
    LOGGER.info("Saved pickle artifact", extra={"path": str(path), "bytes": path.stat().st_size})


def _save_json(value: Mapping[str, Any], path: pathlib.Path) -> None:
    """Persist a JSON artifact through a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)
    LOGGER.info("Saved JSON artifact", extra={"path": str(path), "bytes": path.stat().st_size})


def _build_manifest(
    *,
    artifacts: PrepArtifacts,
    candidate_count: int,
    embeddings: np.ndarray,
    model: encoder.EmbeddingBackend,
    stage_durations: Mapping[str, float],
    prep_duration_seconds: float,
) -> PrepManifest:
    """Build the final preprocessing manifest."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be two-dimensional, got shape {embeddings.shape}.")

    manifest: PrepManifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "candidate_count": int(candidate_count),
        "embedding_dimension": int(embeddings.shape[1]),
        "model_name": str(getattr(model, "model_name", DEFAULT_MODEL_NAME)),
        "embedding_backend": str(getattr(model, "backend", "unknown")),
        "prep_duration_seconds": round(float(prep_duration_seconds), 3),
        "stage_durations_seconds": {
            stage_name: round(float(duration), 3) for stage_name, duration in stage_durations.items()
        },
        "artifacts": {
            "candidate_embeddings": str(artifacts.embeddings),
            "candidate_features": str(artifacts.features),
            "candidate_meta": str(artifacts.metadata),
            "faiss_index": str(artifacts.faiss_index),
        },
        "feature_names": list(FEATURE_NAMES),
        "metadata_fields": list(METADATA_FIELDS),
    }
    return manifest


def _candidate_id(candidate: Mapping[str, Any], fallback_index: int) -> str:
    """Return a candidate ID for logging and metadata."""
    value = candidate.get("candidate_id")
    return value if isinstance(value, str) and value else f"ROW_{fallback_index:07d}"


def _current_job(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first career-history record explicitly marked current."""
    history = candidate.get("career_history")
    if not isinstance(history, list):
        return {}

    for job in history:
        if isinstance(job, Mapping) and _is_current_job(job):
            return job
    return {}


def _is_current_job(job: Mapping[str, Any]) -> bool:
    """Return whether a career-history record appears to be current."""
    for key in CURRENT_JOB_FLAGS:
        if _truthy(job.get(key)):
            return True
    for key in CURRENT_JOB_END_KEYS:
        value = job.get(key)
        if isinstance(value, str) and value.strip().casefold() in CURRENT_JOB_END_SENTINELS:
            return True
    return False


def _first_text(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty text value from the supplied mappings."""
    value = _first_value(*mappings, keys=keys)
    if _is_missing(value):
        return None
    if isinstance(value, str):
        clean_value = " ".join(value.split())
        return clean_value or None
    if isinstance(value, Real) and not isinstance(value, bool):
        return str(value)
    return None


def _first_number(*mappings: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first non-negative numeric value from supplied mappings."""
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
    """Parse a recruiter response rate as a normalized 0-1 value."""
    rate = _non_negative_number(value)
    if rate is None:
        return None
    if rate > 1.0:
        rate /= 100.0
    return min(rate, 1.0)


def _non_negative_number(value: object) -> float | None:
    """Parse a non-negative finite numeric value from a scalar."""
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, Real):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) and numeric_value >= 0.0 else None
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"immediate", "now", "none", "no notice"}:
            return 0.0
        digits = "".join(character for character in lowered if character.isdigit() or character == ".")
        if not digits:
            return None
        try:
            numeric_value = float(digits)
        except ValueError:
            return None
        return numeric_value if np.isfinite(numeric_value) and numeric_value >= 0.0 else None
    return None


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return mapping values unchanged and malformed values as an empty mapping."""
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
    """Return whether a value is null, blank, or a known missing sentinel."""
    return value is None or value == -1 or (
        isinstance(value, str) and value.strip().casefold() in MISSING_SENTINELS
    )


def _clamp(value: float) -> float:
    """Clamp feature and penalty values to the inclusive 0-1 range."""
    return max(0.0, min(1.0, value))


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
