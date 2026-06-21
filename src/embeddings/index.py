"""FAISS-backed semantic retrieval for candidate embedding search."""

import logging
import pathlib
from typing import Any, Tuple

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR: ImportError | None = exc
else:
    _FAISS_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
EMBEDDING_DIMENSION: int = 384
EPSILON: float = 1e-6


def validate_embedding_matrix(embeddings: np.ndarray) -> None:
    """Validate that embeddings are a finite float32 matrix of shape ``(N, 384)``."""
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"embeddings must be a numpy.ndarray, got {type(embeddings).__name__}.")
    if embeddings.dtype != np.float32:
        raise TypeError(f"embeddings must have dtype float32, got {embeddings.dtype}.")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D with shape (N, 384), got {embeddings.shape}.")
    if embeddings.shape[0] <= 0:
        raise ValueError("embeddings must contain at least one vector.")
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"embeddings must have dimension {EMBEDDING_DIMENSION}, got {embeddings.shape[1]}."
        )
    if np.isnan(embeddings).any():
        raise ValueError("embeddings contain NaN values.")
    if np.isinf(embeddings).any():
        raise ValueError("embeddings contain infinite values.")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return a copy-safe L2-normalized float32 embedding matrix for cosine search."""
    _require_faiss()
    validate_embedding_matrix(embeddings)

    normalized = np.ascontiguousarray(embeddings.copy(), dtype=np.float32)
    faiss.normalize_L2(normalized)
    norms = np.linalg.norm(normalized, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        bad_count = int(np.count_nonzero(np.abs(norms - 1.0) > 1e-4))
        raise ValueError(f"normalized embeddings failed unit-norm validation for {bad_count} rows.")
    return normalized


def build_index(embeddings: np.ndarray) -> Any:
    """Build and return a FAISS ``IndexFlatIP`` over normalized candidate embeddings."""
    _require_faiss()
    validate_embedding_matrix(embeddings)
    normalized = normalize_embeddings(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(normalized)
    if index.ntotal != normalized.shape[0]:
        raise RuntimeError(f"FAISS index size mismatch: ntotal={index.ntotal}, expected={normalized.shape[0]}.")

    LOGGER.info(
        "Built FAISS IndexFlatIP",
        extra={
            "ntotal": int(index.ntotal),
            "dimension": EMBEDDING_DIMENSION,
            "memory_mb": round(normalized.nbytes / (1024 * 1024), 2),
        },
    )
    return index


def save_index(index: Any, filepath: str | pathlib.Path) -> None:
    """Persist a FAISS index to disk and verify the output file was written."""
    _require_faiss()
    _validate_index(index)
    path = pathlib.Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        faiss.write_index(index, str(path))
    except Exception as exc:
        raise OSError(f"Failed to save FAISS index to {path}: {exc}") from exc

    if not path.exists() or path.stat().st_size <= 0:
        raise OSError(f"FAISS index write validation failed for {path}.")
    LOGGER.info("Saved FAISS index", extra={"path": str(path), "bytes": path.stat().st_size})


def load_index(filepath: str | pathlib.Path) -> Any:
    """Load a FAISS index from disk and validate basic metadata."""
    _require_faiss()
    path = pathlib.Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"FAISS index path is not a file: {path}")

    try:
        index = faiss.read_index(str(path))
    except Exception as exc:
        raise OSError(f"Failed to load FAISS index from {path}: {exc}") from exc

    _validate_index(index)
    LOGGER.info(
        "Loaded FAISS index",
        extra={"path": str(path), "ntotal": int(index.ntotal), "dimension": int(index.d)},
    )
    return index


def validate_query_vector(query: np.ndarray) -> np.ndarray:
    """Validate and return a normalized query vector with shape ``(1, 384)``."""
    _require_faiss()
    if not isinstance(query, np.ndarray):
        raise TypeError(f"query must be a numpy.ndarray, got {type(query).__name__}.")
    if query.dtype != np.float32:
        raise TypeError(f"query must have dtype float32, got {query.dtype}.")
    if query.ndim == 1:
        if query.shape[0] != EMBEDDING_DIMENSION:
            raise ValueError(f"query dimension must be {EMBEDDING_DIMENSION}, got {query.shape[0]}.")
        clean_query = query.reshape(1, EMBEDDING_DIMENSION)
    elif query.ndim == 2 and query.shape == (1, EMBEDDING_DIMENSION):
        clean_query = query
    else:
        raise ValueError(f"query must have shape (384,) or (1, 384), got {query.shape}.")
    if np.isnan(clean_query).any():
        raise ValueError("query contains NaN values.")
    if np.isinf(clean_query).any():
        raise ValueError("query contains infinite values.")

    normalized = np.ascontiguousarray(clean_query.copy(), dtype=np.float32)
    faiss.normalize_L2(normalized)
    norm = float(np.linalg.norm(normalized))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError(f"normalized query failed unit-norm validation: norm={norm:.6f}.")
    return normalized


def query_index(index: Any, query_embedding: np.ndarray, k: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index for the top-k matches to one query embedding."""
    _require_faiss()
    _validate_index(index)
    query = validate_query_vector(query_embedding)
    search_k = _validated_k(k, index.ntotal)

    start_ms = _now_ms()
    distances, indices = index.search(query, search_k)
    latency_ms = _now_ms() - start_ms
    LOGGER.info("FAISS query complete", extra={"k": search_k, "latency_ms": round(latency_ms, 3)})
    return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def batch_query_index(index: Any, query_embeddings: np.ndarray, k: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index for the top-k matches to multiple query embeddings."""
    _require_faiss()
    _validate_index(index)
    queries = _validate_query_matrix(query_embeddings)
    search_k = _validated_k(k, index.ntotal)

    start_ms = _now_ms()
    distances, indices = index.search(queries, search_k)
    latency_ms = _now_ms() - start_ms
    LOGGER.info(
        "FAISS batch query complete",
        extra={"queries": int(queries.shape[0]), "k": search_k, "latency_ms": round(latency_ms, 3)},
    )
    return np.asarray(distances, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def _validate_query_matrix(queries: np.ndarray) -> np.ndarray:
    """Validate and normalize a batched query matrix with shape ``(Q, 384)``."""
    _require_faiss()
    if not isinstance(queries, np.ndarray):
        raise TypeError(f"query_embeddings must be a numpy.ndarray, got {type(queries).__name__}.")
    if queries.dtype != np.float32:
        raise TypeError(f"query_embeddings must have dtype float32, got {queries.dtype}.")
    if queries.ndim != 2 or queries.shape[0] <= 0 or queries.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"query_embeddings must have shape (Q, 384) with Q > 0, got {queries.shape}.")
    if np.isnan(queries).any():
        raise ValueError("query_embeddings contain NaN values.")
    if np.isinf(queries).any():
        raise ValueError("query_embeddings contain infinite values.")

    normalized = np.ascontiguousarray(queries.copy(), dtype=np.float32)
    faiss.normalize_L2(normalized)
    norms = np.linalg.norm(normalized, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        bad_count = int(np.count_nonzero(np.abs(norms - 1.0) > 1e-4))
        raise ValueError(f"normalized queries failed unit-norm validation for {bad_count} rows.")
    return normalized


def _validate_index(index: Any) -> None:
    """Validate the FAISS index is populated and compatible with 384-d embeddings."""
    _require_faiss()
    if not isinstance(index, faiss.Index):
        raise TypeError(f"index must be a faiss.Index, got {type(index).__name__}.")
    if int(index.d) != EMBEDDING_DIMENSION:
        raise ValueError(f"index dimension must be {EMBEDDING_DIMENSION}, got {int(index.d)}.")
    if int(index.ntotal) <= 0:
        raise ValueError("index must contain at least one vector.")


def _validated_k(k: int, index_size: int) -> int:
    """Return a valid k that never exceeds the FAISS index size."""
    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    return min(k, int(index_size))


def _require_faiss() -> None:
    """Raise a clear runtime error when FAISS is unavailable."""
    if faiss is None:
        raise RuntimeError(
            "faiss is required for semantic retrieval but is not installed."
        ) from _FAISS_IMPORT_ERROR


def _now_ms() -> float:
    """Return a monotonic timestamp in milliseconds using FAISS timing utilities."""
    _require_faiss()
    return float(faiss.getmillisecs())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    rng = np.random.default_rng(42)
    sample_embeddings = rng.normal(size=(10_000, EMBEDDING_DIMENSION)).astype(np.float32)
    sample_query = rng.normal(size=(EMBEDDING_DIMENSION,)).astype(np.float32)

    retrieval_index = build_index(sample_embeddings)
    top_distances, top_indices = query_index(retrieval_index, sample_query, k=10)
    batch_distances, batch_indices = batch_query_index(
        retrieval_index,
        rng.normal(size=(3, EMBEDDING_DIMENSION)).astype(np.float32),
        k=10,
    )

    print("FAISS self-test diagnostics")
    print(f"index_ntotal={retrieval_index.ntotal}")
    print(f"single_distances_shape={top_distances.shape}")
    print(f"single_indices_shape={top_indices.shape}")
    print(f"batch_distances_shape={batch_distances.shape}")
    print(f"batch_indices_shape={batch_indices.shape}")
