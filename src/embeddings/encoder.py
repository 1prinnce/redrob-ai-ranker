"""CPU-oriented embedding model loading, encoding, and persistence utilities."""

import pathlib
from typing import Any, Iterable, Protocol

import numpy as np

from src.utils.logger import get_logger, tqdm_loader


LOGGER = get_logger(__name__)
DEFAULT_BATCH_SIZE: int = 256
EPSILON: float = 1e-12


class EmbeddingBackend(Protocol):
    """Minimal interface implemented by supported embedding backends."""

    backend: str
    model_name: str
    dimension: int | None

    def encode(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Encode a batch of text strings into an embedding matrix."""
        ...


class SentenceTransformerBackend:
    """SentenceTransformers backend wrapper with a consistent encode interface."""

    backend = "sentence-transformers"

    def __init__(self, model: Any, model_name: str) -> None:
        self.model = model
        self.model_name = model_name
        self.dimension = _safe_dimension(model)

    def encode(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Encode texts with SentenceTransformers on CPU."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return _as_float32_matrix(embeddings)


class OnnxEmbeddingBackend:
    """Optimum ONNX Runtime backend wrapper with mean-pooling."""

    backend = "onnxruntime"

    def __init__(self, model: Any, tokenizer: Any, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dimension = _config_dimension(model)

    def encode(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Encode texts with an ONNX feature-extraction model on CPU."""
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        outputs = self.model(**encoded)
        token_embeddings = _last_hidden_state(outputs)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.float32)[..., None]
        summed = (token_embeddings * attention_mask).sum(axis=1)
        counts = np.clip(attention_mask.sum(axis=1), EPSILON, None)
        return _as_float32_matrix(summed / counts)


def load_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingBackend:
    """Load an embedding model, preferring ONNX Runtime and falling back to SentenceTransformers.

    The returned object exposes a small, stable ``encode`` interface used by this module.
    ONNX loading requires ``onnxruntime``, ``optimum``, and ``transformers``. If those are not
    installed or model export/loading fails, SentenceTransformers is attempted on CPU.
    """
    errors: list[str] = []

    onnx_model, onnx_error = _try_load_onnx(model_name)
    if onnx_model is not None:
        LOGGER.info("Using embedding backend: %s", onnx_model.backend)
        return onnx_model
    if onnx_error:
        errors.append(onnx_error)

    st_model, st_error = _try_load_sentence_transformer(model_name)
    if st_model is not None:
        LOGGER.info("Using embedding backend: %s", st_model.backend)
        return st_model
    if st_error:
        errors.append(st_error)

    details = " | ".join(errors) if errors else "No supported embedding backend is available."
    raise RuntimeError(f"Failed to load embedding model {model_name!r}. {details}")


def encode_texts(
    texts: Iterable[Any],
    model: EmbeddingBackend,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Encode texts in batches and return a L2-normalized float32 embedding matrix.

    Empty, null, and malformed text inputs are converted to zero vectors. Non-empty text is
    normalized for whitespace before encoding. The function preallocates the result matrix after
    the first batch, keeping memory usage predictable for large datasets such as 100K candidates.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    clean_texts = [_clean_text(text) for text in texts]
    if not clean_texts:
        dimension = _model_dimension(model) or 0
        return np.empty((0, dimension), dtype=np.float32)

    total_batches = (len(clean_texts) + batch_size - 1) // batch_size
    embeddings: np.ndarray | None = None
    offset = 0

    batch_starts = range(0, len(clean_texts), batch_size)
    for start in tqdm_loader(batch_starts, total=total_batches, desc="Encoding texts"):
        batch = clean_texts[start : start + batch_size]
        batch_embeddings = _encode_batch(batch, model, batch_size)
        if embeddings is None:
            embeddings = np.empty((len(clean_texts), batch_embeddings.shape[1]), dtype=np.float32)
        elif embeddings.shape[1] != batch_embeddings.shape[1]:
            raise ValueError("Embedding dimension changed between batches.")

        embeddings[offset : offset + len(batch)] = batch_embeddings
        offset += len(batch)

    if embeddings is None:
        dimension = _model_dimension(model) or 0
        return np.empty((0, dimension), dtype=np.float32)

    LOGGER.info("Encoded %d texts into shape %s", len(clean_texts), embeddings.shape)
    return embeddings


def encode_single(text: Any, model: EmbeddingBackend) -> np.ndarray:
    """Encode one text value and return a single L2-normalized embedding vector."""
    embeddings = encode_texts([text], model, batch_size=1)
    if embeddings.shape[0] != 1:
        raise ValueError("Expected one embedding for a single text input.")
    return embeddings[0]


def save_embeddings(embeddings: np.ndarray, filepath: str | pathlib.Path) -> None:
    """Save a validated float32 embedding matrix to a ``.npy`` file."""
    path = pathlib.Path(filepath)
    if path.suffix != ".npy":
        raise ValueError(f"Embeddings must be saved as .npy files: {path}")

    matrix = _validate_embedding_matrix(embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix, allow_pickle=False)
    LOGGER.info("Saved embeddings with shape %s to %s", matrix.shape, path)


def load_embeddings(filepath: str | pathlib.Path) -> np.ndarray:
    """Load a ``.npy`` embedding matrix and validate it is two-dimensional."""
    path = pathlib.Path(filepath)
    if path.suffix != ".npy":
        raise ValueError(f"Embeddings must be loaded from .npy files: {path}")

    try:
        embeddings = np.load(path, allow_pickle=False)
    except OSError as exc:
        raise OSError(f"Unable to load embeddings from {path}: {exc}") from exc

    matrix = _validate_embedding_matrix(embeddings)
    LOGGER.info("Loaded embeddings with shape %s from %s", matrix.shape, path)
    return matrix


def _try_load_onnx(model_name: str) -> tuple[OnnxEmbeddingBackend | None, str | None]:
    """Attempt to load an ONNX Runtime feature-extraction model."""
    model_id = _hf_model_name(model_name)
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        LOGGER.info("ONNX backend unavailable: %s", exc)
        return None, f"ONNX unavailable: {exc}"

    try:
        providers = ort.get_available_providers()
        provider = "CPUExecutionProvider" if "CPUExecutionProvider" in providers else None
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            if provider:
                model = ORTModelForFeatureExtraction.from_pretrained(
                    model_id,
                    export=True,
                    provider=provider,
                )
            else:
                model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        except TypeError:
            model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

        LOGGER.info("Loaded ONNX embedding model %s with provider=%s", model_id, provider or "default")
        return OnnxEmbeddingBackend(model=model, tokenizer=tokenizer, model_name=model_id), None
    except Exception as exc:
        LOGGER.warning("ONNX model loading failed for %s: %s", model_id, exc)
        return None, f"ONNX load failed: {exc}"


def _try_load_sentence_transformer(
    model_name: str,
) -> tuple[SentenceTransformerBackend | None, str | None]:
    """Attempt to load a SentenceTransformers model on CPU."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        LOGGER.info("SentenceTransformers backend unavailable: %s", exc)
        return None, f"SentenceTransformers unavailable: {exc}"

    try:
        model = SentenceTransformer(model_name, device="cpu")
        LOGGER.info("Loaded SentenceTransformers embedding model %s on CPU", model_name)
        return SentenceTransformerBackend(model=model, model_name=model_name), None
    except Exception as exc:
        LOGGER.error("SentenceTransformers model loading failed for %s: %s", model_name, exc)
        return None, f"SentenceTransformers load failed: {exc}"


def _encode_batch(batch: list[str], model: EmbeddingBackend, batch_size: int) -> np.ndarray:
    """Encode and normalize a single sanitized batch."""
    dimension = _model_dimension(model)
    empty_mask = np.asarray([not text for text in batch], dtype=bool)
    if empty_mask.all() and dimension:
        return np.zeros((len(batch), dimension), dtype=np.float32)

    safe_batch = [text if text else " " for text in batch]
    try:
        embeddings = _as_float32_matrix(model.encode(safe_batch, batch_size=batch_size))
    except Exception as exc:
        raise RuntimeError(f"Embedding batch encoding failed for backend {model.backend}: {exc}") from exc

    if embeddings.shape[0] != len(batch):
        raise ValueError(f"Expected {len(batch)} embeddings, received {embeddings.shape[0]}.")

    embeddings[empty_mask] = 0.0
    if not np.isfinite(embeddings).all():
        LOGGER.warning("Non-finite embedding values detected; replacing with zeros.")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    return _l2_normalize(embeddings)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized embeddings while preserving zero vectors."""
    matrix = _as_float32_matrix(embeddings)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.zeros_like(matrix, dtype=np.float32)
    valid_rows = norms[:, 0] > EPSILON
    normalized[valid_rows] = matrix[valid_rows] / norms[valid_rows]
    return normalized


def _validate_embedding_matrix(embeddings: Any) -> np.ndarray:
    """Validate and coerce an embedding matrix to finite float32 values."""
    matrix = _as_float32_matrix(embeddings)
    if matrix.shape[1] <= 0:
        raise ValueError("Embedding matrix must have at least one dimension.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    return matrix


def _as_float32_matrix(values: Any) -> np.ndarray:
    """Convert backend output to a two-dimensional float32 numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}.")
    return matrix.astype(np.float32, copy=False)


def _last_hidden_state(outputs: Any) -> np.ndarray:
    """Extract the last hidden state from common Optimum/Transformers output shapes."""
    if hasattr(outputs, "last_hidden_state"):
        hidden_state = outputs.last_hidden_state
    elif isinstance(outputs, tuple) and outputs:
        hidden_state = outputs[0]
    elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
        hidden_state = outputs["last_hidden_state"]
    else:
        raise ValueError("ONNX model output does not contain last_hidden_state.")
    return np.asarray(hidden_state, dtype=np.float32)


def _clean_text(text: Any) -> str:
    """Normalize valid text and convert malformed values to an empty string."""
    if text is None or text == -1:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    if not isinstance(text, str):
        return ""
    return " ".join(text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").split())


def _model_dimension(model: EmbeddingBackend) -> int | None:
    """Return a positive model dimension when the backend exposes one."""
    dimension = getattr(model, "dimension", None)
    return dimension if isinstance(dimension, int) and dimension > 0 else None


def _safe_dimension(model: Any) -> int | None:
    """Read the embedding dimension from a SentenceTransformers model when possible."""
    try:
        dimension = model.get_sentence_embedding_dimension()
    except Exception:
        return None
    return dimension if isinstance(dimension, int) and dimension > 0 else None


def _config_dimension(model: Any) -> int | None:
    """Read hidden size from a Transformers-compatible model config."""
    config = getattr(model, "config", None)
    dimension = getattr(config, "hidden_size", None)
    return dimension if isinstance(dimension, int) and dimension > 0 else None


def _hf_model_name(model_name: str) -> str:
    """Return a Hugging Face model id suitable for Optimum ONNX loading."""
    if "/" in model_name or pathlib.Path(model_name).exists():
        return model_name
    return f"sentence-transformers/{model_name}"
