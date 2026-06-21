"""Logging and progress-display utilities for the ranking pipeline."""

from collections.abc import Iterable, Iterator
import logging
from typing import TypeVar

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only without optional tqdm.
    tqdm = None  # type: ignore[assignment]

T = TypeVar("T")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return an INFO-level logger configured for stream output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger


def tqdm_loader(iterable: Iterable[T], total: int | None = None, desc: str = "") -> Iterator[T]:
    """Yield items from an iterable with optional tqdm progress output."""
    if tqdm is None:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc)
