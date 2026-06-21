"""Candidate data loading utilities for JSONL inputs."""

from collections.abc import Iterator
from typing import Any
import gzip
import json
import pathlib
import re

from src.utils.logger import get_logger, tqdm_loader


LOGGER = get_logger(__name__)
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_\d{7}$")


def validate_candidate_id(candidate_id: object) -> bool:
    """Return whether a candidate ID matches the CAND_0000000 format."""
    return isinstance(candidate_id, str) and CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is not None


def stream_candidates(filepath: str | pathlib.Path) -> Iterator[dict[str, Any]]:
    """Stream candidate dictionaries from a .jsonl or .jsonl.gz file."""
    path = pathlib.Path(filepath)
    if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
        raise ValueError(f"Unsupported candidate file type: {path}")

    opener = gzip.open if path.name.endswith(".gz") else pathlib.Path.open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(tqdm_loader(handle, desc="Loading candidates"), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.error("Malformed JSON in %s at line %d: %s", path, line_number, exc)
                    continue
                if not isinstance(candidate, dict):
                    LOGGER.error("Expected JSON object in %s at line %d", path, line_number)
                    continue
                yield candidate
    except OSError as exc:
        LOGGER.error("Unable to read candidate file %s: %s", path, exc)
        raise


def load_all(filepath: str | pathlib.Path) -> list[dict[str, Any]]:
    """Load all candidate dictionaries from a .jsonl or .jsonl.gz file."""
    return list(stream_candidates(filepath))
