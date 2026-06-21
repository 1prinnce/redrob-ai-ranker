"""Runtime budget tracking for long-running ranking jobs."""

import time

from .logger import get_logger


LOGGER = get_logger(__name__)


class BudgetTimer:
    """Track elapsed and remaining wall-clock time against a fixed budget."""

    def __init__(self, max_seconds: float = 300) -> None:
        """Initialize and start a timer with a maximum runtime in seconds."""
        if max_seconds <= 0:
            raise ValueError("max_seconds must be greater than zero.")

        self.max_seconds: float = float(max_seconds)
        self._started_at: float = 0.0
        self.start()

    def start(self) -> None:
        """Start or reset the timer."""
        self._started_at = time.monotonic()

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.monotonic() - self._started_at

    def remaining(self) -> float:
        """Return remaining budget time in seconds, floored at zero."""
        return max(0.0, self.max_seconds - self.elapsed())

    def check(self, label: str) -> None:
        """Log budget status and raise TimeoutError when under 10 seconds remain."""
        elapsed_seconds = self.elapsed()
        remaining_seconds = max(0.0, self.max_seconds - elapsed_seconds)
        LOGGER.info(
            "%s | elapsed=%.2fs | remaining=%.2fs",
            label,
            elapsed_seconds,
            remaining_seconds,
        )

        if remaining_seconds < 10:
            raise TimeoutError(f"Time budget nearly exhausted at {label!r}.")
