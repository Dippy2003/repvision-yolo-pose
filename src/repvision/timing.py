"""Small deterministic timing helpers for the workout loop."""

from time import monotonic


class FpsMeter:
    """Smooth frame-rate estimates to keep the overlay readable."""

    def __init__(self, smoothing: float = 0.2) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        self.smoothing = smoothing
        self.fps = 0.0
        self._last_timestamp: float | None = None

    def update(self, timestamp: float | None = None) -> float:
        """Record one frame timestamp and return the smoothed frame rate."""
        now = monotonic() if timestamp is None else timestamp
        if self._last_timestamp is not None:
            elapsed = now - self._last_timestamp
            if elapsed > 0.0:
                measured = 1.0 / elapsed
                self.fps = (
                    measured
                    if self.fps == 0.0
                    else self.fps + self.smoothing * (measured - self.fps)
                )
        self._last_timestamp = now
        return self.fps

    def reset(self) -> None:
        """Clear both the displayed value and the previous timestamp."""
        self.fps = 0.0
        self._last_timestamp = None
