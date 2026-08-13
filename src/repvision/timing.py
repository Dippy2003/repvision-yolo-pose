"""Small deterministic timing helpers for the workout loop."""

from dataclasses import dataclass
from math import ceil, isfinite
from statistics import fmean, median
from time import monotonic


@dataclass(frozen=True, slots=True)
class FrameTimings:
    """Measured durations for one fully processed frame."""

    capture_seconds: float
    inference_seconds: float
    analysis_seconds: float
    render_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.capture_seconds,
            self.inference_seconds,
            self.analysis_seconds,
            self.render_seconds,
            self.total_seconds,
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("frame timing values must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DurationSummary:
    """Descriptive statistics for one measured pipeline stage."""

    sample_count: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    minimum_seconds: float
    maximum_seconds: float


def summarize_durations(values: list[float]) -> DurationSummary:
    """Summarize non-negative durations using nearest-rank p95."""
    if not values:
        raise ValueError("at least one duration is required")
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("durations must be finite and non-negative")
    ordered = sorted(values)
    p95_index = ceil(0.95 * len(ordered)) - 1
    return DurationSummary(
        len(ordered),
        fmean(ordered),
        median(ordered),
        ordered[p95_index],
        ordered[0],
        ordered[-1],
    )


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """Latency summaries for each measured frame-processing stage."""

    capture: DurationSummary
    inference: DurationSummary
    analysis: DurationSummary
    render: DurationSummary
    total: DurationSummary


class PipelineProfiler:
    """Collect bounded scalar timings without retaining any frame data."""

    def __init__(self) -> None:
        self._samples: list[FrameTimings] = []

    @property
    def sample_count(self) -> int:
        """Return the number of recorded frames."""
        return len(self._samples)

    def record(self, timings: FrameTimings) -> None:
        """Append timing values for one frame."""
        self._samples.append(timings)

    def summary(self) -> PipelineSummary:
        """Summarize every stage across all recorded frames."""
        return PipelineSummary(
            summarize_durations([item.capture_seconds for item in self._samples]),
            summarize_durations([item.inference_seconds for item in self._samples]),
            summarize_durations([item.analysis_seconds for item in self._samples]),
            summarize_durations([item.render_seconds for item in self._samples]),
            summarize_durations([item.total_seconds for item in self._samples]),
        )

    def reset(self) -> None:
        """Discard recorded scalar timings."""
        self._samples.clear()


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
        if not isfinite(now):
            raise ValueError("timestamp must be finite")
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
