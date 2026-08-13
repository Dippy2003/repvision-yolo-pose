from collections.abc import Iterator

import numpy as np
import pytest

from repvision.benchmark import BenchmarkResult, format_benchmark, run_benchmark
from repvision.config import AppConfig, Arm
from repvision.frame_source import Frame
from repvision.pose_detector import PoseObservation, PoseStatus
from repvision.timing import DurationSummary, PipelineSummary


def duration(value: float, count: int = 3) -> DurationSummary:
    return DurationSummary(count, value, value, value, value, value)


def test_benchmark_report_formats_latency_and_throughput() -> None:
    result = BenchmarkResult(
        "video curl.mp4",
        480,
        2,
        3,
        PipelineSummary(
            duration(0.01),
            duration(0.10),
            duration(0.02),
            duration(0.03),
            duration(0.20),
        ),
    )

    report = format_benchmark(result)

    assert "Source: video curl.mp4" in report
    assert "Input size: 480" in report
    assert "Inference: mean=100.0 ms" in report
    assert "Effective throughput: 5.0 FPS" in report


class FakeSource:
    description = "finite benchmark source"

    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.released = False

    def open(self) -> None:
        pass

    def read(self) -> Frame:
        self.frame_count -= 1
        return np.zeros((300, 500, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True

    def __enter__(self) -> "FakeSource":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class FakeDetector:
    def __init__(self, _config: AppConfig) -> None:
        pass

    def detect(self, _frame: Frame, _arm: Arm) -> PoseObservation:
        return PoseObservation((), None, None, PoseStatus.NO_PERSON)


def increasing_clock(step: float = 0.01) -> Iterator[float]:
    value = 0.0
    while True:
        yield value
        value += step


def test_run_benchmark_measures_each_pipeline_stage() -> None:
    source = FakeSource(3)
    timestamps = increasing_clock()

    result = run_benchmark(
        AppConfig(input_size=480),
        lambda: source,
        measured_frames=2,
        warmup_frames=1,
        detector_factory=FakeDetector,
        clock=lambda: next(timestamps),
    )

    assert source.released
    assert result.source == "finite benchmark source"
    assert result.warmup_frames == 1
    assert result.measured_frames == 2
    assert result.timings.capture.mean_seconds == pytest.approx(0.01)
    assert result.timings.total.mean_seconds == pytest.approx(0.04)
