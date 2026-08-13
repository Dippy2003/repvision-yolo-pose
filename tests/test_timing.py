import pytest

from repvision.timing import (
    FpsMeter,
    FrameTimings,
    PipelineProfiler,
    summarize_durations,
)


def test_fps_meter_starts_after_two_frames() -> None:
    meter = FpsMeter()

    assert meter.update(10.0) == 0.0
    assert meter.update(10.1) == pytest.approx(10.0)


def test_fps_meter_smooths_frame_rate_changes() -> None:
    meter = FpsMeter(smoothing=0.25)
    meter.update(1.0)
    meter.update(1.1)

    assert meter.update(1.15) == pytest.approx(12.5)


def test_fps_meter_ignores_non_increasing_timestamp() -> None:
    meter = FpsMeter()
    meter.update(2.0)

    assert meter.update(2.0) == 0.0


def test_fps_meter_reset_clears_measurement() -> None:
    meter = FpsMeter()
    meter.update(1.0)
    meter.update(1.1)

    meter.reset()

    assert meter.fps == 0.0
    assert meter.update(5.0) == 0.0


@pytest.mark.parametrize("smoothing", [0.0, -0.1, 1.1])
def test_fps_meter_rejects_invalid_smoothing(smoothing: float) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        FpsMeter(smoothing)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -float("inf")])
def test_fps_meter_rejects_non_finite_timestamp(timestamp: float) -> None:
    with pytest.raises(ValueError, match="timestamp must be finite"):
        FpsMeter().update(timestamp)


def test_frame_timings_keep_pipeline_stages_typed() -> None:
    timings = FrameTimings(0.01, 0.1, 0.002, 0.003, 0.12)

    assert timings.capture_seconds == 0.01
    assert timings.inference_seconds == 0.1
    assert timings.total_seconds == 0.12


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_frame_timings_reject_invalid_duration(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        FrameTimings(value, 0.1, 0.1, 0.1, 0.4)


def test_duration_summary_reports_central_and_tail_latency() -> None:
    summary = summarize_durations([0.01, 0.02, 0.03, 0.04, 0.2])

    assert summary.sample_count == 5
    assert summary.mean_seconds == pytest.approx(0.06)
    assert summary.median_seconds == pytest.approx(0.03)
    assert summary.p95_seconds == pytest.approx(0.2)
    assert summary.minimum_seconds == pytest.approx(0.01)
    assert summary.maximum_seconds == pytest.approx(0.2)


def test_duration_summary_requires_a_sample() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_durations([])


def test_pipeline_profiler_summarizes_each_stage() -> None:
    profiler = PipelineProfiler()
    profiler.record(FrameTimings(0.01, 0.10, 0.02, 0.03, 0.16))
    profiler.record(FrameTimings(0.02, 0.20, 0.04, 0.06, 0.32))

    summary = profiler.summary()

    assert profiler.sample_count == 2
    assert summary.capture.mean_seconds == pytest.approx(0.015)
    assert summary.inference.median_seconds == pytest.approx(0.15)
    assert summary.total.maximum_seconds == pytest.approx(0.32)


def test_pipeline_profiler_reset_discards_samples() -> None:
    profiler = PipelineProfiler()
    profiler.record(FrameTimings(0.01, 0.10, 0.02, 0.03, 0.16))

    profiler.reset()

    assert profiler.sample_count == 0
    with pytest.raises(ValueError, match="at least one"):
        profiler.summary()
