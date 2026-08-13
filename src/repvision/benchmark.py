"""Repeatable local performance measurements for the workout pipeline."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from repvision.config import AppConfig
from repvision.frame_source import EndOfStream, FrameSource
from repvision.pose_detector import PoseDetector
from repvision.renderer import Renderer
from repvision.timing import (
    DurationSummary,
    FrameTimings,
    PipelineProfiler,
    PipelineSummary,
)
from repvision.workout import WorkoutEngine


class BenchmarkError(RuntimeError):
    """Raised when a useful local benchmark cannot be completed."""


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Scalar benchmark results that contain no image or keypoint data."""

    source: str
    input_size: int
    warmup_frames: int
    measured_frames: int
    timings: PipelineSummary


def run_benchmark(
    config: AppConfig,
    source_factory: Callable[[], FrameSource],
    *,
    measured_frames: int = 30,
    warmup_frames: int = 2,
    detector_factory: Callable[[AppConfig], PoseDetector] = PoseDetector,
    renderer_factory: Callable[[], Renderer] = Renderer,
    clock: Callable[[], float] = perf_counter,
) -> BenchmarkResult:
    """Measure the complete local pipeline without opening a display."""
    if measured_frames < 1:
        raise ValueError("measured_frames must be positive")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    detector = detector_factory(config)
    renderer = renderer_factory()
    engine = WorkoutEngine(config)
    profiler = PipelineProfiler()
    completed_warmups = 0
    source = source_factory()
    with source:
        for index in range(warmup_frames + measured_frames):
            frame_start = clock()
            try:
                frame = source.read()
            except EndOfStream:
                break
            capture_end = clock()
            observation = detector.detect(frame, engine.state.arm)
            inference_end = clock()
            analysis = engine.process(observation, inference_end)
            analysis_end = clock()
            renderer.render(
                frame,
                analysis.overlay(engine.state),
                observation.selected_arm,
            )
            render_end = clock()
            if index < warmup_frames:
                completed_warmups += 1
                continue
            profiler.record(
                FrameTimings(
                    capture_end - frame_start,
                    inference_end - capture_end,
                    analysis_end - inference_end,
                    render_end - analysis_end,
                    render_end - frame_start,
                )
            )
    if profiler.sample_count == 0:
        raise BenchmarkError("Source ended before any benchmark frames were measured.")
    return BenchmarkResult(
        source.description,
        config.input_size,
        completed_warmups,
        profiler.sample_count,
        profiler.summary(),
    )


def format_benchmark(result: BenchmarkResult) -> str:
    """Format a compact terminal report in milliseconds and FPS."""
    lines = [
        "RepVision pipeline benchmark",
        f"Source: {result.source}",
        f"Input size: {result.input_size}",
        f"Warm-up frames: {result.warmup_frames}",
        f"Measured frames: {result.measured_frames}",
    ]
    for label, summary in (
        ("Capture", result.timings.capture),
        ("Inference", result.timings.inference),
        ("Analysis", result.timings.analysis),
        ("Render", result.timings.render),
        ("Total", result.timings.total),
    ):
        lines.append(_format_duration(label, summary))
    total_mean = result.timings.total.mean_seconds
    throughput = (
        "unavailable" if total_mean == 0.0 else f"{1.0 / total_mean:.1f} FPS"
    )
    lines.append(f"Effective throughput: {throughput}")
    return "\n".join(lines)


def _format_duration(label: str, summary: DurationSummary) -> str:
    return (
        f"{label}: mean={summary.mean_seconds * 1000:.1f} ms, "
        f"median={summary.median_seconds * 1000:.1f} ms, "
        f"p95={summary.p95_seconds * 1000:.1f} ms"
    )
