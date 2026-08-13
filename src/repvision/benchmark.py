"""Repeatable local performance measurements for the workout pipeline."""

from dataclasses import dataclass

from repvision.timing import DurationSummary, PipelineSummary


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Scalar benchmark results that contain no image or keypoint data."""

    source: str
    input_size: int
    warmup_frames: int
    measured_frames: int
    timings: PipelineSummary


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
    fps = 1.0 / result.timings.total.mean_seconds
    lines.append(f"Effective throughput: {fps:.1f} FPS")
    return "\n".join(lines)


def _format_duration(label: str, summary: DurationSummary) -> str:
    return (
        f"{label}: mean={summary.mean_seconds * 1000:.1f} ms, "
        f"median={summary.median_seconds * 1000:.1f} ms, "
        f"p95={summary.p95_seconds * 1000:.1f} ms"
    )
