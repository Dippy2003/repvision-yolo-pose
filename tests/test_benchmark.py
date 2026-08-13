from repvision.benchmark import BenchmarkResult, format_benchmark
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
