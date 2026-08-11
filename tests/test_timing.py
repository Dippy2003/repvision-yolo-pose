import pytest

from repvision.timing import FpsMeter


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
