from repvision.calibration import (
    CalibrationError,
    CalibrationPosition,
    CalibrationRangeError,
    CalibrationStorageError,
)


def test_calibration_positions_have_stable_values() -> None:
    assert CalibrationPosition.EXTENDED.value == "extended"
    assert CalibrationPosition.CURLED.value == "curled"


def test_calibration_failures_share_public_base() -> None:
    assert isinstance(CalibrationRangeError("range"), CalibrationError)
    assert isinstance(CalibrationStorageError("storage"), CalibrationError)
