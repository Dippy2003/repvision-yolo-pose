"""Personal arm-range calibration and local profile persistence."""

from enum import StrEnum


class CalibrationError(RuntimeError):
    """Base class for expected calibration failures."""


class CalibrationRangeError(CalibrationError):
    """Raised when captured positions do not define enough movement range."""


class CalibrationStorageError(CalibrationError):
    """Raised when calibration profiles cannot be loaded or saved safely."""


class CalibrationPosition(StrEnum):
    """Arm endpoints captured during calibration."""

    EXTENDED = "extended"
    CURLED = "curled"
