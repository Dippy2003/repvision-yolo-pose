"""Webcam capture boundary for the desktop application."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class CaptureDevice(Protocol):
    """Small interface implemented by OpenCV and test capture devices."""

    def isOpened(self) -> bool: ...  # noqa: N802

    def read(self) -> tuple[bool, Frame | None]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int], CaptureDevice]


def open_opencv_capture(index: int) -> CaptureDevice:
    """Create the real OpenCV video capture without hiding it in tests."""
    import cv2

    return cv2.VideoCapture(index)


class CameraError(RuntimeError):
    """Base class for expected camera failures."""


class CameraOpenError(CameraError):
    """Raised when a configured camera cannot be opened."""


class CameraNotOpenError(CameraError):
    """Raised when frames are requested before opening a camera."""


class FrameReadError(CameraError):
    """Raised when an opened camera does not provide a frame."""
