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


class Camera:
    """Own a single webcam capture device."""

    def __init__(
        self,
        index: int = 0,
        capture_factory: CaptureFactory = open_opencv_capture,
    ) -> None:
        if index < 0:
            raise ValueError("camera index must be zero or greater")
        self.index = index
        self._capture_factory = capture_factory
        self._capture: CaptureDevice | None = None

    @property
    def is_open(self) -> bool:
        """Return whether this wrapper owns a usable capture device."""
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        """Open the configured camera or raise an understandable error."""
        if self.is_open:
            return

        capture = self._capture_factory(self.index)
        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(
                f"Could not open camera index {self.index}. "
                "Check that it is connected and not in use by another application."
            )
        self._capture = capture

    def read(self) -> Frame:
        """Read one frame from the open camera."""
        if self._capture is None or not self._capture.isOpened():
            raise CameraNotOpenError("Camera must be opened before reading frames.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise FrameReadError(
                f"Could not read a frame from camera index {self.index}."
            )
        return frame


class CameraError(RuntimeError):
    """Base class for expected camera failures."""


class CameraOpenError(CameraError):
    """Raised when a configured camera cannot be opened."""


class CameraNotOpenError(CameraError):
    """Raised when frames are requested before opening a camera."""


class FrameReadError(CameraError):
    """Raised when an opened camera does not provide a frame."""
