"""Webcam capture boundary for the desktop application."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class CaptureDevice(Protocol):
    """Small interface implemented by OpenCV and test capture devices."""

    def isOpened(self) -> bool: ...

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

        try:
            capture = self._capture_factory(self.index)
        except (OSError, RuntimeError) as error:
            raise CameraOpenError(
                f"Could not create camera index {self.index}: {error}"
            ) from error
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

        try:
            success, frame = self._capture.read()
        except (OSError, RuntimeError) as error:
            raise FrameReadError(
                f"Camera index {self.index} failed while reading: {error}"
            ) from error
        if not success or frame is None:
            raise FrameReadError(
                f"Could not read a frame from camera index {self.index}."
            )
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise InvalidFrameError(
                f"Camera index {self.index} returned an invalid BGR frame."
            )
        return frame

    def release(self) -> None:
        """Release the owned capture device; repeated calls are safe."""
        if self._capture is not None:
            capture = self._capture
            self._capture = None
            try:
                capture.release()
            except (OSError, RuntimeError) as error:
                raise CameraReleaseError(
                    f"Could not release camera index {self.index}: {error}"
                ) from error

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            self.release()
        except CameraReleaseError:
            if exc_type is None:
                raise


class CameraError(RuntimeError):
    """Base class for expected camera failures."""


class CameraOpenError(CameraError):
    """Raised when a configured camera cannot be opened."""


class CameraNotOpenError(CameraError):
    """Raised when frames are requested before opening a camera."""


class FrameReadError(CameraError):
    """Raised when an opened camera does not provide a frame."""


class InvalidFrameError(CameraError):
    """Raised when a camera frame cannot be processed as uint8 BGR data."""


class CameraReleaseError(CameraError):
    """Raised when the capture backend cannot release the camera."""
