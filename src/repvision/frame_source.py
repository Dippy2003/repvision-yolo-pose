"""Typed boundaries for sources that provide local BGR frames."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class CaptureDevice(Protocol):
    """OpenCV-compatible capture surface shared by local sources."""

    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, Frame | None]: ...

    def release(self) -> None: ...


class FrameSourceError(RuntimeError):
    """Base class for expected local frame-source failures."""


class EndOfStream(FrameSourceError):
    """Raised when a finite frame source has been consumed completely."""


class InvalidFrameError(FrameSourceError):
    """Raised when a source does not provide uint8 BGR image data."""


def validate_bgr_frame(frame: object, source: str) -> Frame:
    """Return a valid OpenCV BGR frame or raise a focused source error."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.size == 0
    ):
        raise InvalidFrameError(f"{source} returned an invalid BGR frame.")
    return frame


class FrameSource(Protocol):
    """Resource-owning source used by the workout and benchmark loops."""

    @property
    def description(self) -> str: ...

    def open(self) -> None: ...

    def read(self) -> Frame: ...

    def release(self) -> None: ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None: ...
