"""Typed boundaries for sources that provide local BGR frames."""

from typing import Protocol

from repvision.camera import Frame


class FrameSourceError(RuntimeError):
    """Base class for expected local frame-source failures."""


class EndOfStream(FrameSourceError):
    """Raised when a finite frame source has been consumed completely."""


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
