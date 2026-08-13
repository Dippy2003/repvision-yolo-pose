"""Read-only local video frame source."""

from collections.abc import Callable
from pathlib import Path

from repvision.camera import CaptureDevice
from repvision.frame_source import (
    EndOfStream,
    Frame,
    FrameSourceError,
    validate_bgr_frame,
)

VideoCaptureFactory = Callable[[str], CaptureDevice]


def open_opencv_video(path: str) -> CaptureDevice:
    """Open a local video through OpenCV."""
    import cv2

    return cv2.VideoCapture(path)


class VideoSourceError(FrameSourceError):
    """Raised when a local video cannot be opened or read safely."""


class VideoFileSource:
    """Own an OpenCV capture for an existing local video file."""

    def __init__(
        self,
        path: Path,
        capture_factory: VideoCaptureFactory = open_opencv_video,
    ) -> None:
        self.path = path
        self._capture_factory = capture_factory
        self._capture: CaptureDevice | None = None

    @property
    def description(self) -> str:
        """Return the input filename without copying its contents."""
        return f"video {self.path}"

    def open(self) -> None:
        """Open an existing regular file as a video source."""
        if not self.path.exists():
            raise VideoSourceError(f"Video file does not exist: {self.path}")
        if not self.path.is_file():
            raise VideoSourceError(f"Video path is not a file: {self.path}")
        try:
            capture = self._capture_factory(str(self.path))
        except (OSError, RuntimeError) as error:
            raise VideoSourceError(
                f"Could not create video capture for {self.path}: {error}"
            ) from error
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Could not open video file: {self.path}")
        self._capture = capture

    def read(self) -> Frame:
        """Return the next frame or signal normal end-of-stream."""
        if self._capture is None or not self._capture.isOpened():
            raise VideoSourceError("Video must be opened before reading frames.")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise EndOfStream(f"Reached the end of video: {self.path}")
        return validate_bgr_frame(frame, self.description.capitalize())

    def release(self) -> None:
        """Release the video capture; repeated calls are safe."""
        if self._capture is not None:
            capture = self._capture
            self._capture = None
            capture.release()

    def __enter__(self) -> "VideoFileSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self.release()
