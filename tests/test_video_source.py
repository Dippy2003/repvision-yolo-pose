from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from repvision.video_source import VideoFileSource, VideoSourceError


@dataclass
class FakeVideoCapture:
    opened: bool = True
    frames: list[np.ndarray] = field(default_factory=list)
    released: bool = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


def make_video_path(tmp_path: Path) -> Path:
    path = tmp_path / "workout.mp4"
    path.write_bytes(b"test fixture placeholder")
    return path


def test_video_source_describes_local_path(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)

    assert VideoFileSource(path).description == f"video {path}"


def test_video_source_rejects_missing_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.mp4"

    with pytest.raises(VideoSourceError, match="does not exist"):
        VideoFileSource(path).open()


def test_video_source_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="not a file"):
        VideoFileSource(tmp_path).open()


def test_video_source_opens_existing_file(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    capture = FakeVideoCapture()
    requested_paths: list[str] = []
    source = VideoFileSource(
        path,
        capture_factory=lambda value: requested_paths.append(value) or capture,
    )

    source.open()

    assert requested_paths == [str(path)]
    assert not capture.released
    source.release()


def test_video_source_releases_unusable_capture(tmp_path: Path) -> None:
    path = make_video_path(tmp_path)
    capture = FakeVideoCapture(opened=False)

    with pytest.raises(VideoSourceError, match="Could not open video"):
        VideoFileSource(path, capture_factory=lambda _path: capture).open()

    assert capture.released
