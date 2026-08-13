from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from repvision.video_source import VideoFileSource


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
