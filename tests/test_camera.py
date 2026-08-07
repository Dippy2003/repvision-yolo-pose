from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from repvision.camera import Camera


@dataclass
class FakeCapture:
    opened: bool = True
    read_result: tuple[bool, NDArray[np.uint8] | None] = field(
        default_factory=lambda: (True, np.zeros((2, 2, 3), dtype=np.uint8))
    )
    released: bool = False

    def isOpened(self) -> bool:  # noqa: N802
        return self.opened and not self.released

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        return self.read_result

    def release(self) -> None:
        self.released = True


def factory_for(device: FakeCapture) -> Callable[[int], FakeCapture]:
    return lambda _index: device


def test_camera_uses_default_device_index() -> None:
    assert Camera().index == 0


def test_camera_rejects_negative_device_index() -> None:
    with pytest.raises(ValueError, match="camera index must be zero or greater"):
        Camera(index=-1)
