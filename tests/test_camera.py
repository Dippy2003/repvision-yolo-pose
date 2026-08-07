from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from repvision.camera import Camera, CameraNotOpenError, CameraOpenError, FrameReadError


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


def test_open_passes_configured_index_to_capture_factory() -> None:
    requested_indices: list[int] = []
    device = FakeCapture()

    camera = Camera(
        index=2,
        capture_factory=lambda index: requested_indices.append(index) or device,
    )
    camera.open()

    assert requested_indices == [2]
    assert camera.is_open


def test_open_failure_releases_unusable_capture() -> None:
    device = FakeCapture(opened=False)
    camera = Camera(index=4, capture_factory=factory_for(device))

    with pytest.raises(CameraOpenError, match="Could not open camera index 4"):
        camera.open()

    assert device.released
    assert not camera.is_open


def test_read_returns_frame_from_open_capture() -> None:
    expected = np.full((3, 4, 3), 17, dtype=np.uint8)
    device = FakeCapture(read_result=(True, expected))
    camera = Camera(capture_factory=factory_for(device))
    camera.open()

    assert camera.read() is expected


def test_read_requires_an_open_camera() -> None:
    with pytest.raises(CameraNotOpenError, match="opened before reading"):
        Camera().read()


@pytest.mark.parametrize("read_result", [(False, None), (True, None)])
def test_read_rejects_missing_frames(
    read_result: tuple[bool, NDArray[np.uint8] | None],
) -> None:
    device = FakeCapture(read_result=read_result)
    camera = Camera(index=3, capture_factory=factory_for(device))
    camera.open()

    with pytest.raises(FrameReadError, match="camera index 3"):
        camera.read()


def test_release_is_safe_to_call_more_than_once() -> None:
    device = FakeCapture()
    camera = Camera(capture_factory=factory_for(device))
    camera.open()

    camera.release()
    camera.release()

    assert device.released
    assert not camera.is_open


def test_context_manager_releases_capture_after_an_error() -> None:
    device = FakeCapture()
    camera = Camera(capture_factory=factory_for(device))

    with pytest.raises(RuntimeError, match="processing failed"):
        with camera:
            raise RuntimeError("processing failed")

    assert device.released
    assert not camera.is_open
