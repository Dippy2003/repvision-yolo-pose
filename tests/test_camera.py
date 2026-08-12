from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from repvision.camera import (
    Camera,
    CameraNotOpenError,
    CameraOpenError,
    CameraReleaseError,
    FrameReadError,
    InvalidFrameError,
)


@dataclass
class FakeCapture:
    opened: bool = True
    read_result: tuple[bool, NDArray[np.uint8] | None] = field(
        default_factory=lambda: (True, np.zeros((2, 2, 3), dtype=np.uint8))
    )
    released: bool = False

    def isOpened(self) -> bool:
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


def test_open_wraps_capture_backend_failure() -> None:
    def failing_factory(_index: int) -> FakeCapture:
        raise RuntimeError("backend unavailable")

    with pytest.raises(CameraOpenError, match="backend unavailable"):
        Camera(index=1, capture_factory=failing_factory).open()


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


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_read_rejects_invalid_bgr_frames(frame: NDArray[np.generic]) -> None:
    device = FakeCapture(read_result=(True, frame))  # type: ignore[arg-type]
    camera = Camera(index=2, capture_factory=factory_for(device))
    camera.open()

    with pytest.raises(InvalidFrameError, match="invalid BGR frame"):
        camera.read()


def test_read_wraps_capture_backend_failure() -> None:
    device = FakeCapture()

    def failing_read() -> tuple[bool, NDArray[np.uint8] | None]:
        raise RuntimeError("device disconnected")

    device.read = failing_read  # type: ignore[method-assign]
    camera = Camera(index=2, capture_factory=factory_for(device))
    camera.open()

    with pytest.raises(FrameReadError, match="device disconnected"):
        camera.read()


def test_release_is_safe_to_call_more_than_once() -> None:
    device = FakeCapture()
    camera = Camera(capture_factory=factory_for(device))
    camera.open()

    camera.release()
    camera.release()

    assert device.released
    assert not camera.is_open


def test_release_clears_camera_after_backend_failure() -> None:
    device = FakeCapture()

    def failing_release() -> None:
        raise RuntimeError("release failed")

    device.release = failing_release  # type: ignore[method-assign]
    camera = Camera(capture_factory=factory_for(device))
    camera.open()

    with pytest.raises(CameraReleaseError, match="release failed"):
        camera.release()

    assert not camera.is_open
    camera.release()


def test_context_manager_releases_capture_after_an_error() -> None:
    device = FakeCapture()
    camera = Camera(capture_factory=factory_for(device))

    with pytest.raises(RuntimeError, match="processing failed"), camera:
        raise RuntimeError("processing failed")

    assert device.released
    assert not camera.is_open
