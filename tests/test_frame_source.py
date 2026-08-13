from repvision.camera import CameraError
import numpy as np
import pytest

from repvision.frame_source import (
    EndOfStream,
    FrameSourceError,
    InvalidFrameError,
    validate_bgr_frame,
)


def test_end_of_stream_is_an_expected_source_failure() -> None:
    assert isinstance(EndOfStream("finished"), FrameSourceError)


def test_camera_errors_share_frame_source_base() -> None:
    assert isinstance(CameraError("failed"), FrameSourceError)


def test_validate_bgr_frame_returns_valid_image() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert validate_bgr_frame(frame, "Test source") is frame


def test_validate_bgr_frame_names_invalid_source() -> None:
    with pytest.raises(InvalidFrameError, match="Test source"):
        validate_bgr_frame(np.zeros((2, 3), dtype=np.uint8), "Test source")
