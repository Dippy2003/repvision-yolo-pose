from repvision.camera import CameraError
from repvision.frame_source import EndOfStream, FrameSourceError


def test_end_of_stream_is_an_expected_source_failure() -> None:
    assert isinstance(EndOfStream("finished"), FrameSourceError)


def test_camera_errors_share_frame_source_base() -> None:
    assert isinstance(CameraError("failed"), FrameSourceError)
