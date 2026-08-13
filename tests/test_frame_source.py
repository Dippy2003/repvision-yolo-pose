from repvision.frame_source import EndOfStream, FrameSourceError


def test_end_of_stream_is_an_expected_source_failure() -> None:
    assert isinstance(EndOfStream("finished"), FrameSourceError)
