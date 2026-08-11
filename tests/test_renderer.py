import pytest

from repvision.renderer import curl_progress


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (None, None),
        (180.0, 0.0),
        (155.0, 0.0),
        (102.5, 0.5),
        (50.0, 1.0),
        (20.0, 1.0),
    ],
)
def test_curl_progress_clamps_to_configured_range(
    angle: float | None, expected: float | None
) -> None:
    progress = curl_progress(angle, up_threshold=50.0, down_threshold=155.0)

    if expected is None:
        assert progress is None
    else:
        assert progress == pytest.approx(expected)
