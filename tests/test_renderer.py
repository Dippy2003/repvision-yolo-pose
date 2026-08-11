import pytest

from repvision.config import Arm
from repvision.form_checker import FeedbackMessage, FormFeedback
from repvision.renderer import OverlayData, curl_progress
from repvision.rep_counter import MovementStage


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


def test_overlay_data_keeps_workout_state_typed() -> None:
    overlay = OverlayData(
        arm=Arm.RIGHT,
        repetitions=3,
        angle=91.5,
        stage=MovementStage.DOWN,
        feedback=FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        progress=0.6,
        fps=24.5,
    )

    assert overlay.arm is Arm.RIGHT
    assert overlay.repetitions == 3
    assert overlay.stage is MovementStage.DOWN
    assert not overlay.paused
