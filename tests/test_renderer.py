from unittest.mock import patch

import numpy as np
import pytest

from repvision.config import Arm
from repvision.form_checker import FeedbackMessage, FormFeedback
from repvision.pose_detector import ArmLandmarks, Landmark, Point2D
from repvision.renderer import OverlayData, Renderer, curl_progress, overlay_lines
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


@pytest.mark.parametrize(
    ("up_threshold", "down_threshold"),
    [(50.0, 50.0), (160.0, 50.0), (-1.0, 155.0), (50.0, 181.0)],
)
def test_curl_progress_rejects_invalid_thresholds(
    up_threshold: float, down_threshold: float
) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        curl_progress(90.0, up_threshold, down_threshold)


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


def test_overlay_lines_format_all_workout_measurements() -> None:
    overlay = OverlayData(
        Arm.LEFT,
        3,
        91.5,
        MovementStage.DOWN,
        FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        0.6,
        24.52,
    )

    assert overlay_lines(overlay) == (
        "Arm: LEFT",
        "Reps: 3",
        "Angle: 91.5 deg",
        "Stage: DOWN",
        "FPS: 24.5",
        "Feedback: Good movement",
    )


def test_renderer_adds_panel_without_mutating_source_frame() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    source = frame.copy()
    overlay = OverlayData(
        Arm.RIGHT,
        0,
        None,
        MovementStage.UNKNOWN,
        FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        None,
        0.0,
    )

    rendered = Renderer().render(frame, overlay)

    np.testing.assert_array_equal(frame, source)
    assert rendered.shape == frame.shape
    assert np.any(rendered != source)


def test_renderer_draws_progress_fill() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    overlay = OverlayData(
        Arm.RIGHT,
        1,
        100.0,
        MovementStage.DOWN,
        FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        0.5,
        30.0,
    )

    rendered = Renderer().render(frame, overlay)

    assert tuple(rendered[230, 100]) == (80, 210, 120)
    assert tuple(rendered[230, 350]) != (80, 210, 120)


def test_renderer_draws_selected_arm_landmarks() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    overlay = OverlayData(
        Arm.RIGHT,
        0,
        90.0,
        MovementStage.UNKNOWN,
        FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        0.5,
        0.0,
    )
    landmarks = ArmLandmarks(
        Arm.RIGHT,
        Landmark(Point2D(420, 40), 0.9),
        Landmark(Point2D(430, 100), 0.9),
        Landmark(Point2D(450, 160), 0.9),
        Landmark(Point2D(410, 220), 0.9),
    )

    rendered = Renderer().render(frame, overlay, landmarks)

    assert tuple(rendered[100, 430]) == (40, 255, 100)
    assert np.any(rendered[45:95, 415:435] != 0)


def test_renderer_marks_paused_workout() -> None:
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    overlay = OverlayData(
        Arm.RIGHT,
        2,
        80.0,
        MovementStage.UP,
        FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        0.7,
        20.0,
        paused=True,
    )

    rendered = Renderer().render(frame, overlay)

    assert np.any(rendered[165:235, 200:400] != 0)


def test_renderer_displays_keyboard_help() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    overlay = OverlayData(
        Arm.RIGHT,
        0,
        None,
        MovementStage.UNKNOWN,
        FormFeedback(FeedbackMessage.LOW_CONFIDENCE),
        None,
        0.0,
    )

    with patch("repvision.renderer.cv2.putText") as put_text:
        Renderer().render(frame, overlay)

    rendered_text = [call.args[1] for call in put_text.call_args_list]
    assert "Q Quit | R Reset | P Pause | L Switch arm" in rendered_text
