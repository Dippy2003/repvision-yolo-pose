import pytest

from repvision.config import Arm
from repvision.form_checker import (
    FeedbackMessage,
    FormChecker,
    FormFeedback,
    feedback_for_visibility,
    upper_arm_drift_angle,
)
from repvision.pose_detector import ArmLandmarks, Landmark, Point2D, PoseStatus


def test_feedback_messages_match_interface_copy() -> None:
    assert FeedbackMessage.GOOD_MOVEMENT == "Good movement"
    assert FeedbackMessage.FULLY_EXTEND == "Fully extend your arm"
    assert FeedbackMessage.COMPLETE_CURL == "Complete the curl"
    assert FeedbackMessage.ELBOW_DRIFT == "Keep your elbow close to your body"
    assert FeedbackMessage.MOVE_BACK == "Move back so your arm is visible"
    assert FeedbackMessage.LOW_CONFIDENCE == "Low keypoint confidence"


def test_feedback_defaults_to_non_warning_movement_message() -> None:
    feedback = FormFeedback(FeedbackMessage.GOOD_MOVEMENT)

    assert not feedback.is_form_warning
    assert not feedback.is_visibility_issue


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (PoseStatus.NO_PERSON, FeedbackMessage.MOVE_BACK),
        (PoseStatus.MISSING_KEYPOINTS, FeedbackMessage.MOVE_BACK),
        (PoseStatus.LOW_CONFIDENCE, FeedbackMessage.LOW_CONFIDENCE),
    ],
)
def test_visibility_status_produces_priority_feedback(
    status: PoseStatus, message: FeedbackMessage
) -> None:
    feedback = feedback_for_visibility(status)

    assert feedback is not None
    assert feedback.message is message
    assert feedback.is_visibility_issue
    assert not feedback.is_form_warning


def test_tracking_status_allows_form_evaluation() -> None:
    assert feedback_for_visibility(PoseStatus.TRACKING) is None


def arm_landmarks(
    *,
    shoulder: tuple[float, float] = (0.0, 0.0),
    elbow: tuple[float, float] = (0.0, 1.0),
    wrist: tuple[float, float] = (0.0, 2.0),
    hip: tuple[float, float] = (0.0, 2.0),
    confidence: float = 0.9,
) -> ArmLandmarks:
    def landmark(point: tuple[float, float]) -> Landmark:
        return Landmark(Point2D(*point), confidence)

    return ArmLandmarks(
        Arm.RIGHT,
        landmark(shoulder),
        landmark(elbow),
        landmark(wrist),
        landmark(hip),
    )


def test_upper_arm_drift_compares_arm_with_torso_line() -> None:
    close = arm_landmarks(elbow=(0.0, 1.0))
    diagonal = arm_landmarks(elbow=(1.0, 1.0))

    assert upper_arm_drift_angle(close, 0.5) == pytest.approx(0.0)
    assert upper_arm_drift_angle(diagonal, 0.5) == pytest.approx(45.0)


def test_form_checker_warns_when_upper_arm_exceeds_drift_limit() -> None:
    checker = FormChecker(AppConfig(upper_arm_drift_threshold=30.0))

    feedback = checker.check(
        PoseStatus.TRACKING,
        arm_landmarks(elbow=(1.0, 1.0)),
    )

    assert feedback.message is FeedbackMessage.ELBOW_DRIFT
    assert feedback.is_form_warning
    assert not feedback.is_visibility_issue


def test_form_checker_accepts_conservative_upper_arm_position() -> None:
    checker = FormChecker(AppConfig(upper_arm_drift_threshold=30.0))

    feedback = checker.check(PoseStatus.TRACKING, arm_landmarks(elbow=(0.2, 1.0)))

    assert feedback == FormFeedback(FeedbackMessage.GOOD_MOVEMENT)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (PoseStatus.NO_PERSON, FeedbackMessage.MOVE_BACK),
        (PoseStatus.MISSING_KEYPOINTS, FeedbackMessage.MOVE_BACK),
        (PoseStatus.LOW_CONFIDENCE, FeedbackMessage.LOW_CONFIDENCE),
    ],
)
def test_visibility_feedback_takes_priority_over_drift(
    status: PoseStatus, expected: FeedbackMessage
) -> None:
    checker = FormChecker(AppConfig(upper_arm_drift_threshold=30.0))

    feedback = checker.check(status, arm_landmarks(elbow=(2.0, 1.0)))

    assert feedback.message is expected
    assert feedback.is_visibility_issue
    assert not feedback.is_form_warning
