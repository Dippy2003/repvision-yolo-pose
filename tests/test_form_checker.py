import pytest

from repvision.form_checker import (
    FeedbackMessage,
    FormFeedback,
    feedback_for_visibility,
)
from repvision.pose_detector import PoseStatus


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
