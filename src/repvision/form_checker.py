"""Conservative 2D form and visibility feedback."""

from dataclasses import dataclass
from enum import StrEnum

from repvision.pose_detector import PoseStatus


class FeedbackMessage(StrEnum):
    """User-facing workout feedback text."""

    GOOD_MOVEMENT = "Good movement"
    FULLY_EXTEND = "Fully extend your arm"
    COMPLETE_CURL = "Complete the curl"
    ELBOW_DRIFT = "Keep your elbow close to your body"
    MOVE_BACK = "Move back so your arm is visible"
    LOW_CONFIDENCE = "Low keypoint confidence"


@dataclass(frozen=True, slots=True)
class FormFeedback:
    """One display message with aggregate-warning classification."""

    message: FeedbackMessage
    is_form_warning: bool = False
    is_visibility_issue: bool = False


def feedback_for_visibility(status: PoseStatus) -> FormFeedback | None:
    """Translate pose visibility status before evaluating experimental form."""
    if status in (PoseStatus.NO_PERSON, PoseStatus.MISSING_KEYPOINTS):
        return FormFeedback(FeedbackMessage.MOVE_BACK, is_visibility_issue=True)
    if status is PoseStatus.LOW_CONFIDENCE:
        return FormFeedback(FeedbackMessage.LOW_CONFIDENCE, is_visibility_issue=True)
    return None
