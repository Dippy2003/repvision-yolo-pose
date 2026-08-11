"""Conservative 2D form and visibility feedback."""

from dataclasses import dataclass
from enum import StrEnum


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
