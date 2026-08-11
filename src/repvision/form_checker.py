"""Conservative 2D form and visibility feedback."""

from dataclasses import dataclass
from enum import StrEnum

from repvision.angles import calculate_elbow_angle
from repvision.config import AppConfig
from repvision.pose_detector import ArmLandmarks, PoseStatus
from repvision.rep_counter import MovementStage


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


def upper_arm_drift_angle(
    landmarks: ArmLandmarks, confidence_threshold: float
) -> float | None:
    """Measure upper-arm angle away from the shoulder-to-hip torso line."""
    required = (landmarks.shoulder, landmarks.elbow, landmarks.hip)
    if not all(point.is_reliable(confidence_threshold) for point in required):
        return None
    shoulder = landmarks.shoulder.point
    elbow = landmarks.elbow.point
    hip = landmarks.hip.point
    if shoulder is None or elbow is None or hip is None:
        return None
    return calculate_elbow_angle(
        (elbow.x, elbow.y),
        (shoulder.x, shoulder.y),
        (hip.x, hip.y),
    )


class FormChecker:
    """Choose one conservative feedback message for the current pose."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def check(
        self,
        status: PoseStatus,
        landmarks: ArmLandmarks | None,
        smoothed_angle: float | None = None,
        stage: MovementStage = MovementStage.UNKNOWN,
    ) -> FormFeedback:
        """Evaluate visibility first, then the upper-arm drift heuristic."""
        visibility = feedback_for_visibility(status)
        if visibility is not None:
            return visibility
        if landmarks is not None:
            drift = upper_arm_drift_angle(
                landmarks, self.config.confidence_threshold
            )
            if drift is not None and drift > self.config.upper_arm_drift_threshold:
                return FormFeedback(
                    FeedbackMessage.ELBOW_DRIFT,
                    is_form_warning=True,
                )
        if (
            stage is MovementStage.UP
            and smoothed_angle is not None
            and self.config.up_angle_threshold
            < smoothed_angle
            < self.config.down_angle_threshold
        ):
            return FormFeedback(FeedbackMessage.FULLY_EXTEND)
        if (
            stage is MovementStage.DOWN
            and smoothed_angle is not None
            and self.config.up_angle_threshold
            < smoothed_angle
            < self.config.down_angle_threshold
        ):
            return FormFeedback(FeedbackMessage.COMPLETE_CURL)
        return FormFeedback(FeedbackMessage.GOOD_MOVEMENT)
