"""Camera orchestration for private guided arm calibration."""

from repvision.angles import calculate_arm_angle
from repvision.calibration import GuidedCalibration
from repvision.calibration_renderer import CalibrationOverlay
from repvision.config import AppConfig
from repvision.pose_detector import PoseObservation, PoseStatus


def calibration_angle(
    observation: PoseObservation, config: AppConfig
) -> float | None:
    """Return a reliable raw elbow angle suitable for endpoint sampling."""
    if observation.status is not PoseStatus.TRACKING:
        return None
    return calculate_arm_angle(
        observation.selected_arm,
        config.confidence_threshold,
    )


def calibration_overlay(
    workflow: GuidedCalibration,
    observation: PoseObservation,
    angle: float | None,
) -> CalibrationOverlay:
    """Build display data from the current guided workflow state."""
    return CalibrationOverlay(
        workflow.arm,
        workflow.stage,
        angle,
        workflow.sample_count,
        workflow.collector.config.calibration_sample_target,
        observation.status,
    )
