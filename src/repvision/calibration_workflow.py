"""Camera orchestration for private guided arm calibration."""

from repvision.angles import calculate_arm_angle
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
