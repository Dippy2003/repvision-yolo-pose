"""Readable overlay for camera-guided movement calibration."""

from dataclasses import dataclass

from repvision.calibration import CalibrationStage
from repvision.config import Arm
from repvision.pose_detector import PoseStatus


@dataclass(frozen=True, slots=True)
class CalibrationOverlay:
    """Display-only state for one calibration frame."""

    arm: Arm
    stage: CalibrationStage
    angle: float | None
    sample_count: int
    sample_target: int
    pose_status: PoseStatus


def calibration_lines(data: CalibrationOverlay) -> tuple[str, ...]:
    """Format stable calibration guidance and measurements."""
    prompts = {
        CalibrationStage.READY_EXTENDED: "Extend arm, then press SPACE",
        CalibrationStage.CAPTURING_EXTENDED: "Hold arm fully extended",
        CalibrationStage.READY_CURLED: "Curl arm, then press SPACE",
        CalibrationStage.CAPTURING_CURLED: "Hold arm fully curled",
        CalibrationStage.COMPLETE: "Calibration complete",
    }
    angle = "unavailable" if data.angle is None else f"{data.angle:.1f} deg"
    return (
        f"Arm: {data.arm.value.upper()}",
        f"Instruction: {prompts[data.stage]}",
        f"Angle: {angle}",
        f"Samples: {data.sample_count}/{data.sample_target}",
        f"Pose: {data.pose_status.value}",
    )
