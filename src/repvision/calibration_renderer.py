"""Readable overlay for camera-guided movement calibration."""

from dataclasses import dataclass
from math import isfinite

import cv2

from repvision.calibration import CalibrationStage
from repvision.camera import Frame
from repvision.config import Arm
from repvision.pose_detector import ArmLandmarks, PoseStatus
from repvision.renderer import draw_arm


@dataclass(frozen=True, slots=True)
class CalibrationOverlay:
    """Display-only state for one calibration frame."""

    arm: Arm
    stage: CalibrationStage
    angle: float | None
    sample_count: int
    sample_target: int
    pose_status: PoseStatus

    def __post_init__(self) -> None:
        if self.sample_target < 1:
            raise ValueError("sample_target must be positive")
        if not 0 <= self.sample_count <= self.sample_target:
            raise ValueError("sample_count must be between zero and sample_target")
        if self.angle is not None and (
            not isfinite(self.angle) or not 0.0 <= self.angle <= 180.0
        ):
            raise ValueError("angle must be unavailable or between 0 and 180")


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


class CalibrationRenderer:
    """Render guided calibration without mutating camera frames."""

    def render(
        self,
        frame: Frame,
        data: CalibrationOverlay,
        landmarks: ArmLandmarks | None = None,
    ) -> Frame:
        """Return a frame containing instructions and capture progress."""
        canvas = frame.copy()
        draw_arm(canvas, landmarks)
        panel_width = min(460, canvas.shape[1])
        cv2.rectangle(canvas, (0, 0), (panel_width, 190), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            "RepVision | Calibration",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for index, line in enumerate(calibration_lines(data)):
            cv2.putText(
                canvas,
                line,
                (16, 58 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "SPACE Confirm | R Restart | Q Cancel",
            (12, max(18, canvas.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas
