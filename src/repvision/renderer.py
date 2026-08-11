"""Readable OpenCV workout overlay rendering."""

from dataclasses import dataclass

import cv2

from repvision.camera import Frame
from repvision.config import Arm
from repvision.form_checker import FormFeedback
from repvision.rep_counter import MovementStage


@dataclass(frozen=True, slots=True)
class OverlayData:
    """Application state required to render one workout frame."""

    arm: Arm
    repetitions: int
    angle: float | None
    stage: MovementStage
    feedback: FormFeedback
    progress: float | None
    fps: float
    paused: bool = False


class Renderer:
    """Render an uncluttered workout HUD without mutating the source frame."""

    def render(self, frame: Frame, data: OverlayData) -> Frame:
        """Return a frame with the RepVision title panel."""
        canvas = frame.copy()
        panel_width = min(390, canvas.shape[1])
        cv2.rectangle(canvas, (0, 0), (panel_width, 245), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            "RepVision | Bicep Curl",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas


def curl_progress(
    angle: float | None, up_threshold: float, down_threshold: float
) -> float | None:
    """Map elbow angle to extension=0 and curl=1 progress."""
    if angle is None:
        return None
    progress = (down_threshold - angle) / (down_threshold - up_threshold)
    return max(0.0, min(1.0, progress))
