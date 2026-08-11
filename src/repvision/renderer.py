"""Readable OpenCV workout overlay rendering."""

from dataclasses import dataclass

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


def curl_progress(
    angle: float | None, up_threshold: float, down_threshold: float
) -> float | None:
    """Map elbow angle to extension=0 and curl=1 progress."""
    if angle is None:
        return None
    progress = (down_threshold - angle) / (down_threshold - up_threshold)
    return max(0.0, min(1.0, progress))
