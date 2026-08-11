"""Readable OpenCV workout overlay rendering."""

from dataclasses import dataclass

import cv2

from repvision.camera import Frame
from repvision.config import Arm
from repvision.form_checker import FormFeedback
from repvision.pose_detector import ArmLandmarks, Landmark
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

    def render(
        self,
        frame: Frame,
        data: OverlayData,
        landmarks: ArmLandmarks | None = None,
    ) -> Frame:
        """Return a frame with the current workout measurements."""
        canvas = frame.copy()
        self._draw_arm(canvas, landmarks)
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
        for index, line in enumerate(overlay_lines(data)):
            cv2.putText(
                canvas,
                line,
                (16, 60 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        self._draw_progress(canvas, data.progress, panel_width)
        return canvas

    @staticmethod
    def _draw_arm(canvas: Frame, landmarks: ArmLandmarks | None) -> None:
        """Draw only available joints from the selected arm."""
        if landmarks is None:
            return
        points = (
            landmarks.shoulder,
            landmarks.elbow,
            landmarks.wrist,
            landmarks.hip,
        )
        connections = (
            (landmarks.shoulder, landmarks.elbow),
            (landmarks.elbow, landmarks.wrist),
            (landmarks.shoulder, landmarks.hip),
        )
        for start, end in connections:
            start_point = _pixel_point(start)
            end_point = _pixel_point(end)
            if start_point is not None and end_point is not None:
                cv2.line(canvas, start_point, end_point, (60, 220, 255), 3)
        for landmark in points:
            point = _pixel_point(landmark)
            if point is not None:
                cv2.circle(canvas, point, 5, (40, 255, 100), -1)

    @staticmethod
    def _draw_progress(canvas: Frame, progress: float | None, width: int) -> None:
        """Draw a clamped curl progress bar when an angle is available."""
        left, right, top, bottom = 16, max(17, width - 16), 225, 237
        cv2.rectangle(canvas, (left, top), (right, bottom), (90, 90, 90), 1)
        if progress is None:
            return
        fill = left + round((right - left) * max(0.0, min(1.0, progress)))
        cv2.rectangle(canvas, (left, top), (fill, bottom), (80, 210, 120), -1)


def overlay_lines(data: OverlayData) -> tuple[str, ...]:
    """Format stable, testable text for a workout overlay."""
    angle = "--" if data.angle is None else f"{data.angle:.1f} deg"
    return (
        f"Arm: {data.arm.value.upper()}",
        f"Reps: {data.repetitions}",
        f"Angle: {angle}",
        f"Stage: {data.stage.value.upper()}",
        f"FPS: {data.fps:.1f}",
        f"Feedback: {data.feedback.message.value}",
    )


def _pixel_point(landmark: Landmark) -> tuple[int, int] | None:
    if landmark.point is None:
        return None
    return round(landmark.point.x), round(landmark.point.y)


def curl_progress(
    angle: float | None, up_threshold: float, down_threshold: float
) -> float | None:
    """Map elbow angle to extension=0 and curl=1 progress."""
    if angle is None:
        return None
    progress = (down_threshold - angle) / (down_threshold - up_threshold)
    return max(0.0, min(1.0, progress))
