"""Readable OpenCV workout overlay rendering."""

from dataclasses import dataclass

import cv2

from repvision.camera import Frame
from repvision.config import Arm
from repvision.form_checker import FormFeedback
from repvision.pose_detector import ArmLandmarks, Landmark
from repvision.rep_counter import MovementStage

SIDE_PANEL_WIDTH = 360
PROGRESS_TOP = 252
PROGRESS_BOTTOM = 264
CONTROLS_LINE_HEIGHT = 24
CONTROLS_BOTTOM_MARGIN = 38
CONTROLS_MINIMUM_BASELINE = PROGRESS_BOTTOM + 20


def side_panel_canvas(
    frame: Frame, panel_width: int = SIDE_PANEL_WIDTH
) -> tuple[Frame, int]:
    """Place a dark information panel beside the complete camera frame."""
    if panel_width < 1:
        raise ValueError("panel_width must be positive")
    canvas = cv2.copyMakeBorder(
        frame,
        0,
        0,
        0,
        panel_width,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )
    return canvas, frame.shape[1]


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
    personalized: bool = False


class Renderer:
    """Render an uncluttered workout HUD without mutating the source frame."""

    def render(
        self,
        frame: Frame,
        data: OverlayData,
        landmarks: ArmLandmarks | None = None,
    ) -> Frame:
        """Return a frame with the current workout measurements."""
        canvas, panel_left = side_panel_canvas(frame)
        draw_arm(canvas, landmarks)
        text_left = panel_left + 16
        cv2.putText(
            canvas,
            "RepVision | Bicep Curl",
            (text_left, 30),
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
                (text_left, 60 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        self._draw_progress(canvas, data.progress, panel_left)
        if data.paused:
            self._draw_paused_banner(canvas, panel_left)
        self._draw_controls(canvas, panel_left)
        return canvas

    @staticmethod
    def _draw_controls(canvas: Frame, panel_left: int) -> None:
        """Show the keyboard controls without overdrawing the progress bar."""
        first_baseline = max(
            CONTROLS_MINIMUM_BASELINE,
            canvas.shape[0] - CONTROLS_BOTTOM_MARGIN,
        )
        for index, text in enumerate(("Q Quit | R Reset", "P Pause | L Switch arm")):
            cv2.putText(
                canvas,
                text,
                (panel_left + 16, first_baseline + index * CONTROLS_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_paused_banner(canvas: Frame, camera_width: int) -> None:
        """Make paused state unmistakable without obscuring the workout panel."""
        center_x = camera_width // 2
        center_y = canvas.shape[0] // 2
        cv2.rectangle(
            canvas,
            (center_x - 100, center_y - 35),
            (center_x + 100, center_y + 35),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            canvas,
            "PAUSED",
            (center_x - 73, center_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_progress(
        canvas: Frame, progress: float | None, panel_left: int
    ) -> None:
        """Draw a clamped curl progress bar when an angle is available."""
        left = panel_left + 16
        right = canvas.shape[1] - 16
        top, bottom = PROGRESS_TOP, PROGRESS_BOTTOM
        cv2.rectangle(canvas, (left, top), (right, bottom), (90, 90, 90), 1)
        if progress is None:
            return
        fill = left + round((right - left) * max(0.0, min(1.0, progress)))
        cv2.rectangle(canvas, (left, top), (fill, bottom), (80, 210, 120), -1)


def draw_arm(canvas: Frame, landmarks: ArmLandmarks | None) -> None:
    """Draw available selected-arm joints for any local camera workflow."""
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


def overlay_lines(data: OverlayData) -> tuple[str, ...]:
    """Format stable, testable text for a workout overlay."""
    angle = "--" if data.angle is None else f"{data.angle:.1f} deg"
    return (
        f"Arm: {data.arm.value.upper()}",
        f"Reps: {data.repetitions}",
        f"Angle: {angle}",
        f"Stage: {data.stage.value.upper()}",
        f"FPS: {data.fps:.1f}",
        f"Thresholds: {'PERSONALIZED' if data.personalized else 'DEFAULT'}",
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
    if not 0.0 <= up_threshold < down_threshold <= 180.0:
        raise ValueError("thresholds must satisfy 0 <= up < down <= 180")
    if angle is None:
        return None
    progress = (down_threshold - angle) / (down_threshold - up_threshold)
    return max(0.0, min(1.0, progress))
