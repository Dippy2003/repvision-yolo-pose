"""State and orchestration for a live local workout."""

from dataclasses import dataclass

from repvision.config import AppConfig, Arm
from repvision.form_checker import FormChecker, FormFeedback
from repvision.pose_detector import PoseObservation
from repvision.renderer import OverlayData, curl_progress
from repvision.rep_counter import CurlTracker, CurlUpdate
from repvision.timing import FpsMeter


@dataclass(slots=True)
class WorkoutState:
    """Mutable user-controlled state for the workout window."""

    arm: Arm
    paused: bool = False

    def toggle_pause(self) -> None:
        """Pause or resume frame processing."""
        self.paused = not self.paused

    def switch_arm(self) -> None:
        """Switch between the supported left and right arms."""
        self.arm = Arm.LEFT if self.arm is Arm.RIGHT else Arm.RIGHT


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    """Derived state produced from one pose observation."""

    update: CurlUpdate
    feedback: FormFeedback
    progress: float | None
    fps: float

    def overlay(self, state: WorkoutState) -> OverlayData:
        """Build renderer input with the latest user-controlled state."""
        return OverlayData(
            state.arm,
            self.update.count,
            self.update.smoothed_angle,
            self.update.stage,
            self.feedback,
            self.progress,
            self.fps,
            state.paused,
        )


class WorkoutEngine:
    """Process pose observations independently from camera and display access."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = WorkoutState(config.selected_arm)
        self.tracker = CurlTracker(config)
        self.form_checker = FormChecker(config)
        self.fps_meter = FpsMeter()

    def process(self, observation: PoseObservation, timestamp: float) -> FrameAnalysis:
        """Update movement, feedback, progress, and FPS for one frame."""
        update = self.tracker.update(observation.selected_arm, timestamp=timestamp)
        feedback = self.form_checker.check(
            observation.status,
            observation.selected_arm,
            update.smoothed_angle,
            update.stage,
        )
        progress = curl_progress(
            update.smoothed_angle,
            self.config.up_angle_threshold,
            self.config.down_angle_threshold,
        )
        return FrameAnalysis(update, feedback, progress, self.fps_meter.update(timestamp))

    def reset_measurements(self) -> None:
        """Clear tracking and timing measurements while preserving controls."""
        self.tracker.reset()
        self.fps_meter.reset()

    def switch_arm(self) -> None:
        """Switch the arm and clear measurements that cannot span arms."""
        self.state.switch_arm()
        self.reset_measurements()
