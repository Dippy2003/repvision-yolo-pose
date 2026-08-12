"""State and orchestration for a live local workout."""

import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic

from repvision.camera import Camera, Frame
from repvision.config import AppConfig, Arm
from repvision.controls import KeyAction
from repvision.display import DisplayError, OpenCVDisplay
from repvision.form_checker import FeedbackMessage, FormChecker, FormFeedback
from repvision.pose_detector import PoseDetector, PoseObservation, PoseStatus
from repvision.renderer import OverlayData, Renderer, curl_progress
from repvision.rep_counter import CurlTracker, CurlUpdate, MovementStage
from repvision.session import SessionAccumulator, SessionLogger
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
        landmarks = observation.selected_arm
        status = observation.status
        if landmarks is not None and landmarks.arm is not self.state.arm:
            landmarks = None
            status = PoseStatus.MISSING_KEYPOINTS
        update = self.tracker.update(landmarks, timestamp=timestamp)
        feedback = self.form_checker.check(
            status,
            landmarks,
            update.smoothed_angle,
            update.stage,
        )
        progress = curl_progress(
            update.smoothed_angle,
            self.config.up_angle_threshold,
            self.config.down_angle_threshold,
        )
        fps = self.fps_meter.update(timestamp)
        return FrameAnalysis(update, feedback, progress, fps)

    def reset_measurements(self) -> None:
        """Clear tracking and timing measurements while preserving controls."""
        self.tracker.reset()
        self.fps_meter.reset()

    def switch_arm(self) -> None:
        """Switch the arm and clear measurements that cannot span arms."""
        self.state.switch_arm()
        self.reset_measurements()

    def toggle_pause(self) -> None:
        """Pause or resume without including idle time in the FPS estimate."""
        self.state.toggle_pause()
        self.fps_meter.reset()


def _cleared_analysis(_previous: FrameAnalysis) -> FrameAnalysis:
    return FrameAnalysis(
        CurlUpdate(None, None, 0, MovementStage.UNKNOWN),
        FormFeedback(FeedbackMessage.GOOD_MOVEMENT),
        None,
        0.0,
    )


def run_workout(
    config: AppConfig,
    *,
    camera_factory: Callable[[int], Camera] = Camera,
    detector_factory: Callable[[AppConfig], PoseDetector] = PoseDetector,
    display_factory: Callable[[], OpenCVDisplay] = OpenCVDisplay,
    renderer_factory: Callable[[], Renderer] = Renderer,
    clock: Callable[[], float] = monotonic,
    wall_clock: Callable[[], datetime] = datetime.now,
) -> Path:
    """Run the local camera loop until Q and save one aggregate summary."""
    detector = detector_factory(config)
    display = display_factory()
    renderer = renderer_factory()
    engine = WorkoutEngine(config)
    started_at = wall_clock()
    started_monotonic = clock()
    accumulator = SessionAccumulator(started_at, started_monotonic)
    latest_frame: Frame | None = None
    latest_observation: PoseObservation | None = None
    latest_analysis: FrameAnalysis | None = None

    try:
        with camera_factory(config.camera_index) as camera:
            while True:
                if not engine.state.paused:
                    latest_frame = camera.read()
                    latest_observation = detector.detect(latest_frame, engine.state.arm)
                    latest_analysis = engine.process(latest_observation, clock())
                    accumulator.record(
                        latest_analysis.update,
                        latest_analysis.feedback,
                    )
                assert latest_frame is not None
                assert latest_observation is not None
                assert latest_analysis is not None
                landmarks = latest_observation.selected_arm
                if landmarks is not None and landmarks.arm is not engine.state.arm:
                    landmarks = None
                rendered = renderer.render(
                    latest_frame,
                    latest_analysis.overlay(engine.state),
                    landmarks,
                )
                display.show(rendered)
                action = display.read_action()
                if action is KeyAction.QUIT:
                    break
                if action is KeyAction.TOGGLE_PAUSE:
                    engine.toggle_pause()
                elif action is KeyAction.RESET:
                    engine.reset_measurements()
                    accumulator.reset(wall_clock(), clock())
                    latest_analysis = _cleared_analysis(latest_analysis)
                elif action is KeyAction.SWITCH_ARM:
                    engine.switch_arm()
                    accumulator.reset(wall_clock(), clock())
                    latest_analysis = _cleared_analysis(latest_analysis)
    finally:
        if sys.exc_info()[0] is None:
            display.close()
        else:
            with suppress(DisplayError):
                display.close()

    summary = accumulator.summary(engine.state.arm, clock())
    return SessionLogger(config.output_directory).save(summary)
