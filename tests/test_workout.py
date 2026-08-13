from datetime import datetime

import numpy as np
import pytest

from repvision.camera import Frame
from repvision.config import AppConfig, Arm
from repvision.controls import KeyAction
from repvision.display import DisplayError
from repvision.form_checker import FeedbackMessage, FormFeedback
from repvision.frame_source import EndOfStream
from repvision.pose_detector import ArmLandmarks, Landmark, PoseObservation, PoseStatus
from repvision.rep_counter import CurlUpdate, MovementStage
from repvision.workout import (
    FrameAnalysis,
    WorkoutEngine,
    WorkoutState,
    _cleared_analysis,
    run_workout,
)


def test_workout_state_toggles_pause() -> None:
    state = WorkoutState(Arm.RIGHT)

    state.toggle_pause()
    assert state.paused
    state.toggle_pause()
    assert not state.paused


def test_workout_state_switches_arm_both_directions() -> None:
    state = WorkoutState(Arm.RIGHT)

    state.switch_arm()
    assert state.arm is Arm.LEFT
    state.switch_arm()
    assert state.arm is Arm.RIGHT


def test_workout_engine_processes_missing_pose_without_camera_access() -> None:
    engine = WorkoutEngine(AppConfig())
    observation = PoseObservation((), None, None, PoseStatus.NO_PERSON)

    analysis = engine.process(observation, 10.0)

    assert analysis.update.stage is MovementStage.UNKNOWN
    assert analysis.feedback.message is FeedbackMessage.MOVE_BACK
    assert analysis.progress is None
    assert analysis.overlay(engine.state).arm is Arm.RIGHT


def test_workout_engine_switch_resets_arm_specific_measurements() -> None:
    engine = WorkoutEngine(AppConfig())
    observation = PoseObservation((), None, None, PoseStatus.NO_PERSON)
    engine.process(observation, 1.0)
    engine.process(observation, 1.1)
    assert engine.fps_meter.fps > 0.0

    engine.switch_arm()

    assert engine.state.arm is Arm.LEFT
    assert engine.fps_meter.fps == 0.0
    assert engine.tracker.counter.count == 0


def test_workout_engine_pause_resets_fps_baseline() -> None:
    engine = WorkoutEngine(AppConfig())
    observation = PoseObservation((), None, None, PoseStatus.NO_PERSON)
    engine.process(observation, 1.0)
    engine.process(observation, 1.1)
    assert engine.fps_meter.fps > 0.0

    engine.toggle_pause()

    assert engine.state.paused
    assert engine.fps_meter.fps == 0.0


def test_workout_engine_ignores_observation_for_wrong_arm() -> None:
    engine = WorkoutEngine(AppConfig(selected_arm=Arm.RIGHT))
    missing = Landmark(None, 0.0)
    left_arm = ArmLandmarks(Arm.LEFT, missing, missing, missing, missing)
    observation = PoseObservation((), None, left_arm, PoseStatus.TRACKING)

    analysis = engine.process(observation, 1.0)

    assert analysis.update.smoothed_angle is None
    assert analysis.feedback.message is FeedbackMessage.MOVE_BACK


def test_cleared_analysis_removes_stale_warning() -> None:
    previous = FrameAnalysis(
        update=CurlUpdate(None, 90.0, 3, MovementStage.DOWN),
        feedback=FormFeedback(FeedbackMessage.ELBOW_DRIFT, is_form_warning=True),
        progress=0.5,
        fps=20.0,
    )

    cleared = _cleared_analysis(previous)

    assert cleared.update.count == 0
    assert cleared.feedback.message is FeedbackMessage.GOOD_MOVEMENT
    assert cleared.progress is None
    assert cleared.fps == 0.0


class FakeCamera:
    def __init__(self, _index: int) -> None:
        self.released = False
        self.read_count = 0

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *args: object) -> None:
        self.released = True

    def read(self) -> Frame:
        self.read_count += 1
        return np.zeros((300, 500, 3), dtype=np.uint8)


class FakeDetector:
    def __init__(self, _config: AppConfig) -> None:
        self.arms: list[Arm] = []

    def detect(self, _frame: Frame, arm: Arm) -> PoseObservation:
        self.arms.append(arm)
        return PoseObservation((), None, None, PoseStatus.NO_PERSON)


class FakeDisplay:
    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self.closed = False

    def show(self, frame: Frame) -> None:
        self.frames.append(frame)

    def read_action(self, delay_ms: int = 1) -> KeyAction:
        del delay_ms
        return KeyAction.QUIT

    def close(self) -> None:
        self.closed = True


class SequenceDisplay(FakeDisplay):
    def __init__(self, actions: list[KeyAction]) -> None:
        super().__init__()
        self.actions = iter(actions)

    def read_action(self, delay_ms: int = 1) -> KeyAction:
        del delay_ms
        return next(self.actions)


class FiniteSource(FakeCamera):
    @property
    def description(self) -> str:
        return "finite test source"

    def read(self) -> Frame:
        if self.read_count == 1:
            raise EndOfStream("finished")
        return super().read()


def test_run_workout_releases_resources_and_saves_only_aggregates(tmp_path) -> None:
    camera = FakeCamera(0)
    detector = FakeDetector(AppConfig())
    display = FakeDisplay()
    timestamps = iter([10.0, 10.1, 11.0])

    path = run_workout(
        AppConfig(output_directory=tmp_path),
        camera_factory=lambda _index: camera,
        detector_factory=lambda _config: detector,
        display_factory=lambda: display,
        clock=lambda: next(timestamps),
        wall_clock=lambda: datetime(2026, 8, 11, 14),
    )

    assert camera.released
    assert display.closed
    assert len(display.frames) == 1
    assert detector.arms == [Arm.RIGHT]
    assert path.name == "sessions.csv"
    assert [item.name for item in tmp_path.iterdir()] == ["sessions.csv"]


def test_window_cleanup_failure_does_not_hide_processing_error(tmp_path) -> None:
    class FailingDetector(FakeDetector):
        def detect(self, _frame: Frame, arm: Arm) -> PoseObservation:
            del arm
            raise RuntimeError("processing failed")

    class FailingDisplay(FakeDisplay):
        def close(self) -> None:
            raise DisplayError("cleanup failed")

    timestamps = iter([10.0])

    with pytest.raises(RuntimeError, match="processing failed"):
        run_workout(
            AppConfig(output_directory=tmp_path),
            camera_factory=FakeCamera,
            detector_factory=FailingDetector,
            display_factory=FailingDisplay,
            clock=lambda: next(timestamps),
            wall_clock=lambda: datetime(2026, 8, 12),
        )


def test_live_loop_applies_pause_reset_and_arm_switch_controls(tmp_path) -> None:
    camera = FakeCamera(0)
    detector = FakeDetector(AppConfig())
    display = SequenceDisplay(
        [
            KeyAction.TOGGLE_PAUSE,
            KeyAction.TOGGLE_PAUSE,
            KeyAction.RESET,
            KeyAction.SWITCH_ARM,
            KeyAction.QUIT,
        ]
    )
    timestamps = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    path = run_workout(
        AppConfig(output_directory=tmp_path),
        camera_factory=lambda _index: camera,
        detector_factory=lambda _config: detector,
        display_factory=lambda: display,
        clock=lambda: next(timestamps),
        wall_clock=lambda: datetime(2026, 8, 12, 12),
    )

    assert camera.read_count == 4
    assert len(display.frames) == 5
    assert detector.arms == [Arm.RIGHT, Arm.RIGHT, Arm.RIGHT, Arm.LEFT]
    assert ",left,0.20,0,0," in path.read_text(encoding="utf-8")


def test_live_loop_finishes_cleanly_at_end_of_finite_source(tmp_path) -> None:
    source = FiniteSource(0)
    detector = FakeDetector(AppConfig())
    display = SequenceDisplay([KeyAction.NONE])
    timestamps = iter([0.0, 0.1, 0.2])

    path = run_workout(
        AppConfig(output_directory=tmp_path),
        source_factory=lambda: source,
        detector_factory=lambda _config: detector,
        display_factory=lambda: display,
        clock=lambda: next(timestamps),
        wall_clock=lambda: datetime(2026, 8, 13),
    )

    assert source.released
    assert len(display.frames) == 1
    assert path.exists()
