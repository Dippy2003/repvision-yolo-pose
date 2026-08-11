from datetime import datetime

import numpy as np

from repvision.camera import Frame
from repvision.config import AppConfig, Arm
from repvision.controls import KeyAction
from repvision.form_checker import FeedbackMessage
from repvision.pose_detector import PoseObservation, PoseStatus
from repvision.rep_counter import MovementStage
from repvision.workout import WorkoutEngine, WorkoutState, run_workout


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


class FakeCamera:
    def __init__(self, _index: int) -> None:
        self.released = False

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *args: object) -> None:
        self.released = True

    def read(self) -> Frame:
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
