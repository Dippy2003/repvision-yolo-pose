from repvision.config import AppConfig, Arm
from repvision.form_checker import FeedbackMessage
from repvision.pose_detector import PoseObservation, PoseStatus
from repvision.rep_counter import MovementStage
from repvision.workout import WorkoutEngine, WorkoutState


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
