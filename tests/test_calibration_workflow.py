from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from repvision.calibration import CalibrationStore, GuidedCalibration
from repvision.calibration_workflow import (
    CalibrationCancelled,
    calibration_angle,
    calibration_overlay,
    run_guided_calibration,
)
from repvision.camera import Frame
from repvision.config import AppConfig, Arm
from repvision.controls import KeyAction
from repvision.pose_detector import (
    ArmLandmarks,
    Landmark,
    Point2D,
    PoseObservation,
    PoseStatus,
)


def arm_landmarks() -> ArmLandmarks:
    return ArmLandmarks(
        Arm.RIGHT,
        Landmark(Point2D(1.0, 0.0), 0.9),
        Landmark(Point2D(0.0, 0.0), 0.9),
        Landmark(Point2D(0.0, 1.0), 0.9),
        Landmark(Point2D(0.0, 2.0), 0.9),
    )


def test_calibration_angle_uses_reliable_tracking_landmarks() -> None:
    observation = PoseObservation((), None, arm_landmarks(), PoseStatus.TRACKING)

    assert calibration_angle(observation, AppConfig()) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "status",
    [PoseStatus.NO_PERSON, PoseStatus.MISSING_KEYPOINTS, PoseStatus.LOW_CONFIDENCE],
)
def test_calibration_angle_ignores_unreliable_pose(status: PoseStatus) -> None:
    observation = PoseObservation((), None, arm_landmarks(), status)

    assert calibration_angle(observation, AppConfig()) is None


def test_calibration_overlay_reports_current_capture_progress() -> None:
    config = AppConfig(calibration_sample_target=3)
    workflow = GuidedCalibration(config, Arm.RIGHT)
    workflow.begin_capture()
    workflow.record(160.0)
    observation = PoseObservation((), None, arm_landmarks(), PoseStatus.TRACKING)

    result = calibration_overlay(workflow, observation, 160.0)

    assert result.arm is Arm.RIGHT
    assert result.stage is workflow.stage
    assert result.sample_count == 1
    assert result.sample_target == 3
    assert result.pose_status is PoseStatus.TRACKING


class FakeCamera:
    def __init__(self, _index: int) -> None:
        self.released = False

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *_args: object) -> None:
        self.released = True

    def read(self) -> Frame:
        return np.zeros((240, 320, 3), dtype=np.uint8)


class FakeDetector:
    def __init__(self, _config: AppConfig) -> None:
        pass

    def detect(self, _frame: Frame, _arm: Arm) -> PoseObservation:
        return PoseObservation((), None, arm_landmarks(), PoseStatus.TRACKING)


class FakeRenderer:
    def render(self, frame: Frame, *_args: object) -> Frame:
        return frame.copy()


class FakeDisplay:
    def __init__(self, actions: list[KeyAction]) -> None:
        self.actions = iter(actions)
        self.closed = False
        self.frames: list[Frame] = []

    def show(self, frame: Frame) -> None:
        self.frames.append(frame)

    def read_action(self) -> KeyAction:
        return next(self.actions)

    def close(self) -> None:
        self.closed = True


def test_guided_camera_calibration_saves_completed_profile(tmp_path: Path) -> None:
    camera = FakeCamera(0)
    display = FakeDisplay(
        [
            KeyAction.CONFIRM,
            KeyAction.NONE,
            KeyAction.NONE,
            KeyAction.CONFIRM,
            KeyAction.NONE,
            KeyAction.NONE,
        ]
    )
    angles = iter([None, 160.0, 165.0, 170.0, 30.0, 35.0, 40.0])
    store = CalibrationStore(tmp_path / "calibration.json")

    with patch(
        "repvision.calibration_workflow.calibration_angle",
        side_effect=lambda *_args: next(angles),
    ):
        profile = run_guided_calibration(
            AppConfig(calibration_sample_target=3),
            store=store,
            camera_factory=lambda _index: camera,
            detector_factory=FakeDetector,
            display_factory=lambda: display,
            renderer_factory=FakeRenderer,
        )

    assert camera.released
    assert display.closed
    assert len(display.frames) == 7
    assert profile.extended_angle == 165.0
    assert profile.curled_angle == 35.0
    assert store.load(Arm.RIGHT) == profile


def test_guided_camera_calibration_cancels_without_profile(tmp_path: Path) -> None:
    camera = FakeCamera(0)
    display = FakeDisplay([KeyAction.QUIT])
    store = CalibrationStore(tmp_path / "calibration.json")

    with pytest.raises(CalibrationCancelled, match="no profile saved"):
        run_guided_calibration(
            AppConfig(calibration_sample_target=3),
            store=store,
            camera_factory=lambda _index: camera,
            detector_factory=FakeDetector,
            display_factory=lambda: display,
            renderer_factory=FakeRenderer,
        )

    assert camera.released
    assert display.closed
    assert not store.path.exists()


def test_guided_camera_calibration_restart_discards_prior_samples(
    tmp_path: Path,
) -> None:
    display = FakeDisplay(
        [
            KeyAction.CONFIRM,
            KeyAction.RESET,
            KeyAction.CONFIRM,
            KeyAction.NONE,
            KeyAction.NONE,
            KeyAction.CONFIRM,
            KeyAction.NONE,
            KeyAction.NONE,
        ]
    )
    angles = iter([None, 175.0, None, 161.0, 162.0, 163.0, 30.0, 35.0, 40.0])

    with patch(
        "repvision.calibration_workflow.calibration_angle",
        side_effect=lambda *_args: next(angles),
    ):
        result = run_guided_calibration(
            AppConfig(calibration_sample_target=3),
            store=CalibrationStore(tmp_path / "calibration.json"),
            camera_factory=FakeCamera,
            detector_factory=FakeDetector,
            display_factory=lambda: display,
            renderer_factory=FakeRenderer,
        )

    assert result.extended_angle == 162.0
