import pytest

from repvision.calibration import GuidedCalibration
from repvision.calibration_workflow import calibration_angle, calibration_overlay
from repvision.config import AppConfig, Arm
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
