from repvision.calibration import CalibrationStage
from repvision.calibration_renderer import CalibrationOverlay, calibration_lines
from repvision.config import Arm
from repvision.pose_detector import PoseStatus


def test_calibration_lines_prompt_for_extended_capture() -> None:
    data = CalibrationOverlay(
        Arm.RIGHT,
        CalibrationStage.READY_EXTENDED,
        162.25,
        0,
        20,
        PoseStatus.TRACKING,
    )

    assert calibration_lines(data) == (
        "Arm: RIGHT",
        "Instruction: Extend arm, then press SPACE",
        "Angle: 162.2 deg",
        "Samples: 0/20",
        "Pose: tracking",
    )


def test_calibration_lines_show_unavailable_angle() -> None:
    data = CalibrationOverlay(
        Arm.LEFT,
        CalibrationStage.CAPTURING_CURLED,
        None,
        3,
        20,
        PoseStatus.LOW_CONFIDENCE,
    )

    lines = calibration_lines(data)

    assert "Instruction: Hold arm fully curled" in lines
    assert "Angle: unavailable" in lines
    assert "Pose: low_confidence" in lines
