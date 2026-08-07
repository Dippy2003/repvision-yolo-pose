from repvision.config import Arm


def test_arm_values_are_command_line_friendly() -> None:
    assert Arm.LEFT.value == "left"
    assert Arm.RIGHT.value == "right"
