import numpy as np
import pytest

from repvision.angles import _coordinates, calculate_elbow_angle


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((1, 2), (1.0, 2.0)),
        ([1.5, 2.5], (1.5, 2.5)),
        (np.asarray([3, 4]), (3.0, 4.0)),
    ],
)
def test_coordinates_accept_numeric_2d_inputs(
    point: object, expected: tuple[float, float]
) -> None:
    assert _coordinates(point) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("point", [(), (1,), (1, 2, 3), ("x", 2), None])
def test_coordinates_reject_invalid_inputs(point: object) -> None:
    with pytest.raises(ValueError, match="exactly two numeric"):
        _coordinates(point)  # type: ignore[arg-type]


def test_straight_arm_angle_is_180_degrees() -> None:
    angle = calculate_elbow_angle((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))

    assert angle == pytest.approx(180.0)
