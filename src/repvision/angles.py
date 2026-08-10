"""Pure elbow-angle calculation and robust temporal smoothing."""

from collections.abc import Sequence
from math import acos, degrees, hypot

Point2DLike = Sequence[float]


def _coordinates(point: Point2DLike) -> tuple[float, float]:
    try:
        values = tuple(point)
    except TypeError as error:
        raise ValueError("point must contain exactly two numeric coordinates") from error
    if len(values) != 2:
        raise ValueError("point must contain exactly two numeric coordinates")
    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError) as error:
        raise ValueError("point must contain exactly two numeric coordinates") from error


def _clamp_cosine(value: float) -> float:
    return max(-1.0, min(1.0, value))


def calculate_elbow_angle(
    shoulder: Point2DLike,
    elbow: Point2DLike,
    wrist: Point2DLike,
) -> float | None:
    """Calculate the shoulder-elbow-wrist angle in degrees."""
    shoulder_x, shoulder_y = _coordinates(shoulder)
    elbow_x, elbow_y = _coordinates(elbow)
    wrist_x, wrist_y = _coordinates(wrist)

    upper_arm = (shoulder_x - elbow_x, shoulder_y - elbow_y)
    forearm = (wrist_x - elbow_x, wrist_y - elbow_y)
    upper_length = hypot(*upper_arm)
    forearm_length = hypot(*forearm)
    if upper_length == 0.0 or forearm_length == 0.0:
        return None

    cosine = (
        upper_arm[0] * forearm[0] + upper_arm[1] * forearm[1]
    ) / (upper_length * forearm_length)
    return degrees(acos(_clamp_cosine(cosine)))
