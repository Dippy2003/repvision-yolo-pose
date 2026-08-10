"""Pure elbow-angle calculation and robust temporal smoothing."""

from collections.abc import Sequence

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
