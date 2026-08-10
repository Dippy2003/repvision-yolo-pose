"""Pure elbow-angle calculation and robust temporal smoothing."""

from collections import deque
from collections.abc import Sequence
from math import acos, degrees, hypot, isfinite
from statistics import median

Point2DLike = Sequence[float]


class AngleSmoother:
    """Maintain a bounded history of valid elbow angles."""

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._history: deque[float] = deque(maxlen=window_size)

    @property
    def sample_count(self) -> int:
        """Return the number of valid angles currently retained."""
        return len(self._history)

    @property
    def value(self) -> float | None:
        """Return the current median without adding a measurement."""
        if not self._history:
            return None
        return float(median(self._history))

    def add(self, angle: float | None) -> float | None:
        """Add one valid angle; missing/non-finite measurements return None."""
        if angle is None or not isfinite(angle):
            return None
        if not 0.0 <= angle <= 180.0:
            raise ValueError("angle must be between 0 and 180 degrees")
        self._history.append(float(angle))
        smoothed = self.value
        assert smoothed is not None
        return smoothed


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
    if not all(
        isfinite(value)
        for value in (
            shoulder_x,
            shoulder_y,
            elbow_x,
            elbow_y,
            wrist_x,
            wrist_y,
        )
    ):
        return None

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
