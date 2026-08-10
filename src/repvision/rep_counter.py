"""Confirmed-frame bicep-curl movement state and repetition counting."""

from enum import StrEnum


class MovementStage(StrEnum):
    """Accepted position of the selected arm."""

    UNKNOWN = "unknown"
    DOWN = "down"
    UP = "up"
