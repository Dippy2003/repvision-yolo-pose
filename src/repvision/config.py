"""Typed runtime configuration for RepVision."""

from enum import Enum


class Arm(str, Enum):
    """Arm selected for pose tracking."""

    LEFT = "left"
    RIGHT = "right"
