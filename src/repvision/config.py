"""Typed runtime configuration for RepVision."""

from dataclasses import dataclass
from enum import Enum


class Arm(str, Enum):
    """Arm selected for pose tracking."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated settings shared by application components."""

    model_name: str = "yolo26n-pose.pt"
    camera_index: int = 0
    selected_arm: Arm = Arm.RIGHT
    confidence_threshold: float = 0.5
