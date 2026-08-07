"""Typed runtime configuration for RepVision."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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
    up_angle_threshold: float = 50.0
    down_angle_threshold: float = 155.0
    confirmation_frames: int = 3
    smoothing_window: int = 5
    cooldown_seconds: float = 0.3
    input_size: int = 640
    output_directory: Path = Path("outputs")

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.camera_index < 0:
            raise ValueError("camera_index must be zero or greater")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.up_angle_threshold < self.down_angle_threshold <= 180.0:
            raise ValueError(
                "angle thresholds must satisfy 0 <= up < down <= 180"
            )
