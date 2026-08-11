"""State and orchestration for a live local workout."""

from dataclasses import dataclass

from repvision.config import Arm


@dataclass(slots=True)
class WorkoutState:
    """Mutable user-controlled state for the workout window."""

    arm: Arm
    paused: bool = False

    def toggle_pause(self) -> None:
        """Pause or resume frame processing."""
        self.paused = not self.paused

    def switch_arm(self) -> None:
        """Switch between the supported left and right arms."""
        self.arm = Arm.LEFT if self.arm is Arm.RIGHT else Arm.RIGHT
