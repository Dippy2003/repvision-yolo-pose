"""OpenCV workout-window boundary."""

import cv2

from repvision.camera import Frame
from repvision.controls import KeyAction, decode_key


class DisplayError(RuntimeError):
    """Raised when OpenCV cannot create or update the workout window."""


class OpenCVDisplay:
    """Show workout frames and expose decoded keyboard actions."""

    def __init__(self, window_name: str = "RepVision") -> None:
        self.window_name = window_name

    def show(self, frame: Frame) -> None:
        """Display one processed frame in the local desktop window."""
        try:
            cv2.imshow(self.window_name, frame)
        except cv2.error as error:
            message = f"Could not display the workout window: {error}"
            raise DisplayError(message) from error

    def read_action(self, delay_ms: int = 1) -> KeyAction:
        """Wait briefly for a key and return its application action."""
        if delay_ms < 1:
            raise ValueError("delay_ms must be positive")
        try:
            action = decode_key(cv2.waitKey(delay_ms))
            if action is not KeyAction.NONE:
                return action
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            return KeyAction.NONE if visible >= 1.0 else KeyAction.QUIT
        except cv2.error as error:
            raise DisplayError(f"Could not read window input: {error}") from error

    def close(self) -> None:
        """Close all windows owned by OpenCV."""
        try:
            cv2.destroyAllWindows()
        except cv2.error as error:
            raise DisplayError(f"Could not close workout windows: {error}") from error
