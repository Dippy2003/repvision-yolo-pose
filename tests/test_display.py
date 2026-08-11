from unittest.mock import patch

import cv2
import numpy as np
import pytest

from repvision.controls import KeyAction
from repvision.display import DisplayError, OpenCVDisplay


def test_display_shows_frame_in_named_window() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    with patch("repvision.display.cv2.imshow") as imshow:
        OpenCVDisplay("Workout").show(frame)

    imshow.assert_called_once_with("Workout", frame)


def test_display_decodes_waited_key() -> None:
    with patch("repvision.display.cv2.waitKey", return_value=ord("q")) as wait_key:
        action = OpenCVDisplay().read_action(5)

    assert action is KeyAction.QUIT
    wait_key.assert_called_once_with(5)


def test_display_wraps_opencv_window_error() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    error = cv2.error("OpenCV", "imshow", "unavailable", "display.cpp", 1)

    with (
        patch("repvision.display.cv2.imshow", side_effect=error),
        pytest.raises(DisplayError, match="Could not display"),
    ):
        OpenCVDisplay().show(frame)


def test_display_closes_opencv_windows() -> None:
    with patch("repvision.display.cv2.destroyAllWindows") as destroy:
        OpenCVDisplay().close()

    destroy.assert_called_once_with()
