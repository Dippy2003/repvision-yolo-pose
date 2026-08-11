"""Keyboard actions for the local workout window."""

from enum import StrEnum


class KeyAction(StrEnum):
    """Application actions produced from one OpenCV key code."""

    NONE = "none"
    QUIT = "quit"
    RESET = "reset"
    TOGGLE_PAUSE = "toggle_pause"
    SWITCH_ARM = "switch_arm"


def decode_key(key_code: int) -> KeyAction:
    """Translate a platform-dependent key code into an application action."""
    if key_code < 0:
        return KeyAction.NONE
    key = key_code & 0xFF
    if key in (ord("q"), ord("Q")):
        return KeyAction.QUIT
    if key in (ord("r"), ord("R")):
        return KeyAction.RESET
    if key in (ord("p"), ord("P")):
        return KeyAction.TOGGLE_PAUSE
    if key in (ord("l"), ord("L")):
        return KeyAction.SWITCH_ARM
    return KeyAction.NONE
