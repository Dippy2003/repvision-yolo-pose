import pytest

from repvision.controls import KeyAction, decode_key


@pytest.mark.parametrize("key_code", [ord("q"), ord("Q"), 0x100 + ord("q")])
def test_decode_key_recognizes_quit(key_code: int) -> None:
    assert decode_key(key_code) is KeyAction.QUIT


@pytest.mark.parametrize("key_code", [-1, ord("x"), 255])
def test_decode_key_ignores_unmapped_input(key_code: int) -> None:
    assert decode_key(key_code) is KeyAction.NONE


@pytest.mark.parametrize("key_code", [ord("r"), ord("R")])
def test_decode_key_recognizes_reset(key_code: int) -> None:
    assert decode_key(key_code) is KeyAction.RESET


@pytest.mark.parametrize("key_code", [ord("p"), ord("P")])
def test_decode_key_recognizes_pause_toggle(key_code: int) -> None:
    assert decode_key(key_code) is KeyAction.TOGGLE_PAUSE


@pytest.mark.parametrize("key_code", [ord("l"), ord("L")])
def test_decode_key_recognizes_arm_switch(key_code: int) -> None:
    assert decode_key(key_code) is KeyAction.SWITCH_ARM


@pytest.mark.parametrize("key_code", [ord(" "), 13])
def test_decode_key_accepts_calibration_confirmation(key_code: int) -> None:
    assert decode_key(key_code) is KeyAction.CONFIRM
