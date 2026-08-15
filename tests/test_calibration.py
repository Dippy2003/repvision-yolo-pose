import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repvision.calibration import (
    CalibrationCollector,
    CalibrationError,
    CalibrationPosition,
    CalibrationProfile,
    CalibrationRangeError,
    CalibrationStorageError,
    CalibrationStore,
    default_calibration_path,
    profile_from_dict,
    profile_to_dict,
)
from repvision.config import AppConfig, Arm


def test_calibration_positions_have_stable_values() -> None:
    assert CalibrationPosition.EXTENDED.value == "extended"
    assert CalibrationPosition.CURLED.value == "curled"


def test_calibration_failures_share_public_base() -> None:
    assert isinstance(CalibrationRangeError("range"), CalibrationError)
    assert isinstance(CalibrationStorageError("storage"), CalibrationError)


def profile() -> CalibrationProfile:
    return CalibrationProfile(
        Arm.RIGHT,
        curled_angle=42.0,
        extended_angle=164.0,
        up_threshold=52.0,
        down_threshold=154.0,
        samples_per_position=20,
        calibrated_at=datetime(2026, 8, 15, 9, 30, tzinfo=UTC),
    )


def test_calibration_profile_keeps_personalized_range() -> None:
    result = profile()

    assert result.arm is Arm.RIGHT
    assert result.movement_range == 122.0
    assert result.samples_per_position == 20


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("curled_angle", -1.0),
        ("extended_angle", 181.0),
        ("up_threshold", float("nan")),
        ("down_threshold", float("inf")),
    ],
)
def test_calibration_profile_rejects_invalid_angles(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite and between"):
        replace(profile(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("curled_angle", 60.0),
        ("up_threshold", 154.0),
        ("down_threshold", 52.0),
        ("extended_angle", 154.0),
    ],
)
def test_calibration_profile_requires_ordered_endpoints(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="curled < up < down < extended"):
        replace(profile(), **{field: value})


def test_calibration_profile_requires_robust_sample_count() -> None:
    with pytest.raises(ValueError, match="samples_per_position"):
        replace(profile(), samples_per_position=2)


def test_calibration_profile_requires_timezone() -> None:
    with pytest.raises(ValueError, match="include a timezone"):
        replace(profile(), calibrated_at=datetime(2026, 8, 15, 9, 30))


def test_calibration_collector_retains_valid_endpoint_angles() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.LEFT)

    assert collector.add(CalibrationPosition.EXTENDED, 160.0) == 1
    assert collector.add(CalibrationPosition.EXTENDED, 162.0) == 2
    assert collector.sample_count(CalibrationPosition.EXTENDED) == 2
    assert collector.sample_count(CalibrationPosition.CURLED) == 0


@pytest.mark.parametrize("angle", [None, float("nan"), float("inf")])
def test_calibration_collector_ignores_missing_measurements(
    angle: float | None,
) -> None:
    collector = CalibrationCollector(AppConfig(), Arm.RIGHT)

    assert collector.add(CalibrationPosition.CURLED, angle) == 0
    assert collector.sample_count(CalibrationPosition.CURLED) == 0


@pytest.mark.parametrize("angle", [-0.1, 180.1])
def test_calibration_collector_rejects_out_of_range_angle(angle: float) -> None:
    collector = CalibrationCollector(AppConfig(), Arm.RIGHT)

    with pytest.raises(ValueError, match="between 0 and 180"):
        collector.add(CalibrationPosition.EXTENDED, angle)


def test_calibration_collector_bounds_retained_history() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.RIGHT)

    for angle in (160.0, 161.0, 162.0, 170.0):
        count = collector.add(CalibrationPosition.EXTENDED, angle)

    assert count == 3
    assert collector.sample_count(CalibrationPosition.EXTENDED) == 3


def test_calibration_collector_reports_endpoint_readiness() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.RIGHT)
    for angle in (160.0, 161.0, 162.0):
        collector.add(CalibrationPosition.EXTENDED, angle)

    assert collector.position_ready(CalibrationPosition.EXTENDED)
    assert not collector.position_ready(CalibrationPosition.CURLED)
    assert not collector.complete

    for angle in (40.0, 41.0, 42.0):
        collector.add(CalibrationPosition.CURLED, angle)

    assert collector.complete


def test_calibration_profile_uses_endpoint_medians_and_margin() -> None:
    collector = CalibrationCollector(
        AppConfig(
            calibration_sample_target=3,
            calibration_threshold_margin=10.0,
        ),
        Arm.LEFT,
    )
    for angle in (160.0, 164.0, 170.0):
        collector.add(CalibrationPosition.EXTENDED, angle)
    for angle in (30.0, 42.0, 50.0):
        collector.add(CalibrationPosition.CURLED, angle)
    timestamp = datetime(2026, 8, 15, 10, tzinfo=UTC)

    result = collector.build_profile(timestamp)

    assert result.arm is Arm.LEFT
    assert result.extended_angle == 164.0
    assert result.curled_angle == 42.0
    assert result.up_threshold == 52.0
    assert result.down_threshold == 154.0
    assert result.calibrated_at == timestamp


def test_calibration_profile_requires_both_positions() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.RIGHT)
    for angle in (160.0, 161.0, 162.0):
        collector.add(CalibrationPosition.EXTENDED, angle)

    with pytest.raises(CalibrationError, match="extended and curled"):
        collector.build_profile()


def test_calibration_profile_rejects_insufficient_movement_range() -> None:
    collector = CalibrationCollector(
        AppConfig(calibration_sample_target=3, calibration_minimum_range=60.0),
        Arm.RIGHT,
    )
    for angle in (100.0, 101.0, 102.0):
        collector.add(CalibrationPosition.EXTENDED, angle)
    for angle in (50.0, 51.0, 52.0):
        collector.add(CalibrationPosition.CURLED, angle)

    with pytest.raises(CalibrationRangeError, match=r"too small.*minimum 60\.0"):
        collector.build_profile()


def test_calibration_profile_default_timestamp_is_timezone_aware() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.RIGHT)
    for angle in (160.0, 161.0, 162.0):
        collector.add(CalibrationPosition.EXTENDED, angle)
    for angle in (40.0, 41.0, 42.0):
        collector.add(CalibrationPosition.CURLED, angle)

    result = collector.build_profile()

    assert result.calibrated_at.utcoffset() is not None


def test_calibration_collector_resets_one_or_both_positions() -> None:
    collector = CalibrationCollector(AppConfig(calibration_sample_target=3), Arm.RIGHT)
    collector.add(CalibrationPosition.EXTENDED, 160.0)
    collector.add(CalibrationPosition.CURLED, 40.0)

    collector.reset(CalibrationPosition.EXTENDED)

    assert collector.sample_count(CalibrationPosition.EXTENDED) == 0
    assert collector.sample_count(CalibrationPosition.CURLED) == 1

    collector.reset()

    assert collector.sample_count(CalibrationPosition.CURLED) == 0


def test_profile_serialization_contains_only_aggregate_calibration() -> None:
    data = profile_to_dict(profile())

    assert data == {
        "arm": "right",
        "curled_angle": 42.0,
        "extended_angle": 164.0,
        "up_threshold": 52.0,
        "down_threshold": 154.0,
        "samples_per_position": 20,
        "calibrated_at": "2026-08-15T09:30:00+00:00",
    }
    assert "frame" not in data
    assert "keypoints" not in data


def test_profile_serialization_round_trip() -> None:
    original = profile()

    restored = profile_from_dict(profile_to_dict(original))

    assert restored == original


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {**profile_to_dict(profile()), "unexpected": True},
        {**profile_to_dict(profile()), "curled_angle": "forty"},
        {**profile_to_dict(profile()), "samples_per_position": True},
        {**profile_to_dict(profile()), "calibrated_at": "not-a-date"},
    ],
)
def test_profile_parser_rejects_malformed_data(data: object) -> None:
    with pytest.raises(CalibrationStorageError, match="profile"):
        profile_from_dict(data)


def test_default_calibration_path_uses_windows_local_app_data() -> None:
    path = default_calibration_path(
        {"LOCALAPPDATA": r"C:\Users\Test\AppData\Local"}
    )

    assert path == Path(r"C:\Users\Test\AppData\Local\RepVision\calibration.json")


def test_default_calibration_path_uses_cross_platform_config_home() -> None:
    assert default_calibration_path(
        {"XDG_CONFIG_HOME": "/config"}, home=Path("/home/test")
    ) == Path("/config/repvision/calibration.json")
    assert default_calibration_path({}, home=Path("/home/test")) == Path(
        "/home/test/.config/repvision/calibration.json"
    )


def test_calibration_store_returns_empty_before_first_save(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path / "settings" / "calibration.json")

    assert store.load_all() == {}


def test_calibration_store_loads_versioned_arm_profiles(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "arms": {"right": profile_to_dict(profile())},
            }
        ),
        encoding="utf-8",
    )

    loaded = CalibrationStore(path).load_all()

    assert loaded == {Arm.RIGHT: profile()}


def test_calibration_store_explains_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(CalibrationStorageError, match="Could not read"):
        CalibrationStore(path).load_all()


@pytest.mark.parametrize(
    "document",
    [[], {}, {"version": 1}, {"version": 1, "arms": [], "extra": True}],
)
def test_calibration_store_rejects_invalid_document_structure(
    tmp_path: Path, document: object
) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CalibrationStorageError, match="structure"):
        CalibrationStore(path).load_all()


def test_calibration_store_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": 2, "arms": {}}), encoding="utf-8")

    with pytest.raises(CalibrationStorageError, match="Unsupported.*2"):
        CalibrationStore(path).load_all()


def test_calibration_store_requires_arm_mapping(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": 1, "arms": []}), encoding="utf-8")

    with pytest.raises(CalibrationStorageError, match="arms must be an object"):
        CalibrationStore(path).load_all()
