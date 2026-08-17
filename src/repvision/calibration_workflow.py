"""Camera orchestration for private guided arm calibration."""

import sys
from collections.abc import Callable
from contextlib import suppress

from repvision.angles import calculate_arm_angle
from repvision.calibration import (
    CalibrationError,
    CalibrationProfile,
    CalibrationStage,
    CalibrationStore,
    GuidedCalibration,
)
from repvision.calibration_renderer import CalibrationOverlay, CalibrationRenderer
from repvision.camera import Camera
from repvision.config import AppConfig
from repvision.controls import KeyAction
from repvision.display import DisplayError, OpenCVDisplay
from repvision.pose_detector import PoseDetector, PoseObservation, PoseStatus


class CalibrationCancelled(CalibrationError):
    """Raised when the user closes or cancels guided calibration."""


def calibration_angle(
    observation: PoseObservation, config: AppConfig
) -> float | None:
    """Return a reliable raw elbow angle suitable for endpoint sampling."""
    if observation.status is not PoseStatus.TRACKING:
        return None
    return calculate_arm_angle(
        observation.selected_arm,
        config.confidence_threshold,
    )


def calibration_overlay(
    workflow: GuidedCalibration,
    observation: PoseObservation,
    angle: float | None,
) -> CalibrationOverlay:
    """Build display data from the current guided workflow state."""
    return CalibrationOverlay(
        workflow.arm,
        workflow.stage,
        angle,
        workflow.sample_count,
        workflow.collector.config.calibration_sample_target,
        observation.status,
    )


def run_guided_calibration(
    config: AppConfig,
    *,
    store: CalibrationStore | None = None,
    camera_factory: Callable[[int], Camera] = Camera,
    detector_factory: Callable[[AppConfig], PoseDetector] = PoseDetector,
    display_factory: Callable[[], OpenCVDisplay] = OpenCVDisplay,
    renderer_factory: Callable[[], CalibrationRenderer] = CalibrationRenderer,
) -> CalibrationProfile:
    """Capture both arm endpoints locally and persist their aggregate profile."""
    selected_store = CalibrationStore() if store is None else store
    workflow = GuidedCalibration(config, config.selected_arm)
    detector = detector_factory(config)
    display = display_factory()
    renderer = renderer_factory()

    try:
        with camera_factory(config.camera_index) as camera:
            while workflow.stage is not CalibrationStage.COMPLETE:
                frame = camera.read()
                observation = detector.detect(frame, config.selected_arm)
                angle = calibration_angle(observation, config)
                workflow.record(angle)
                rendered = renderer.render(
                    frame,
                    calibration_overlay(workflow, observation, angle),
                    observation.selected_arm,
                )
                display.show(rendered)
                if workflow.stage is CalibrationStage.COMPLETE:
                    break
                action = display.read_action()
                if action is KeyAction.QUIT:
                    raise CalibrationCancelled(
                        "Calibration cancelled; no profile saved."
                    )
                if action is KeyAction.RESET:
                    workflow.reset()
                elif action is KeyAction.CONFIRM:
                    workflow.begin_capture()
    finally:
        if sys.exc_info()[0] is None:
            display.close()
        else:
            with suppress(DisplayError):
                display.close()

    profile = workflow.build_profile()
    selected_store.save(profile)
    return profile
