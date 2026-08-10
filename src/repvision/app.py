"""Command-line entry point for RepVision."""

import argparse
from collections.abc import Sequence

from repvision.camera import Camera, CameraError
from repvision.config import AppConfig, Arm
from repvision.pose_detector import PoseDetector, PoseDetectorError, PoseObservation
from repvision.rep_counter import CurlTracker


def build_parser() -> argparse.ArgumentParser:
    """Build the RepVision command-line parser."""
    defaults = AppConfig()
    parser = argparse.ArgumentParser(
        prog="repvision",
        description="Run local bicep-curl tracking with YOLO Pose.",
    )
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--camera-index", type=int, default=defaults.camera_index)
    parser.add_argument(
        "--arm",
        type=Arm,
        choices=list(Arm),
        default=defaults.selected_arm,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=defaults.confidence_threshold,
        help="minimum confidence for shoulder, elbow, and wrist",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=defaults.input_size,
        help="square model inference size in pixels",
    )
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--check-camera",
        action="store_true",
        help="open the camera, read one frame, and exit",
    )
    diagnostics.add_argument(
        "--check-pose",
        action="store_true",
        help="load the model, analyze one camera frame, and exit",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Convert parsed command-line values into validated settings."""
    return AppConfig(
        model_name=args.model,
        camera_index=args.camera_index,
        selected_arm=args.arm,
        confidence_threshold=args.confidence,
        input_size=args.input_size,
    )


def check_camera(config: AppConfig) -> tuple[int, ...]:
    """Read one frame and release the camera immediately."""
    with Camera(config.camera_index) as camera:
        return camera.read().shape


def check_pose(config: AppConfig) -> PoseObservation:
    """Load the model and analyze exactly one locally captured frame."""
    detector = PoseDetector(config)
    with Camera(config.camera_index) as camera:
        return detector.detect(camera.read())


def main(argv: Sequence[str] | None = None) -> int:
    """Validate startup settings without starting future processing stages."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.check_camera:
        try:
            frame_shape = check_camera(config)
        except CameraError as error:
            parser.error(str(error))
        print(f"Camera check passed (frame shape={frame_shape}).")
        return 0
    if args.check_pose:
        try:
            observation = check_pose(config)
        except (CameraError, PoseDetectorError) as error:
            parser.error(str(error))
        curl_update = CurlTracker(config).update(observation.selected_arm)
        angle_text = (
            "unavailable"
            if curl_update.smoothed_angle is None
            else f"{curl_update.smoothed_angle:.1f}"
        )
        print(
            "Pose check passed "
            f"(people={len(observation.persons)}, status={observation.status.value}, "
            f"arm={config.selected_arm.value}, angle={angle_text}, "
            f"stage={curl_update.stage.value}, reps={curl_update.count})."
        )
        return 0
    print(
        "RepVision pose pipeline is configured "
        f"(camera={config.camera_index}, arm={config.selected_arm.value}, "
        f"model={config.model_name})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
