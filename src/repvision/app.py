"""Command-line entry point for RepVision."""

import argparse
from collections.abc import Sequence

from repvision.camera import Camera, CameraError
from repvision.config import AppConfig, Arm


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
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="open the camera, read one frame, and exit",
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
    print(
        "RepVision foundation is ready "
        f"(camera={config.camera_index}, arm={config.selected_arm.value}, "
        f"model={config.model_name})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
