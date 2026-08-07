"""Command-line entry point for RepVision."""

import argparse
from collections.abc import Sequence

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
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Convert parsed command-line values into validated settings."""
    return AppConfig(
        model_name=args.model,
        camera_index=args.camera_index,
        selected_arm=args.arm,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate startup settings without starting future processing stages."""
    config = config_from_args(build_parser().parse_args(argv))
    print(
        "RepVision foundation is ready "
        f"(camera={config.camera_index}, arm={config.selected_arm.value}, "
        f"model={config.model_name})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
