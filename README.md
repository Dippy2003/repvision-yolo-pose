# RepVision

RepVision is a local desktop Python project for real-time bicep-curl counting
and basic form feedback using Ultralytics YOLO Pose and OpenCV.

Development is in progress. The current foundation provides validated runtime
configuration, safe OpenCV camera access, and deterministic unit tests. Pose
inference and movement analysis are not implemented yet.

## Requirements

- Python 3.11 or another version supported by the declared dependencies
- A webcam for the optional camera diagnostic

## Installation

Create and activate a virtual environment, then install the package and its
development tools:

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Ultralytics model weights are not downloaded during installation or tests.

## Commands

Validate configuration without opening a webcam:

```console
repvision
```

Open the configured camera, read exactly one frame, release it, and exit:

```console
repvision --check-camera
```

Select a different camera, arm, or future model:

```console
repvision --camera-index 1 --arm left --model yolo26n-pose.pt
```

Run the quality checks:

```console
ruff check .
pytest
python -m repvision.app --help
```

## Configuration

Defaults live in the frozen `AppConfig` dataclass. They include camera and arm
selection, the model name, keypoint confidence, movement thresholds, temporal
filtering, inference size, and generated-output location. Invalid ranges fail
early with focused error messages.

## Privacy

RepVision is designed to process webcam frames locally. The camera diagnostic
does not upload, record, or save its frame. Model files and generated image,
video, and CSV artifacts are excluded from Git.

## Current limitations

The live pose loop, keypoint extraction, angle calculation, repetition state
machine, visual interface, keyboard controls, form feedback, and session CSV
logging remain to be implemented.
