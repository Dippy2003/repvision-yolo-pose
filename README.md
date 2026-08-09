# RepVision

RepVision is a local desktop Python project for real-time bicep-curl counting
and basic form feedback using Ultralytics YOLO Pose and OpenCV.

Development is in progress. The current foundation provides validated runtime
configuration, safe OpenCV camera access, YOLO pose inference, deterministic
primary-person selection, selected-arm keypoint extraction, and offline tests.
Movement analysis is not implemented yet.

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
The default `yolo26n-pose.pt` weights download automatically from Ultralytics
the first time a real pose check is run.

## Commands

Validate configuration without opening a webcam:

```console
repvision
```

Open the configured camera, read exactly one frame, release it, and exit:

```console
repvision --check-camera
```

Load the configured pose model, analyze one webcam frame locally, print the
number of people and selected-arm visibility status, then exit:

```console
repvision --check-pose
```

Select a different camera, arm, or future model:

```console
repvision --check-pose --camera-index 1 --arm left --confidence 0.6 --input-size 480 --model yolo26n-pose.pt
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

## Pose detection

The detector loads one Ultralytics model per application instance and requests
quiet inference at the configured input size. Each result is converted into
application-owned bounding boxes and COCO landmarks. The person with the
largest valid bounding box is selected; model order breaks equal-area ties.

For the selected left or right arm, RepVision extracts shoulder, elbow, wrist,
and hip. Shoulder, elbow, and wrist must each meet the confidence threshold for
the result to have `tracking` status. Other possible statuses are `no_person`,
`missing_keypoints`, and `low_confidence`.

## Privacy

RepVision is designed to process webcam frames locally. Neither diagnostic
uploads, records, nor saves its frame. Model files and generated image, video,
and CSV artifacts are excluded from Git.

## Current limitations

The live processing loop, angle calculation, smoothing, repetition state
machine, visual interface, keyboard controls, form feedback, and session CSV
logging remain to be implemented. A single 2D pose can also be affected by
occlusion, camera placement, and depth ambiguity.
