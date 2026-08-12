# RepVision

RepVision is a local desktop Python project for real-time bicep-curl counting
and basic form feedback using Ultralytics YOLO Pose and OpenCV.

It provides validated runtime configuration, safe OpenCV camera access, YOLO
pose inference, deterministic primary-person selection, selected-arm keypoint
extraction, elbow-angle smoothing, confirmed repetition counting, conservative
form feedback, a live overlay, keyboard controls, and aggregate session logs.

## Main features

- Local real-time pose inference with a configurable nano YOLO Pose model
- Left- or right-arm tracking for the largest detected person
- Confidence filtering, robust angle smoothing, and confirmed-frame counting
- Readable live feedback, progress, repetition count, movement stage, and FPS
- Pause, reset, arm-switch, keyboard quit, and window-close handling
- Privacy-safe CSV summaries without saved webcam images or video

## Technology stack

- Python 3.11+
- Ultralytics YOLO Pose and PyTorch
- OpenCV and NumPy
- Standard-library CSV persistence
- Pytest and Ruff for offline quality checks

## Requirements

- Python 3.11 or another version supported by the declared dependencies
- A webcam for live workouts and camera diagnostics

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

Start a live workout (press `Q` to finish):

```console
repvision
```

The camera window shows the selected arm, repetition count, smoothed elbow
angle, movement stage, feedback, curl progress, and frame rate. Its controls
are:

- `Q`: finish the workout and close the window
- `R`: reset the current counter and aggregate statistics
- `P`: pause or resume camera inference
- `L`: switch between left-arm and right-arm tracking

Open the configured camera, read exactly one frame, release it, and exit:

```console
repvision --check-camera
```

Load the configured pose model, analyze one webcam frame locally, print the
number of people, selected-arm visibility, elbow angle, movement stage, and
repetition count, then exit:

```console
repvision --check-pose
```

Select a different camera, arm, confidence, input size, or model:

```console
repvision --check-pose --camera-index 1 --arm left --confidence 0.6 --input-size 480 --model yolo26n-pose.pt
```

Choose a different aggregate-log directory for a live workout:

```console
repvision --output-directory my-sessions
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

## Angle and repetition logic

The elbow angle is calculated at the elbow from the shoulder-to-elbow and
wrist-to-elbow vectors. Undefined zero-length or non-finite geometry produces
no measurement. A bounded median window reduces single-frame pose noise while
missing measurements are ignored.

The repetition counter uses explicit `unknown`, `down`, and `up` stages. It
requires configurable consecutive endpoint frames before accepting a stage.
A repetition is counted only for a confirmed `down` to confirmed `up`
transition. Middle-range jitter, repeated frames, partial movement, missing or
low-confidence points, and movements inside the cooldown do not add reps.

The one-frame pose diagnostic can verify angle extraction but cannot complete
a repetition. Use `repvision` for continuous tracking and perform a full
extension followed by a full curl to exercise the repetition state machine.

## Feedback and session data

Visibility feedback asks the user to move back or reports low keypoint
confidence. Movement prompts encourage full extension and curl completion. A
conservative shoulder-elbow-hip angle heuristic can warn about upper-arm drift;
this warning never rejects an otherwise valid repetition.

When a workout ends, RepVision appends one row to `outputs/sessions.csv` by
default. The row contains only the start datetime, exercise, selected arm,
duration, repetitions, warning count, and average reliable repetition duration.

## Privacy

RepVision processes webcam frames locally. Neither a diagnostic nor a live
workout uploads, records, or saves frames or video. Only aggregate session
values are written. Model files and generated image, video, and CSV artifacts
are excluded from Git.

## Current limitations

The tracker supports one selected arm and the largest detected person. Its 2D
angle and form heuristic can be affected by occlusion, camera placement, loose
clothing, and depth ambiguity. Form feedback is guidance, not a medical or
biomechanical assessment.
