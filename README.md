# RepVision

RepVision is a local desktop Python project for real-time bicep-curl counting
and basic form feedback using Ultralytics YOLO Pose and OpenCV.

It provides validated runtime configuration, safe OpenCV camera access, YOLO
pose inference, deterministic primary-person selection, selected-arm keypoint
extraction, elbow-angle smoothing, confirmed repetition counting, conservative
form feedback, personalized movement calibration, a live overlay, keyboard
controls, and aggregate session logs.

## Main features

- Local real-time pose inference with a configurable nano YOLO Pose model
- Left- or right-arm tracking for the largest detected person
- Confidence filtering, robust angle smoothing, and confirmed-frame counting
- Readable live feedback, progress, repetition count, movement stage, and FPS
- Pause, reset, arm-switch, keyboard quit, and window-close handling
- Separate, privacy-safe movement calibration profiles for each arm
- Optional local audio cues for completed repetitions and warning episodes
- Aggregate workout-history totals and recent-session reporting
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
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. If the
`repvision` command is not found after installation, keep the environment
activated or use `python -m repvision.app` with the same options.

Ultralytics model weights are not downloaded during installation or tests.
The default `yolo26n-pose.pt` weights download automatically from Ultralytics
the first time a real pose check is run.

## Architecture

The live loop composes small boundaries instead of mixing input, model, and
movement responsibilities. `frame_source.py` defines the shared input boundary;
`camera.py` and `video_source.py` safely own their OpenCV capture resources;
`pose_detector.py` converts YOLO output into application-owned pose types;
`angles.py`, `rep_counter.py`, and `form_checker.py` contain independently
testable movement rules. `renderer.py` draws the view, while `session.py` saves
aggregate-only results. `workout.py` coordinates those components,
`calibration.py` derives and safely stores personalized movement thresholds,
`calibration_workflow.py` owns its camera-guided capture loop, `history.py`
summarizes existing aggregate logs, `audio.py` emits opt-in local cues,
`benchmark.py` measures scalar pipeline latency, and `app.py` provides the
command-line entry point.

The model is constructed once per workout. Webcam frames pass through memory
to inference and rendering and are not passed to session persistence.

## Commands

Start a live workout (press `Q` to finish):

```console
repvision
```

For the most reliable side view, place the camera far enough away to keep your
selected shoulder, elbow, wrist, and hip visible. Start with a fully extended
arm, hold each endpoint briefly, and then complete the curl.

The camera window shows the selected arm, repetition count, smoothed elbow
angle, movement stage, feedback, curl progress, and frame rate. Its controls
are:

- `Q`: finish the workout and close the window
- `R`: reset the current counter and aggregate statistics
- `P`: pause or resume camera inference
- `L`: switch between left-arm and right-arm tracking

Add a local terminal-bell cue for completed repetitions and new form-warning
episodes:

```console
repvision --audio-cues
```

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

Analyze a local prerecorded workout with the same pose and counting pipeline:

```console
repvision --video C:\Videos\curl-test.mp4 --confidence 0.3
```

The input video is opened read-only, processed locally, and never copied or
saved by RepVision. The window closes automatically at the end of the video;
the regular `Q`, `R`, `P`, and `L` controls remain available during playback.

Measure the complete camera pipeline without opening the workout window:

```console
repvision --benchmark --benchmark-frames 30 --warmup-frames 2 --confidence 0.3
```

For a repeatable comparison, benchmark the same local video at different input
sizes:

```console
repvision --benchmark --video C:\Videos\curl-test.mp4 --input-size 320
repvision --benchmark --video C:\Videos\curl-test.mp4 --input-size 480
repvision --benchmark --video C:\Videos\curl-test.mp4 --input-size 640
```

The report includes mean, median, and p95 latency for capture, inference,
movement analysis, rendering, and the complete frame, plus effective FPS.
Warm-up frames are processed but excluded from those statistics. Only scalar
timings are retained; benchmark mode does not save frames, video, or CSV data.

Run the quality checks:

```console
ruff check .
pytest
python -m repvision.app --help
```

## Repeatable verification workflow

Run these checks in order whenever you want to verify the complete system.
The first phase is fully offline; later phases use your own camera or video.

### Installation and offline quality

```console
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m repvision.app --help
```

This phase passes when Ruff reports no errors, every test passes, and the help
screen lists the camera, pose, calibration, benchmark, history, and audio
options. Tests never open a camera or download model weights.

### Camera access

```console
repvision --check-camera
```

This phase passes when one frame shape is printed and the camera is released.
If it fails, close applications using the camera or try `--camera-index 1`.

### Pose visibility

```console
repvision --check-pose --confidence 0.3 --arm right
```

This phase passes when `status=tracking` and a numeric angle are printed. A
single-frame check normally remains at `stage=unknown` and `reps=0`; continuous
frames are required to confirm movement stages and repetitions.

### Personalized calibration

```console
repvision --calibrate --arm right --confidence 0.3
repvision --calibration-status --arm right
repvision --check-pose --arm right --confidence 0.3
```

This phase passes when the guided capture completes, status prints numeric
curled/extended/up/down values, and the pose check reports
`thresholds=personalized`. Repeat with `--arm left` to create a separate left
profile. To repeat from scratch, run `repvision --reset-calibration --arm right`.

### Live counting and controls

```console
repvision --arm right --confidence 0.3
```

The window should say `Thresholds: PERSONALIZED` when that arm has a profile.
Confirm that a complete extension followed by a complete curl counts once,
partial motion does not count, and `P`, `R`, `L`, and `Q` work. Compare defaults
with `repvision --no-calibration --arm right --confidence 0.3`.

### Prerecorded video

First confirm that the file really exists, then analyze it:

```powershell
Test-Path "C:\Users\DIPNA\Videos\curl-test.mp4"
repvision --video "C:\Users\DIPNA\Videos\curl-test.mp4" --confidence 0.3 --input-size 480
```

`Test-Path` must print `True`. RepVision does not create or download this video;
replace the example with the full path of a video already on your computer.

### Performance benchmark

```console
repvision --benchmark --benchmark-frames 30 --warmup-frames 2 --confidence 0.3
```

This phase passes when capture, inference, analysis, rendering, total latency,
and effective FPS are printed. Repeat with input sizes `320`, `480`, and `640`
on the same camera setup or video for a fair comparison.

### Saved aggregate history

```console
repvision --history
repvision --history --history-limit 10
```

This phase passes when totals and recent workouts are printed, or an explicit
message says that no aggregate workout sessions exist yet.

## Configuration

Defaults live in the frozen `AppConfig` dataclass. They include camera and arm
selection, the model name, keypoint confidence, movement thresholds, temporal
filtering, inference size, calibration sample count, minimum movement range,
threshold margin, and generated-output location. Invalid ranges fail early
with focused error messages.

Create a personalized profile for the selected arm:

```console
repvision --calibrate --arm right --confidence 0.3
```

The camera window guides the complete process:

1. Stand side-on with the selected shoulder, elbow, and wrist visible.
2. Fully extend the arm and press `Space`.
3. Hold still until the extended sample counter reaches its target.
4. Fully curl the arm and press `Space` again.
5. Hold still until the curled sample counter reaches its target.

Press `R` to restart both endpoint captures or `Q` to cancel without saving.
The default target is 20 reliable frames per endpoint. It can be adjusted for
diagnosis with `--calibration-samples`, but values below three are rejected.

Inspect or remove one arm's profile without opening the camera:

```console
repvision --calibration-status --arm right
repvision --reset-calibration --arm right
```

Saved profiles apply automatically during pose checks, benchmarks, videos, and
live workouts. The workout window and pose-check output show whether thresholds
are `PERSONALIZED` or `DEFAULT`. For a one-run comparison that ignores profiles:

```console
repvision --no-calibration
```

The calibration engine collects a bounded number of reliable elbow angles at
the curled and extended positions. It uses each position's median, rejects an
unsafe movement range, and derives thresholds inside the measured endpoints.
Profiles for the left and right arms remain separate, including when `L`
switches arms during a workout.

Calibration profiles use a versioned JSON document stored under the current
user's local application-data directory (`RepVision/calibration.json` on
Windows). A profile contains only the arm, aggregate endpoint angles, derived
thresholds, sample count, and timestamp. It never contains frames, video,
keypoints, or sample history. The guided camera workflow that creates and
resets these profiles is entirely local.

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

Show totals and the five most recent aggregate sessions without opening the
camera:

```console
repvision --history
```

## Troubleshooting

- If the camera cannot open, close other camera applications and try
  `repvision --check-camera --camera-index 1` for a second device.
- If angle is `unavailable`, keep the selected shoulder, elbow, and wrist in
  view, improve lighting, and confirm the correct arm with `--arm` or `L`.
- If your keypoints are consistently below the default threshold, try
  `repvision --confidence 0.3`. Lower values accept noisier detections and can
  make the angle less stable.
- If CPU inference is too slow, try `repvision --input-size 480`. Smaller input
  sizes usually improve throughput but can reduce keypoint quality, so confirm
  that full curls still count reliably after changing it.
- The first pose run may need internet access to obtain the configured model.
  Later runs can use the local `.pt` file. Tests never download model weights.
- Keyboard commands work while the OpenCV workout window has focus.
- If a previous `sessions.csv` has different columns, move it outside the
  output directory so RepVision can create the current schema safely.
- If a calibration file is damaged, use `--no-calibration` for an immediate
  workout, then move the reported JSON file aside and calibrate again.
- Audio cues use the local terminal bell and depend on terminal/operating-system
  sound settings. They never send audio or workout data anywhere.

## Privacy

RepVision processes webcam frames locally. Neither a diagnostic nor a live
workout uploads, records, or saves frames or video. Only aggregate session
values are written. Model files and generated image, video, and CSV artifacts
are excluded from Git.

## Limitations

The tracker supports one selected arm and the largest detected person. Its 2D
angle and form heuristic can be affected by occlusion, camera placement, loose
clothing, and depth ambiguity. Form feedback is guidance, not a medical or
biomechanical assessment.

## Future improvements

- Additional exercises built on separate, tested movement state machines
- Measured CPU performance profiles across supported model and input sizes
- An optional graphical history view built from aggregate-only session data
