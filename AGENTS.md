# Repository Guidelines

## Scope

RepVision is a local desktop Python application for bicep-curl counting with
Ultralytics YOLO Pose and OpenCV. Keep webcam frames local and never save raw
frames or video by default.

## Structure

- Use the `src/repvision` package and place deterministic tests in `tests/`.
- Keep pose inference separate from camera access and pure movement logic.
- Keep all tunable runtime values in a typed configuration object.
- Do not use `ultralytics.solutions.AIGym`; implement project logic directly.
- Do not add duplicate tool configuration, notebooks, model weights, generated
  output, IDE files, placeholder files, or unused dependencies.

## Commands

- Install development dependencies: `python -m pip install -e ".[dev]"`
- Lint: `ruff check .`
- Test: `pytest`
- Smoke check: `python -m repvision.app --help`

## Quality and Safety

- Target Python 3.11 or a compatible supported version and use type hints.
- Unit tests must not access a webcam, download weights, require a GPU or
  internet access, or write generated artifacts into the repository.
- Use focused exceptions for expected failures and always release camera
  resources safely.
- Preserve unrelated user changes. Never rewrite Git history or push without
  explicit authorization.
- Before finishing, run Ruff, Pytest, and a non-interactive application smoke
  check; review the final diff, recent commits, and `git status --short`.
