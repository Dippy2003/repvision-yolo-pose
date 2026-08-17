from repvision.audio import AudioCue, AudioCueController
from repvision.form_checker import FeedbackMessage, FormFeedback
from repvision.rep_counter import CurlUpdate, MovementStage


def update(*, rep_completed: bool = False) -> CurlUpdate:
    return CurlUpdate(
        40.0,
        40.0,
        1 if rep_completed else 0,
        MovementStage.UP,
        rep_completed,
    )


def feedback(*, warning: bool = False) -> FormFeedback:
    return FormFeedback(
        FeedbackMessage.ELBOW_DRIFT if warning else FeedbackMessage.GOOD_MOVEMENT,
        is_form_warning=warning,
    )


def test_audio_controller_emits_completed_rep_cue() -> None:
    emitted: list[AudioCue] = []
    controller = AudioCueController(True, emitted.append)

    result = controller.update(update(rep_completed=True), feedback())

    assert result is AudioCue.REP_COMPLETE
    assert emitted == [AudioCue.REP_COMPLETE]


def test_audio_controller_emits_warning_once_per_episode() -> None:
    emitted: list[AudioCue] = []
    controller = AudioCueController(True, emitted.append)

    controller.update(update(), feedback(warning=True))
    controller.update(update(), feedback(warning=True))
    controller.update(update(), feedback())
    controller.update(update(), feedback(warning=True))

    assert emitted == [AudioCue.FORM_WARNING, AudioCue.FORM_WARNING]


def test_disabled_audio_controller_stays_silent() -> None:
    emitted: list[AudioCue] = []
    controller = AudioCueController(False, emitted.append)

    assert controller.update(update(rep_completed=True), feedback()) is None
    assert emitted == []


def test_audio_controller_reset_allows_new_warning_episode() -> None:
    emitted: list[AudioCue] = []
    controller = AudioCueController(True, emitted.append)
    controller.update(update(), feedback(warning=True))

    controller.reset()
    controller.update(update(), feedback(warning=True))

    assert emitted == [AudioCue.FORM_WARNING, AudioCue.FORM_WARNING]
