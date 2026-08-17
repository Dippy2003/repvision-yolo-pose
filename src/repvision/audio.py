"""Optional local-only workout audio cues."""

import sys
from collections.abc import Callable
from enum import StrEnum

from repvision.form_checker import FormFeedback
from repvision.rep_counter import CurlUpdate


class AudioCue(StrEnum):
    """Distinct events that may produce an audible terminal bell."""

    REP_COMPLETE = "rep_complete"
    FORM_WARNING = "form_warning"


AudioBackend = Callable[[AudioCue], None]


def terminal_bell(_cue: AudioCue) -> None:
    """Emit a portable local terminal bell without adding dependencies."""
    sys.stdout.write("\a")
    sys.stdout.flush()


class AudioCueController:
    """Emit enabled cues once per repetition or warning episode."""

    def __init__(
        self,
        enabled: bool,
        backend: AudioBackend = terminal_bell,
    ) -> None:
        self.enabled = enabled
        self.backend = backend
        self._warning_active = False

    def update(
        self, update: CurlUpdate, feedback: FormFeedback
    ) -> AudioCue | None:
        """Consume one frame and return any cue emitted for it."""
        warning_started = feedback.is_form_warning and not self._warning_active
        self._warning_active = feedback.is_form_warning
        cue = None
        if update.rep_completed:
            cue = AudioCue.REP_COMPLETE
        elif warning_started:
            cue = AudioCue.FORM_WARNING
        if self.enabled and cue is not None:
            self.backend(cue)
            return cue
        return None

    def reset(self) -> None:
        """Clear warning-episode state after a workout reset or arm switch."""
        self._warning_active = False
