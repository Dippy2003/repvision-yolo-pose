from repvision.form_checker import FeedbackMessage, FormFeedback


def test_feedback_messages_match_interface_copy() -> None:
    assert FeedbackMessage.GOOD_MOVEMENT == "Good movement"
    assert FeedbackMessage.FULLY_EXTEND == "Fully extend your arm"
    assert FeedbackMessage.COMPLETE_CURL == "Complete the curl"
    assert FeedbackMessage.ELBOW_DRIFT == "Keep your elbow close to your body"
    assert FeedbackMessage.MOVE_BACK == "Move back so your arm is visible"
    assert FeedbackMessage.LOW_CONFIDENCE == "Low keypoint confidence"


def test_feedback_defaults_to_non_warning_movement_message() -> None:
    feedback = FormFeedback(FeedbackMessage.GOOD_MOVEMENT)

    assert not feedback.is_form_warning
    assert not feedback.is_visibility_issue
