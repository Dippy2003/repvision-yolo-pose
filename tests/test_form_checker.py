from repvision.form_checker import FeedbackMessage


def test_feedback_messages_match_interface_copy() -> None:
    assert FeedbackMessage.GOOD_MOVEMENT == "Good movement"
    assert FeedbackMessage.FULLY_EXTEND == "Fully extend your arm"
    assert FeedbackMessage.COMPLETE_CURL == "Complete the curl"
    assert FeedbackMessage.ELBOW_DRIFT == "Keep your elbow close to your body"
    assert FeedbackMessage.MOVE_BACK == "Move back so your arm is visible"
    assert FeedbackMessage.LOW_CONFIDENCE == "Low keypoint confidence"
