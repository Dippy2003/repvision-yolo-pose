from repvision.pose_detector import KeypointIndex


def test_coco_arm_keypoint_indices_match_model_schema() -> None:
    assert KeypointIndex.LEFT_SHOULDER == 5
    assert KeypointIndex.RIGHT_SHOULDER == 6
    assert KeypointIndex.LEFT_ELBOW == 7
    assert KeypointIndex.RIGHT_ELBOW == 8
    assert KeypointIndex.LEFT_WRIST == 9
    assert KeypointIndex.RIGHT_WRIST == 10
    assert KeypointIndex.LEFT_HIP == 11
    assert KeypointIndex.RIGHT_HIP == 12
