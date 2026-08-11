"""Readable OpenCV workout overlay rendering."""


def curl_progress(
    angle: float | None, up_threshold: float, down_threshold: float
) -> float | None:
    """Map elbow angle to extension=0 and curl=1 progress."""
    if angle is None:
        return None
    progress = (down_threshold - angle) / (down_threshold - up_threshold)
    return max(0.0, min(1.0, progress))
