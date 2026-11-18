"""Unit tests for the video smoke test utility (Step 8)."""
from __future__ import annotations

import pytest

from tools.video_smoke_test import run_video_smoke_test


def test_video_smoke_test_nonexistent_file():
    """Missing video file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_video_smoke_test("nonexistent_traffic_video_12345.mp4")


def test_video_smoke_test_opencv_handling():
    """When cv2 is missing or video cannot be opened, appropriate error is raised."""
    with pytest.raises((RuntimeError, FileNotFoundError, ValueError)):
        run_video_smoke_test("dummy_sample.mp4")
