"""
Unit tests for Video Processor and Ingestion Validation
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from core.video_processor import (
    extract_video_metadata,
    validate_video_file,
)


def test_validate_video_file_supported_extensions():
    """Ensures valid surveillance video formats pass validation."""
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        is_valid, msg = validate_video_file(f"traffic_feed{ext}", file_size_bytes=1024 * 1024)
        assert is_valid is True
        assert "supported" in msg.lower()


def test_validate_video_file_unsupported_extensions():
    """Ensures invalid formats are rejected gracefully with helpful messages."""
    for ext in [".txt", ".exe", ".png", ".jpg", ".csv", ".pdf"]:
        is_valid, msg = validate_video_file(f"payload{ext}", file_size_bytes=1024)
        assert is_valid is False
        assert "unsupported" in msg.lower()


def test_validate_video_file_size_limit():
    """Ensures oversized files exceeding MAX_VIDEO_SIZE_MB are rejected."""
    oversized_bytes = (301) * 1024 * 1024  # > 300MB
    is_valid, msg = validate_video_file("heavy_feed.mp4", file_size_bytes=oversized_bytes)
    assert is_valid is False
    assert "exceeds" in msg.lower()


def test_extract_video_metadata_nonexistent_file():
    """Ensures missing files return structured invalid response rather than raising."""
    result = extract_video_metadata("non_existent_path_404.mp4")
    assert result["is_valid"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_extract_video_metadata_valid_video():
    """Creates a temporary valid OpenCV video and verifies metadata extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video_path = os.path.join(tmpdir, "test_clip.mp4")
        
        # Create a simple 10-frame 640x480 test video with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
        
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        
        # Extract metadata
        meta = extract_video_metadata(test_video_path)
        
        assert meta["is_valid"] is True
        assert meta["width"] == 640
        assert meta["height"] == 480
        assert meta["frame_count"] == 10
        assert meta["fps"] == pytest.approx(30.0, abs=1.0)
        assert meta["error"] is None
        assert "640x480" in meta["resolution"]


def test_extract_video_metadata_corrupted_empty_file():
    """Ensures 0-byte or corrupted non-video file fails gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"NOT_A_REAL_VIDEO_HEADER")
        temp_path = f.name
        
    try:
        meta = extract_video_metadata(temp_path)
        assert meta["is_valid"] is False
        assert meta["error"] is not None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
