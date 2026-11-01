"""
CrossGuard Video Ingestion & Validation Module
==============================================
Provides resilient validation, OpenCV metadata extraction, and temporary file
handling for uploaded real traffic surveillance footage.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
from core.config import MAX_VIDEO_SIZE_MB, SUPPORTED_VIDEO_EXTENSIONS


def validate_video_file(file_name: str, file_size_bytes: int = 0) -> Tuple[bool, str]:
    """
    Validates file extension and size constraints prior to full decoding.

    Args:
        file_name: Name of the uploaded file.
        file_size_bytes: Size of the file in bytes.

    Returns:
        Tuple of (is_valid: bool, status_message: str)
    """
    if not file_name:
        return False, "No file provided."

    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported_str = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
        return (
            False,
            f"Unsupported format '{suffix}'. Supported formats are: {supported_str}",
        )

    max_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        return (
            False,
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_VIDEO_SIZE_MB} MB.",
        )

    return True, "File header matches supported surveillance video format."


def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Uses OpenCV to inspect video properties and decode header parameters.

    Args:
        video_path: Local filesystem path to the video file.

    Returns:
        Dictionary containing resolution, FPS, frame count, duration, and validity.
    """
    if not os.path.exists(video_path):
        return {
            "is_valid": False,
            "error": f"Video file not found at: {video_path}",
            "file_name": os.path.basename(video_path),
        }

    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_name = os.path.basename(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "is_valid": False,
            "error": "Failed to open video stream. The file may be corrupt or encoded in an unsupported codec.",
            "file_name": file_name,
            "file_size_mb": round(file_size_mb, 2),
        }

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Decode fourcc code string safely
        try:
            fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
            if not fourcc_str:
                fourcc_str = "H.264 / AVC"
        except Exception:
            fourcc_str = "UNKNOWN"

        # Validate that frames actually exist
        if width <= 0 or height <= 0 or frame_count <= 0:
            return {
                "is_valid": False,
                "error": "Video stream reported zero frames or invalid resolution dimensions.",
                "file_name": file_name,
                "file_size_mb": round(file_size_mb, 2),
            }

        duration_seconds = (frame_count / fps) if fps > 0 else 0.0
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        duration_formatted = f"{minutes:02d}:{seconds:02d} ({duration_seconds:.1f}s)"

        # Classify standard resolution label
        res_label = f"{width}x{height}"
        if width == 1920 and height == 1080:
            res_label += " (1080p FHD)"
        elif width == 1280 and height == 720:
            res_label += " (720p HD)"
        elif width == 3840 and height == 2160:
            res_label += " (4K UHD)"

        return {
            "is_valid": True,
            "file_name": file_name,
            "file_path": video_path,
            "file_size_mb": round(file_size_mb, 2),
            "width": width,
            "height": height,
            "resolution": res_label,
            "fps": round(fps, 2) if fps > 0 else 30.0,
            "frame_count": frame_count,
            "duration_seconds": round(duration_seconds, 2),
            "duration_formatted": duration_formatted,
            "codec": fourcc_str,
            "error": None,
        }
    except Exception as exc:
        return {
            "is_valid": False,
            "error": f"Exception encountered while parsing video: {str(exc)}",
            "file_name": file_name,
            "file_size_mb": round(file_size_mb, 2),
        }
    finally:
        cap.release()


def save_temp_video_file(uploaded_file, directory: Optional[str] = None) -> str:
    """
    Saves an uploaded Streamlit file buffer to disk for OpenCV consumption.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        directory: Optional directory to store file in. Defaults to system temp.

    Returns:
        Absolute filesystem path of saved temporary video.
    """
    if directory is None:
        target_dir = os.path.join(tempfile.gettempdir(), "crossguard_media")
    else:
        target_dir = directory

    os.makedirs(target_dir, exist_ok=True)
    clean_name = Path(uploaded_file.name).name
    save_path = os.path.join(target_dir, clean_name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path
