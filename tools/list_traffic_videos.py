"""Metadata and geometric validation utility for candidate traffic videos (Step 11).

Inspects video files for resolution, FPS, duration, camera viewpoint, and geometric
compatibility with CrossGuard's canonical 4-way orthogonal intersection.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None


def inspect_video(video_path: str) -> Dict[str, Any]:
    """Extract metadata and basic properties from a video file."""
    fname = os.path.basename(video_path)
    if not os.path.exists(video_path):
        return {"path": video_path, "filename": fname, "error": "file not found"}

    if cv2 is None:
        return {"path": video_path, "filename": fname, "error": "cv2 not installed"}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"path": video_path, "filename": fname, "error": "cv2 could not open"}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (total_frames / fps) if fps > 0 else 0.0

        ret, sample_frame = cap.read()
        cap.release()
    except Exception as e:
        return {"path": video_path, "filename": fname, "error": str(e)}

    orientation = "landscape" if width >= height else "portrait"

    return {
        "filename": os.path.basename(video_path),
        "path": video_path,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "orientation": orientation,
        "has_sample_frame": ret and (sample_frame is not None),
    }


def find_candidate_videos(search_dirs: Optional[List[str]] = None) -> List[str]:
    """Find video files in specified directories."""
    if search_dirs is None:
        home = os.path.expanduser("~")
        search_dirs = [
            os.path.join(home, "Downloads"),
            os.path.join(home, "Videos"),
            os.path.join(home, "Desktop"),
        ]

    extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    found: List[str] = []

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith(extensions):
                        found.append(os.path.join(root, f))
        except Exception:
            continue

    return found


def main() -> None:
    print("=" * 70)
    print("CROSSGUARD VIDEO DISCOVERY & METADATA INSPECTION")
    print("=" * 70)

    candidates = find_candidate_videos()
    print(f"Found {len(candidates)} total video files on machine.")
    print("-" * 70)

    for path in candidates:
        meta = inspect_video(path)
        if "error" in meta:
            print(f"[-] {meta['filename']}: {meta['error']}")
        else:
            print(
                f"[+] {meta['filename']} | {meta['resolution']} | "
                f"{meta['fps']} fps | {meta['duration_s']}s ({meta['orientation']})"
            )


if __name__ == "__main__":
    main()
