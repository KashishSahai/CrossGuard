"""Video validation and smoke-testing utility for CrossGuard (Step 8 & Step 9).

Validates an actual video file through AdaptivePipeline:
  Video -> Frames -> Tracking -> Spatial Analysis -> Telemetry -> FSM -> Bloom City Signal State

Clearly distinguishes real active YOLO detection from fallback/unavailable states.

Usage:
  python tools/video_smoke_test.py [video_path] [--max-frames 200] [--fps 24.0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import PedestrianSignal, Phase, VehicleSignal
from core.pipeline import AdaptivePipeline, PipelineFrameResult
from simulation.controller_bridge import BloomCitySignalState


def run_video_smoke_test(
    video_path: str,
    max_frames: Optional[int] = 200,
    target_fps: Optional[float] = None,
) -> Dict[str, Any]:
    """Run smoke test on a local video file and collect diagnostic statistics."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Inspect video metadata with OpenCV
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "OpenCV (cv2) is required to run video_smoke_test.py directly. "
            "Install opencv-python or run with a Python environment containing cv2."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file via cv2: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    effective_fps = target_fps if target_fps and target_fps > 0 else (src_fps if src_fps > 0 else 10.0)

    # Initialize Adaptive Pipeline
    pipeline = AdaptivePipeline(default_fps=effective_fps)
    vision_status = pipeline.vision_status
    vision_load_error = pipeline.vision_load_error

    print("=" * 65)
    print("CROSSGUARD VIDEO SMOKE TEST (Step 9)")
    print("=" * 65)
    print(f"Video path:         {video_path}")
    print(f"Source Resolution:  {width} x {height}")
    print(f"Source FPS:         {src_fps:.2f}")
    print(f"Total Video Frames: {total_frames}")
    print(f"Max Frames to Test: {max_frames}")
    print(f"Device:             {pipeline.device} (auto-resolved)")
    print(f"Vision Backend:     {vision_status}")
    if vision_load_error:
        print(f"Vision Load Error:  {vision_load_error}")
    print("-" * 65)

    start_time = time.perf_counter()
    results = pipeline.process_video_file(
        video_path=video_path,
        max_frames=max_frames,
        target_fps=target_fps,
    )
    elapsed_time = time.perf_counter() - start_time

    num_processed = len(results)
    avg_frame_time = (elapsed_time / num_processed) if num_processed > 0 else 0.0
    approx_fps = (num_processed / elapsed_time) if elapsed_time > 0 else 0.0

    # Aggregate statistics
    frames_with_detections = sum(1 for r in results if r.num_tracked_objects > 0)
    total_tracked_observed = sum(r.num_tracked_objects for r in results)
    peak_tracked = max((r.num_tracked_objects for r in results), default=0)
    avg_detections_per_frame = (total_tracked_observed / num_processed) if num_processed > 0 else 0.0

    total_waiting_peds = sum(r.waiting_pedestrians for r in results)
    total_crossing_peds = sum(r.crossing_pedestrians for r in results)
    total_queued_veh = sum(r.queued_vehicles for r in results)
    total_approaching_veh = sum(r.approaching_vehicles for r in results)

    phases_observed: Set[str] = {r.current_phase.value for r in results}
    vehicle_signals_observed: Set[str] = {r.vehicle_signal.value for r in results}
    pedestrian_signals_observed: Set[str] = {r.pedestrian_signal.value for r in results}

    all_signals_safe = all(r.signal_state.is_safe for r in results)

    # Intersection compatibility check
    # Existing Bloom City canonical canvas: 1000 x 800 with 4-way orthogonal crossing
    is_canonical_canvas = (width == 1000 and height == 800)
    compatibility = "compatible" if is_canonical_canvas else "incompatible (different aspect/geometry)"

    print("--- VIDEO METRICS ---")
    print(f"Frames processed:           {num_processed}")
    print(f"Effective dt:               {1.0 / effective_fps:.4f} s")
    print(f"Camera View Compatibility:  {compatibility}")
    print()
    print("--- VISION & PERCEPTION ---")
    print(f"YOLO Status:                {vision_status}")
    if vision_status != "ACTIVE":
        print("  [!] WARNING: YOLO is unavailable in this environment.")
        print(f"  [!] Reason: {vision_load_error}")
        print("  [!] 0 detections observed is due to backend unavailability, not empty scene.")
    print(f"Frames with detections:     {frames_with_detections}")
    print(f"Total tracked objects:      {total_tracked_observed}")
    print(f"Peak tracked in a frame:    {peak_tracked}")
    print(f"Avg detections per frame:   {avg_detections_per_frame:.2f}")
    print()
    print("--- SPATIAL DEMAND ---")
    print(f"Waiting pedestrians:        {total_waiting_peds}")
    print(f"Crossing pedestrians:       {total_crossing_peds}")
    print(f"Queued vehicles:            {total_queued_veh}")
    print(f"Approaching vehicles:       {total_approaching_veh}")
    print()
    print("--- CONTROLLER & SIGNALS ---")
    print(f"Controller phases observed: {sorted(phases_observed)}")
    print(f"Vehicle signals observed:   {sorted(vehicle_signals_observed)}")
    print(f"Pedestrian signals observed:{sorted(pedestrian_signals_observed)}")
    print(f"Bloom City Signal Safety:   {'PASS' if all_signals_safe else 'FAIL'}")
    print()
    print("--- PERFORMANCE ---")
    print(f"Elapsed Processing Time:    {elapsed_time:.3f} s")
    print(f"Approx Processing FPS:      {approx_fps:.1f} fps ({avg_frame_time * 1000:.1f} ms/frame)")
    print("=" * 65)

    return {
        "video_path": video_path,
        "width": width,
        "height": height,
        "src_fps": src_fps,
        "effective_fps": effective_fps,
        "dt": 1.0 / effective_fps,
        "frames_processed": num_processed,
        "frames_with_detections": frames_with_detections,
        "total_tracked_objects": total_tracked_observed,
        "peak_tracked": peak_tracked,
        "avg_detections_per_frame": avg_detections_per_frame,
        "waiting_pedestrians": total_waiting_peds,
        "crossing_pedestrians": total_crossing_peds,
        "queued_vehicles": total_queued_veh,
        "approaching_vehicles": total_approaching_veh,
        "phases_observed": sorted(phases_observed),
        "vehicle_signals_observed": sorted(vehicle_signals_observed),
        "pedestrian_signals_observed": sorted(pedestrian_signals_observed),
        "signals_safe": all_signals_safe,
        "elapsed_time": elapsed_time,
        "approx_fps": approx_fps,
        "compatibility": compatibility,
        "device": pipeline.device,
        "vision_status": vision_status,
        "vision_load_error": vision_load_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossGuard Video Validation Smoke Test")
    parser.add_argument("video_path", nargs="?", default=None, help="Path to traffic video file")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to process")
    parser.add_argument("--fps", type=float, default=None, help="Target processing FPS")
    args = parser.parse_args()

    video_path = args.video_path
    if not video_path:
        # Check standard default search locations
        candidates = [
            r"C:\Users\KIIT0001\Downloads\19166351-uhd_3840_2160_24fps.mp4",
            "traffic_sample.mp4",
        ]
        for c in candidates:
            if os.path.exists(c):
                video_path = c
                break

    if not video_path or not os.path.exists(video_path):
        print("ERROR: No traffic video file specified or found.")
        sys.exit(1)

    run_video_smoke_test(video_path=video_path, max_frames=args.max_frames, target_fps=args.fps)


if __name__ == "__main__":
    main()
