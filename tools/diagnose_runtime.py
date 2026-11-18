"""Diagnostic script for CrossGuard Runtime Issues:
1. GPU Worker reachability and health
2. Live Vision detections trace for 97885-649761107.mp4 and intersection_50s.mp4
"""
import os
import sys
import time
import urllib.request
import json
import cv2

sys.path.insert(0, os.path.abspath("."))

print("=" * 60)
print("DIAGNOSTIC 1: GPU WORKER CONFIGURATION & CONNECTIVITY")
print("=" * 60)

env_path = os.path.abspath(".env")
print(f".env path: {env_path}")
print(f".env exists: {os.path.exists(env_path)}")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        print(".env contents:")
        for line in f:
            if "URL" in line or "WORKER" in line or "GPU" in line:
                print("  " + line.strip())

from core.gpu_config import get_configured_worker_url, check_worker_status, normalize_worker_url, _STATUS_CACHE

raw_env = os.environ.get("CROSSGUARD_GPU_WORKER_URL")
print(f"os.environ['CROSSGUARD_GPU_WORKER_URL']: {raw_env}")

url = get_configured_worker_url()
print(f"get_configured_worker_url(): {url}")

print(f"_STATUS_CACHE before check: {_STATUS_CACHE}")
status = check_worker_status(force_refresh=True)
print(f"check_worker_status(force_refresh=True): {status}")

if url:
    print(f"\nTesting direct HTTP GET to {url}/health ...")
    req = urllib.request.Request(
        f"{url}/health",
        headers={"User-Agent": "CrossGuard-Diagnostic/1.0"},
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            elapsed = time.perf_counter() - t0
            body = resp.read().decode("utf-8")
            print(f"  HTTP Status: {resp.status} (took {elapsed:.2f}s)")
            print(f"  Response Body: {body}")
    except Exception as ex:
        print(f"  Direct GET failed: {type(ex).__name__}: {ex}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 2: LIVE VISION DETECTIONS TRACE")
print("=" * 60)

from core.pipeline import AdaptivePipeline
from vision.vision_pipeline import VisionPipelineConfig
from core.camera_calibration import CameraCalibration

for vpath in ["tools/uploads/97885-649761107.mp4", "tools/intersection_50s.mp4", "97885-649761107.mp4"]:
    if not os.path.exists(vpath):
        # search for it
        matches = [os.path.join(r, f) for r, d, fs in os.walk(".") for f in fs if "97885" in f]
        if matches and vpath == "tools/uploads/97885-649761107.mp4":
            vpath = matches[0]
        else:
            continue
    print(f"\nTesting video: {vpath}")
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"Could not open {vpath}")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  Resolution: {w}x{h}, FPS: {fps}, Frames: {total_frames}")

    # Set up pipeline exactly as app.py does
    calib = None
    if "50s" in vpath or "intersection_50s" in vpath:
        calib = CameraCalibration.for_traffic_intersection_50s(source_width=float(w), source_height=float(h))
    elif "97885" in vpath:
        calib = CameraCalibration.for_drone_intersection(source_width=float(w), source_height=float(h))

    pipeline = AdaptivePipeline(
        default_fps=fps,
        calibration=calib,
        source_resolution=(float(w), float(h)),
        device="cpu",
    )

    # Test frames: 0, 10, 50, 100, 124
    test_indices = [0, 10, 50, 100, min(124, total_frames - 1)]
    for f_idx in test_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  Frame {f_idx}: Read failed")
            continue
        
        # Test raw YOLO directly
        raw_dets = pipeline.vision.backend.detect(frame)
        class_counts = {}
        for d in raw_dets:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        
        # Run process_frame
        res = pipeline.process_frame(frame, frame_index=f_idx, dt=0.04)
        print(f"  Frame {f_idx:3d}: Shape: {frame.shape} | Raw YOLO: {len(raw_dets)} {class_counts} | Tracked: {len(res.tracked_detections)} | Spatial: {len(res.spatial_detections)} | Dem: v={res.vehicle_demand}, p={res.waiting_pedestrians}")
    cap.release()
