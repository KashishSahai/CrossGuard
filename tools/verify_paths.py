import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
from core.pipeline import AdaptivePipeline
from core.camera_calibration import CameraCalibration
from vision.models import DeviceType
from app import process_video_batch

def test_97885():
    print("=== Testing 97885-649761107.mp4 ===")
    calib = CameraCalibration.for_drone_intersection(source_width=1920, source_height=1080)
    
    # Path A: Direct pipeline
    pA = AdaptivePipeline(default_fps=24.0, calibration=calib, device=DeviceType.CPU)
    cap = cv2.VideoCapture("97885-649761107.mp4")
    resA = None
    for f in range(125):
        ret, frame = cap.read()
        if not ret:
            break
        resA = pA.process_frame(frame, frame_index=f, dt=1.0/24.0)
    cap.release()
    diagA = pA.get_last_frame_diagnostics()
    
    # Path B: process_video_batch
    pB = AdaptivePipeline(default_fps=24.0, calibration=calib, device=DeviceType.CPU)
    tracker = {}
    seen = {}
    resB, annotated, next_f, fps = process_video_batch(
        pipeline=pB,
        video_path="97885-649761107.mp4",
        start_frame=0,
        num_frames=125,
        fps=24.0,
        waiting_peds_tracker=tracker,
        waiting_peds_last_seen=seen,
        stride=1
    )
    diagB = pB.get_last_frame_diagnostics()
    
    print(f"Path A (Direct) Frame {resA.frame_index}:")
    for k, v in diagA.items():
        print(f"  {k}: {v}")
        
    print(f"\nPath B (Streamlit batch) Frame {resB.frame_index}:")
    for k, v in diagB.items():
        print(f"  {k}: {v}")

def test_50s():
    print("\n=== Testing tools/intersection_50s.mp4 ===")
    calib = CameraCalibration.for_traffic_intersection_50s(source_width=1920, source_height=1080)
    p = AdaptivePipeline(default_fps=24.0, calibration=calib, device=DeviceType.CPU)
    tracker = {}
    seen = {}
    res, annotated, next_f, fps = process_video_batch(
        pipeline=p,
        video_path="tools/intersection_50s.mp4",
        start_frame=0,
        num_frames=50,
        fps=24.0,
        waiting_peds_tracker=tracker,
        waiting_peds_last_seen=seen,
        stride=2
    )
    diag = p.get_last_frame_diagnostics()
    print(f"50s batch at Frame {res.frame_index}:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    test_97885()
    test_50s()
