"""FastAPI Server for CrossGuard Google Colab CUDA Worker.

Runs in Google Colab with an active GPU runtime (e.g. NVIDIA T4 / A100).
Receives video frames, runs YOLOv8n CUDA inference, measures GPU latency,
and returns genuine detections.
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: fastapi and pydantic are required. Install with: pip install fastapi pydantic uvicorn")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is required. Install with: pip install ultralytics")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Global Worker State & Model Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CrossGuard Remote GPU Worker",
    version="1.0.0",
    description="High-performance CUDA inference worker for CrossGuard adaptive traffic control",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else "CPU Only"
MODEL_NAME = os.environ.get("CROSSGUARD_YOLO_MODEL", "yolov8n.pt")

print(f"[*] Initializing CrossGuard GPU Worker...")
print(f"[*] Hardware Device: {DEVICE}")
print(f"[*] GPU Name:        {GPU_NAME}")
print(f"[*] PyTorch Version: {getattr(torch, '__version__', 'unknown')}")

yolo_model = None
yolo_loaded = False
try:
    yolo_model = YOLO(MODEL_NAME)
    # Warm-up inference
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = yolo_model.predict(source=dummy, device=DEVICE, verbose=False)
    yolo_loaded = True
    print(f"[+] YOLOv8n successfully loaded on {DEVICE}")
except Exception as e:
    print(f"[-] Failed to load YOLO model: {e}")
    yolo_loaded = False


# -----------------------------------------------------------------------------
# Dynamic Registry Integration (GitHub Gist)
# -----------------------------------------------------------------------------
_heartbeat_stop = threading.Event()
_heartbeat_thread: Optional[threading.Thread] = None
_registered_worker_url: Optional[str] = None


def publish_registry_record(
    worker_url: str,
    status: str = "online",
    gist_id: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Publish or update worker URL record in GitHub Gist."""
    gid = (gist_id or os.environ.get("CROSSGUARD_GPU_REGISTRY_GIST_ID", "")).strip()
    tok = (token or os.environ.get("CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN", "")).strip()
    if not gid or not tok:
        try:
            from google.colab import userdata
            gid = (gid or str(userdata.get("CROSSGUARD_GPU_REGISTRY_GIST_ID") or "")).strip()
            tok = (tok or str(userdata.get("CROSSGUARD_GPU_REGISTRY_GITHUB_TOKEN") or "")).strip()
        except Exception:
            pass
    if not gid or not tok:
        return False

    import re
    if "gist.github" in gid:
        m = re.search(r"([a-f0-9]{6,})", gid, re.IGNORECASE)
        if m:
            gid = m.group(1)

    record_json = json.dumps(
        {
            "worker_url": worker_url.strip(),
            "worker_id": "crossguard-colab",
            "gpu_name": GPU_NAME,
            "status": status,
            "updated_at": time.time(),
            "protocol_version": "1",
            "heartbeat_interval_s": 60.0,
        },
        indent=2,
    )
    payload = {
        "description": f"CrossGuard Remote GPU Worker Registry [{status.upper()}]",
        "files": {
            "crossguard_worker.json": {
                "content": record_json,
            },
            "gistfile1.txt": {
                "content": record_json,
            },
        },
    }

    url = f"https://api.github.com/gists/{gid}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "User-Agent": "CrossGuard-Colab-Publisher/1.0",
        },
        method="PATCH",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status in (200, 201)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        print(f"[-] Registry update HTTP Error {e.code}: {e.reason} - {err_body}")
        return False
    except Exception as e:
        print(f"[-] Registry update error: {e}")
        return False


def start_registry_heartbeat(worker_url: str, interval_s: float = 60.0):
    global _heartbeat_thread, _heartbeat_stop, _registered_worker_url
    if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
        return                      # already heartbeating: never stack threads that publish different URLs
    _registered_worker_url = worker_url
    _heartbeat_stop.clear()

    ok = publish_registry_record(worker_url, status="online")
    if ok:
        print(f"[+] Registry: updated ({worker_url})")
    else:
        print("[-] Registry: not updated (credentials missing or endpoint unreachable)")

    def _loop():
        while not _heartbeat_stop.wait(interval_s):
            publish_registry_record(worker_url, status="online")

    _heartbeat_thread = threading.Thread(target=_loop, daemon=True, name="RegistryHeartbeat")
    _heartbeat_thread.start()


def stop_registry_heartbeat():
    global _heartbeat_thread, _heartbeat_stop, _registered_worker_url
    _heartbeat_stop.set()
    if _registered_worker_url:
        publish_registry_record(_registered_worker_url, status="offline")


atexit.register(stop_registry_heartbeat)


@app.on_event("startup")
def on_startup():
    pub_url = os.environ.get("CROSSGUARD_GPU_WORKER_URL") or os.environ.get("PUBLIC_TUNNEL_URL")
    if pub_url:
        start_registry_heartbeat(pub_url)


@app.on_event("shutdown")
def on_shutdown():
    stop_registry_heartbeat()


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class DetectionItemModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class InferenceRequestModel(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded JPEG or PNG image")
    confidence_threshold: float = Field(0.35, ge=0.01, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.01, le=1.0)
    frame_index: Optional[int] = None
    target_classes: Optional[List[str]] = None


class InferenceResponseModel(BaseModel):
    status: str = "ok"
    detections: List[DetectionItemModel] = []
    inference_time_ms: float = 0.0
    device: str = DEVICE
    gpu_name: str = GPU_NAME
    frame_index: Optional[int] = None
    error: Optional[str] = None


class HealthResponseModel(BaseModel):
    status: str = "ok"
    cuda_available: bool
    gpu_name: str
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_allocated_mb: Optional[float] = None
    yolo_loaded: bool
    model_name: str
    torch_version: str
    device: str


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponseModel)
def health() -> HealthResponseModel:
    cuda_avail = bool(torch is not None and torch.cuda.is_available())
    vram_total = None
    vram_alloc = None
    if cuda_avail:
        try:
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
            vram_alloc = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        except Exception:
            pass

    return HealthResponseModel(
        status="ok" if (yolo_loaded and cuda_avail) else "degraded",
        cuda_available=cuda_avail,
        gpu_name=GPU_NAME,
        gpu_memory_total_mb=vram_total,
        gpu_memory_allocated_mb=vram_alloc,
        yolo_loaded=yolo_loaded,
        model_name=MODEL_NAME,
        torch_version=getattr(torch, "__version__", "unknown"),
        device=DEVICE,
    )


@app.post("/detect", response_model=InferenceResponseModel)
def detect(req: InferenceRequestModel) -> InferenceResponseModel:
    if not yolo_loaded or yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model is not loaded on worker")

    try:
        img_bytes = base64.b64decode(req.image_b64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Decoded image is empty or invalid")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Precise GPU execution and timing
    t0 = time.perf_counter()
    try:
        results = yolo_model.predict(
            source=frame,
            device=DEVICE,
            conf=req.confidence_threshold,
            iou=req.iou_threshold,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")
    t1 = time.perf_counter()
    inference_ms = round((t1 - t0) * 1000.0, 2)

    detections: List[DetectionItemModel] = []
    if results and len(results) > 0:
        res = results[0]
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy()
            names = res.names if hasattr(res, "names") else {}

            for i in range(len(xyxy)):
                c_id = int(clses[i])
                c_name = names.get(c_id, str(c_id)).lower()
                detections.append(
                    DetectionItemModel(
                        x1=float(xyxy[i][0]),
                        y1=float(xyxy[i][1]),
                        x2=float(xyxy[i][2]),
                        y2=float(xyxy[i][3]),
                        confidence=float(confs[i]),
                        class_id=c_id,
                        class_name=c_name,
                    )
                )

    return InferenceResponseModel(
        status="ok",
        detections=detections,
        inference_time_ms=inference_ms,
        device=DEVICE,
        gpu_name=GPU_NAME,
        frame_index=req.frame_index,
    )



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("worker_server:app", host="0.0.0.0", port=port, reload=False)
