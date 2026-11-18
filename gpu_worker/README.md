# CrossGuard Remote GPU Worker Layer

This package enables distributed hardware execution for CrossGuard. While CrossGuard's authoritative simulation, ByteTrack tracking, homography mapping, spatial analysis, and NEMA dual-ring adaptive controller execute locally, the compute-intensive YOLOv8n object detection workload can be offloaded to a connected CUDA worker (such as an NVIDIA T4 running on Google Colab or any cloud GPU instance).

## End User vs. Deployment Operator Roles

### End User Experience
- The end user simply opens CrossGuard and selects **GPU (CUDA)** (or **Auto**) under **Processing Device**.
- No Colab setup, notebook upload, tunnel configuration, or URL pasting is required from the user.
- CrossGuard automatically discovers and connects to the active worker backend and streams genuine CUDA detections.

### Project Owner / Demo Operator Deployment
- A compatible CUDA worker must be running at the configured endpoint or registered via GitHub Gist.
- **Dynamic Registry (Recommended for Colab)**:
  Configure `CROSSGUARD_GPU_REGISTRY_GIST_ID` in `.env`:
  ```env
  CROSSGUARD_GPU_REGISTRY_GIST_ID=your_github_gist_id
  ```
- **Static Fallback**:
  Configure `CROSSGUARD_GPU_WORKER_URL` in `.env`:
  ```env
  CROSSGUARD_GPU_WORKER_URL=https://your-tunnel-url.trycloudflare.com
  ```

> **Important Deployment Note**:
> Google Colab is an ephemeral compute environment. The worker URL changes whenever the runtime or Cloudflare tunnel restarts, but CrossGuard discovers the newly registered URL automatically via the GitHub Gist registry without requiring manual edits to `.env`.

---

## Dynamic Registry Architecture

```text
Colab GPU Worker Starts (Tesla T4)
        │
        ▼
Cloudflare Tunnel Generated (https://<random>.trycloudflare.com)
        │
        ▼
Worker Publishes Registration → Fixed GitHub Gist (crossguard_worker.json)
        │                       (Heartbeat thread refreshes timestamp every 60s)
        │
        ▼
Local CrossGuard Reads Fixed Gist
        │
        ├── 1. Valid & Fresh (< 300s TTL)?  ──► Use Registry URL
        │
        └── 2. Stale / Offline / Error?     ──► Fall back to CROSSGUARD_GPU_WORKER_URL
                                                (or None if unconfigured)
        │
        ▼
Lightweight /health Check
        │
        ▼
Remote CUDA Inference (/detect)
```

### URL Resolution Precedence
The authoritative choke-point `get_configured_worker_url()` resolves in this order:
1. **Explicit parameter**: Overrides all external sources (`source: explicit`).
2. **Dynamic Registry**: Valid, online, fresh Gist record (`source: registry`).
3. **Static Environment Fallback**: `CROSSGUARD_GPU_WORKER_URL` from `.env` (`source: env`).
4. **None**: Neither configured (`source: none`).

### Stale Worker Protection
A registry entry is treated as stale if `updated_at + TTL < current_time` (default TTL: 300 seconds / 5 minutes). When stale or marked `status: "offline"`, CrossGuard rejects the entry and falls back to `CROSSGUARD_GPU_WORKER_URL`.

---

## Architecture Flow

```text
Local CrossGuard (Authoritative)
      │
      │ 1. Frame Base64 + Detection Thresholds
      ↓
Remote CUDA GPU Worker (FastAPI + CUDA)
      │
      │ 2. YOLOv8n CUDA Inference (torch.cuda)
      ↓
Detections + Latency Metadata
      │
      ↓
Local CrossGuard
      │
      ├── 3. ByteTrack Multi-Object Tracking (Sequential State Preserved)
      ├── 4. Camera Homography Calibration
      ├── 5. 16-Zone Spatial Analysis
      ├── 6. Telemetry Aggregation
      ├── 7. NEMA Dual-Ring Adaptive Controller (FSM)
      ├── 8. Bloom City Simulation Bridge
      └── 9. Research Analytics
```

---

## Protocol Specification

The worker exposes standard HTTP REST endpoints:

### `GET /health`
Queries the worker's hardware availability and readiness.

**Response (200 OK):**
```json
{
  "status": "ok",
  "cuda_available": true,
  "gpu_name": "Tesla T4",
  "gpu_memory_total_mb": 15109.0,
  "gpu_memory_allocated_mb": 128.5,
  "yolo_loaded": true,
  "model_name": "yolov8n.pt",
  "torch_version": "2.1.0+cu121",
  "device": "cuda"
}
```

### `POST /detect`
Processes a single video frame with YOLOv8n on CUDA.

**Request:**
```json
{
  "image_b64": "<base64 encoded JPEG>",
  "confidence_threshold": 0.35,
  "iou_threshold": 0.5,
  "frame_index": 42
}
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "detections": [
    {
      "x1": 425.2,
      "y1": 512.0,
      "x2": 580.4,
      "y2": 690.1,
      "confidence": 0.89,
      "class_id": 2,
      "class_name": "car"
    }
  ],
  "inference_time_ms": 11.8,
  "device": "cuda",
  "gpu_name": "Tesla T4",
  "frame_index": 42
}
```
