# CrossGuard // Bloom City
> **AI-Powered Adaptive Pedestrian-Priority Traffic Management Research Simulator**

CrossGuard is an intelligent urban traffic management and surveillance simulator designed to prioritize pedestrian safety and dynamic crossing demand while optimizing vehicular throughput at complex signalized intersections.

---

## Processing Pipeline Vision

```
[ Real Traffic Video Feed ]
            │
            ▼
[ YOLOv8 / YOLOv11 Multi-Object Detection ]
            │
            ▼
[ ByteTrack Multi-Target Association & Trajectory Tracking ]
            │
            ▼
[ Pedestrian Waiting-Zone & Crosswalk Spatial Occupancy Engine ]
            │
            ▼
[ Adaptive Signal Priority Controller (XAI Policy Engine) ]
            │
            ▼
[ Bloom City Digital Twin Kinetic Animated Intersection ]
```

---

## Stage 1: The CrossGuard AI Command Centre

Stage 1 establishes the command center architecture, visual design system, and multi-page application foundation:

- **Futuristic AI Surveillance Command Centre Theme:**
  - Deep navy and charcoal background (`#0A0E17`, `#0D1322`, `#111827`).
  - Dark blue-grey elevated panels with illuminated borders and subtle glow.
  - Cyber accents: Neon cyan (`#00F0FF`), electric lavender (`#A78BFA`), subtle magenta (`#F43F5E`).
  - Standard signal optics: Red (`#EF4444`), Amber (`#F59E0B`), Green (`#10B981`).
  - Compact, information-dense layout optimized for standard laptop displays.

- **Six Core Dashboard Sections:**
  1. **Command Overview (`dashboard/overview.py`):** System branding, live mode indicators, architectural health diagnostics, live signal preview, and real-time audit event stream.
  2. **AI Vision Monitor (`dashboard/vision_monitor.py`):** Real surveillance video upload (`.mp4`, `.avi`, `.mov`, `.mkv`), OpenCV metadata extraction (resolution, FPS, duration, codec), video playback preview, central HUD detection canvas with corner targeting brackets, and tracking telemetry placeholders.
  3. **Bloom City Digital Twin (`dashboard/bloom_city.py`):** Dedicated simulation workspace, connection telemetry to simulator socket bridge, perspective selection (Overhead 2D, Isometric 3D, Driver/Pedestrian eye levels), environmental lighting dials, and strict non-mock safety assertions.
  4. **Adaptive Signal Control (`dashboard/signal_control.py`):** Realistic digital traffic signal head component (with glowing red, amber, green lenses, pedestrian Walk/Don't Walk symbol, and countdown display), phase ring diagram, safety clearance timing bounds, and explainable decision rationale.
  5. **Traffic Analytics (`dashboard/analytics.py`):** Temporal filters and four Plotly command-centre analytical charts with cyber gridlines and informative empty states for pedestrian density, waiting-time distributions, vehicle queues, and crosswalk occupancy.
  6. **GPU Performance Lab (`dashboard/gpu_performance.py`):** Real system hardware diagnostics (CPU cores, RAM, Python/OpenCV/PyTorch versions, CUDA acceleration availability), model benchmark configuration harness for Stage 4, and real-time surveillance SLA latency budget table (&le; 30 ms end-to-end target).

---

## Application Structure

```
CrossGuard/
│
├── .streamlit/
│   └── config.toml                  # Dark theme & server config
│
├── assets/
│   └── styles/
│       └── theme.css                # Command Centre CSS design system
│
├── core/
│   ├── __init__.py
│   ├── config.py                    # Constants, registry, and page configuration
│   ├── video_processor.py           # Video validation & OpenCV metadata extraction
│   └── hardware_info.py             # Genuine hardware & PyTorch telemetry probe
│
├── dashboard/
│   ├── __init__.py
│   ├── components.py                # Reusable visual components (HUD, signals, metrics, plots)
│   ├── overview.py                  # Page 1: Command Overview
│   ├── vision_monitor.py            # Page 2: AI Vision Monitor
│   ├── bloom_city.py                # Page 3: Bloom City Simulation Workspace
│   ├── signal_control.py            # Page 4: Adaptive Signal Control
│   ├── analytics.py                 # Page 5: Traffic Analytics
│   └── gpu_performance.py           # Page 6: GPU Performance Lab
│
├── tests/
│   ├── __init__.py
│   ├── test_video_processor.py      # Format validation & OpenCV stream tests
│   ├── test_hardware_info.py        # Genuine hardware inspection tests
│   └── test_dashboard_pages.py      # AppTest navigation and callable verification
│
├── app.py                           # Lightweight application orchestrator
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## Launch Instructions

### Prerequisites
- Python 3.11
- Virtual environment with dependencies installed (located at `.venv`)

### Exact Windows Command Prompt (CMD) Command

Open Command Prompt (`cmd.exe`), navigate to the project directory, activate the virtual environment, and launch Streamlit:

```cmd
cd /d C:\Users\KIIT0001\Projects\CrossGuard
.venv\Scripts\activate.bat && streamlit run app.py
```

*Alternatively, as a single direct command without activating:*

```cmd
C:\Users\KIIT0001\Projects\CrossGuard\.venv\Scripts\python.exe -m streamlit run app.py
```

### Exact Windows PowerShell Command

```powershell
cd C:\Users\KIIT0001\Projects\CrossGuard
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Once launched, the command centre will be available at:
```
http://localhost:8501
```

---

## Subsystem Architectural Readiness Matrix

| Subsystem Module | Target Stage | Current Status | Notes |
| :--- | :--- | :--- | :--- |
| **Command Centre UI & Shell** | Stage 1 | **ACTIVE** | Full 6-section navigation, dark theme, top status bar |
| **Video Ingestion & Validation** | Stage 1 | **ACTIVE** | Format verification, OpenCV metadata extraction, preview |
| **Hardware Telemetry Probe** | Stage 1 | **ACTIVE** | Real CPU/RAM/PyTorch/CUDA runtime diagnostics |
| **YOLO Object Detector** | Stage 2 | **STAGE 2 PENDING** | Ultralytics model weights and inference loop scheduled |
| **ByteTrack Spatial Association** | Stage 2 | **STAGE 2 PENDING** | Kalman filter and trajectory association scheduled |
| **Waiting Zone & Crosswalk Engine** | Stage 2 | **STAGE 2 PENDING** | Curb holding polygons and occupancy ratio scheduled |
| **Adaptive Signal Controller** | Stage 3 | **STANDBY** | Dynamic extension & preemption policy engine scheduled |
| **Bloom City Simulator Bridge** | Stage 3 | **STANDBY** | WebSocket / shared memory intersection twin scheduled |
| **Hardware Benchmark Suite** | Stage 4 | **STANDBY** | Model FPS, TensorRT latency, and throughput testing |

---

## Verification & Testing

Run the automated test suite with `pytest`:

```cmd
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Test coverage includes:
- Supported format validation (`.mp4`, `.avi`, `.mov`, `.mkv`) and rejection of invalid extensions.
- Oversized file constraint checks.
- Real OpenCV video metadata decoding and error handling for corrupt/missing files.
- Real hardware telemetry probing without fake metrics.
- Comprehensive `AppTest` execution navigating across all six dashboard sections to verify zero runtime exceptions.

---

## Remaining Features for Stage 2 (Roadmap)

1. **YOLO Detection Pipeline Integration:**
   - Integrate pre-trained `yolov8n.pt` / `yolov11n.pt` weights for pedestrian (`class 0`) and vehicular (`class 2, 3, 5, 7`) detection.
   - Implement frame extraction generator and asynchronous inference worker.
2. **ByteTrack Multi-Target Association:**
   - Implement Kalman-filter based bounding box tracking to assign persistent IDs to pedestrians and vehicles across frames.
   - Calculate speed and trajectory vectors to anticipate crossing intent.
3. **Spatial Zone Polygons & Homography:**
   - Define interactive North and South curb waiting-zone polygons.
   - Define crosswalk zebra zone polygon.
   - Implement point-in-polygon tests to calculate active waiting-zone saturation and crossing occupancy.
4. **Real-Time Detection Overlay Canvas:**
   - Draw annotated bounding boxes, track IDs, velocity trails, and color-coded zone occupancy masks directly onto surveillance frames in the AI Vision Monitor.
5. **Telemetry Event Store Integration:**
   - Connect live detection counts to populate the Traffic Analytics historical charts dynamically.
