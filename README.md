# CrossGuard // Bloom City
> **AI-Powered Adaptive Pedestrian-Priority Traffic Management Research Simulator**

CrossGuard is an intelligent urban traffic management and surveillance research simulator designed to prioritize pedestrian safety, eliminate crossing starvation, and dynamically adjust signal intervals while optimizing vehicular throughput at complex signalized intersections.

---

## Processing Pipeline Architecture

```
[ Real Surveillance Video Feed (.mp4, .avi, .mov, .mkv) ]
                           │
                           ▼
[ Ultralytics YOLOv8n Multi-Class Object Detection ]
  - Detects: Person (0), Car (2), Motorcycle (3), Bus (5), Truck (7)
                           │
                           ▼
[ ByteTrack Multi-Target Association & Trajectory Tracking ]
  - Persistent Track IDs across frame occlusions
  - Base ground-contact centroid tracking
                           │
                           ▼
[ Spatial Zone Engine & Pedestrian Occupancy Detector ]
  - Point-in-polygon containment tests (Resolution Adaptive)
  - Waiting-zone accumulation & dwell duration tracking
  - Multi-frame crosswalk occupancy hysteresis
  - Vehicle approach queue density categorization
                           │
                           ▼
[ Adaptive Pedestrian-Priority Signal Controller (FSM) ]
  - Cases A through F: No peds, accumulation, group threshold, starvation prevention, clearance hold, clear observation
  - MUTCD safety intervals: Min green, Yellow, All-red, Walk, Flashing clearance
  - Explainable AI (XAI) transition audit stream
                           │
                           ▼
[ Bloom City Digital Twin Kinetic Intersection ]
  - Distinctive pastel aesthetic (Lilac, soft pink, slate, cream)
  - Animated miniature vehicles, pedestrian agents, and signal heads
  - Directly synchronized with real-time video telemetry
                           │
                           ▼
[ Longitudinal Telemetry Store & Traffic Analytics ]
  - Real session CSV data export
  - Interactive Plotly time-series and waiting distributions
                           │
                           ▼
[ GPU Performance Lab & Latency Benchmark Harness ]
  - PyTorch runtime inspection (CPU / CUDA autodetect)
  - Warmup cycles, CUDA synchronization, and statistical latency profiling
```

---

## Repository Structure

```
CrossGuard/
│
├── .streamlit/
│   └── config.toml                  # Dark command-centre theme & server config
│
├── assets/
│   └── styles/
│       └── theme.css                # Command Centre CSS design system & DOM resets
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Calibrated zones, signal timers, model thresholds, pastel palette
│
├── core/
│   ├── __init__.py
│   ├── session_manager.py          # VideoSession: Unified single source of truth across all pages
│   ├── config.py                    # Constants, registry, and page configuration
│   ├── video_processor.py           # Video validation & OpenCV stream metadata extraction
│   ├── hardware_info.py             # Genuine hardware & PyTorch telemetry probe
│   ├── spatial_analysis.py          # SpatialZoneEngine, point-in-polygon, dwell time, hysteresis
│   ├── adaptive_controller.py       # AdaptiveSignalController FSM (Cases A-F & Fail-Safe)
│   └── telemetry_manager.py         # In-memory telemetry aggregator, XAI events, CSV exporter
│
├── vision/
│   ├── __init__.py
│   └── vision_pipeline.py           # YOLOv8n + ByteTrack frame processor & HUD overlay renderer
│
├── simulation/
│   ├── __init__.py
│   └── bloom_city_engine.py         # Pastel 2D digital twin intersection & kinetic SVG renderer
│
├── benchmarking/
│   ├── __init__.py
│   └── benchmark_runner.py          # Latency & throughput benchmark harness with CUDA sync
│
├── dashboard/
│   ├── __init__.py
│   ├── components.py                # Reusable visual components with centralized HTML rendering
│   ├── overview.py                  # Page 1: Command Overview & Live Architecture Matrix
│   ├── vision_monitor.py            # Page 2: AI Vision Monitor & Calibration Controls
│   ├── bloom_city.py                # Page 3: Bloom City Simulation Workspace
│   ├── signal_control.py            # Page 4: Adaptive Signal Control & XAI Decision Hub
│   ├── analytics.py                 # Page 5: Traffic Analytics & Plotly Charts
│   └── gpu_performance.py           # Page 6: GPU Performance Lab & Benchmark Execution
│
├── tests/
│   ├── __init__.py
│   ├── test_integration_pipeline.py # End-to-end video lifecycle, zone calibration, FSM, and telemetry
│   ├── test_video_processor.py      # Format validation & OpenCV stream tests
│   ├── test_hardware_info.py        # Hardware telemetry verification
│   ├── test_spatial_analysis.py     # Polygon tests, dwell time, and hysteresis debouncing
│   ├── test_adaptive_controller.py  # FSM Cases A through F and fail-safe safety tests
│   ├── test_vision_pipeline.py      # Model caching, detection filtering, and frame processing
│   ├── test_bloom_city_sim.py       # Simulation entity synchronization and SVG rendering
│   ├── test_benchmark_runner.py     # Benchmark harness statistics and latency evaluations
│   ├── test_telemetry_manager.py    # Metric aggregation and CSV encoding
│   └── test_dashboard_pages.py      # AppTest navigation and HTML escaping verification
│
├── app.py                           # Application entry point & session state orchestrator
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## Launch Instructions

### Prerequisites
- Python 3.11
- Virtual environment installed at `.venv`

### Exact Windows Command Prompt (CMD) Command

Open Command Prompt (`cmd.exe`), navigate to the repository directory, and run:

```cmd
cd /d C:\Users\KIIT0001\Projects\CrossGuard
.venv\Scripts\activate.bat && streamlit run app.py
```

*Alternatively, execute directly using the virtual environment's Python binary:*

```cmd
C:\Users\KIIT0001\Projects\CrossGuard\.venv\Scripts\python.exe -m streamlit run app.py
```

### Exact Windows PowerShell Command

```powershell
cd C:\Users\KIIT0001\Projects\CrossGuard
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Once launched, access the application in your web browser at:
```
http://localhost:8501
```

---

## Operating Instructions & Workflow

### 1. Model Weights Download Behavior
- CrossGuard utilizes `yolov8n.pt` (Nano model, ~6.2 MB, 3.2M parameters) for lightweight real-time surveillance inference.
- Weights are automatically downloaded by Ultralytics on first execution and cached locally in the project root.
- The model is loaded lazily once and cached in memory across Streamlit reruns.

### 2. Video Footage Ingestion & Central Processing Session
- Navigate to **"2. AI Vision Monitor"**.
- Upload real intersection or pedestrian footage in `.mp4`, `.avi`, `.mov`, or `.mkv` format.
- Video stream header, frame count, resolution, and codec are automatically decoded and validated via OpenCV.
- The uploaded video initializes a central `VideoSession` in session state that serves as the single source of truth across all 6 dashboard modules.
- Choose your processing mode:
  - **⚡ Process Uploaded Video**: Runs incremental frame decoding across the video with real-time progress percentage, frame counters, and processing FPS.
  - **⏩ Step 10 Frames**: Processes the next sequential 10 frames.
  - **▶ Step 1 Frame**: Advances and processes a single frame for precise frame-by-frame inspection.
- Configure processing stride (e.g. 1 for full accuracy, 2 or 3 for faster CPU throughput while preserving real video timestamps).

### 3. Visual Spatial Zone Calibration & Presets
- Expand the **"📐 Spatial Zone Polygon Calibration"** drawer in the Vision Monitor.
- View the active video frame preview with normalized polygons rendered directly over the footage, along with detection centroids from the latest processed frame:
  - **North & South Waiting Curbs** (Lavender polygons & centroids)
  - **Zebra Crosswalk Corridor** (Cyan polygon)
  - **Vehicle Approach Queue** (Magenta polygon)
- **Presets Available**:
  - **✈️ Aerial Roadway Preset:** Calibrated for highway/arterial surveillance with horizontal roadway corridor ($Y \in [0.30, 0.68]$).
  - **🏙️ Standard Urban Crosswalk Preset:** Calibrated for perpendicular orthogonal city crosswalks.
- **Centroid Containment Diagnostics:** Real-time counters showing active targets inside the approach queue, inside waiting/crossing zones, or outside calibrated boundaries with an automatic warning if vehicles fall outside the approach queue.
- **Dual-Centroid Geometry:** Vehicles are evaluated using both geometric center and bottom bumper points to ensure full roadway corridor coverage. Pedestrians are evaluated using their base ground contact point $(c_x, y_2)$ for exact curb placement.

### 4. Small-Object Aerial Detection Tuning (1440p / 4K Footage)
- High-altitude aerial surveillance videos (e.g. $2560 \times 1440$) feature small pedestrian signatures (~$18 \times 56$ pixels).
- At standard YOLO $640 \times 640$ resizing, small pedestrians are compressed to ~$4 \times 14$ pixels, causing confidence scores to hover around $0.18 - 0.32$ and be discarded by conventional filters.
- CrossGuard introduces **configurable multi-resolution inference**:
  - **960px (Balanced - Recommended for Aerial):** Preserves feature maps while sustaining smooth CPU inference (~80–120ms).
  - **1280px (High-Precision Small Objects):** Maximum feature preservation for distant pedestrians and cyclists.
  - **640px (Fast CPU Standard):** Lightweight mode for low-resource environments.
- **Class-Specific Thresholding:** Independent confidence filtering allows pedestrian detection at $\ge 0.18$ while requiring $\ge 0.22$ for vehicles, eliminating false positive bleed.

### 5. Adaptive Controller Policy Engine (Finite State Machine)
The controller continuously evaluates spatial measurements against six formal cases:
- **CASE A — No Pedestrians Waiting:** Arterial vehicle green maintained subject to minimum green (12.0s).
- **CASE B — Small Platoon Arrival:** An accumulation interval (8.0s) allows a cluster to gather before initiating yellow.
- **CASE C — Group Threshold Reached:** When waiting pedestrians reach the configured threshold (&ge; 3 peds), priority phase change is scheduled immediately.
- **CASE D — Starvation Prevention (Max Wait Time):** If any pedestrian waits &ge; 25.0s, priority is granted regardless of group size.
- **CASE E — Crosswalk Occupancy Clearance Holds:** If pedestrians remain in the crosswalk during pedestrian clearance, conflicting vehicle green is held, and clearance is extended up to 3 times (+3.0s each).
- **CASE F — Verified Clearance Transition:** Releasing conflicting vehicular traffic requires 3 consecutive clear observations to eliminate sensor flicker.
- **FAIL-SAFE STOP — Conservative Interlock:** Optical occlusion or tracking disruption forces an immediate all-red stop.

### 6. Controller Demonstration & Scenario Testing Facility (Sandbox)
- Navigate to **"4. Adaptive Signal Control"** and select the **"🧪 FSM Scenario Testing & Demonstration Facility"** tab.
- This sandbox facility is **completely isolated** from the video processing session, allowing researchers to validate all 7 operational scenarios (Cases A through F and Fail-Safe Stop) without contaminating live session telemetry.
- Features interactive time-stepping (`+1.0s`, `+5.0s`), manual input override sliders, visual signal head display, and a real-time transition audit trail.

### 7. Bloom City Digital Twin
- Navigate to **"3. Bloom City Twin"** to observe the custom pastel overhead intersection.
- Directly driven by the central `VideoSession` telemetry:
  - Miniature vehicles queue up behind the stop bar during red phases and advance during green phases based on the actual detected queue density and cumulative unique vehicle tracks.
  - Pedestrian agents accumulate on the waiting curbs and cross the zebra stripes when protected walk is active.
- Toggle **"🌙 Night Cyber Glow Mode"** to switch between daylight pastel and night surveillance lighting.

### 8. Traffic Analytics & CSV Export
- Navigate to **"5. Traffic Analytics"** to inspect live longitudinal charts (Pedestrian Volume, Dwell Durations, Vehicle Queue, and Crossing Occupancy).
- View empirical summary KPIs: Total Unique Pedestrians Served, Max Waiting Time, Average Waiting Time, Peak Queue Length, Priority Calls, and Clearance Extensions.
- Review the fleet classification table breaking down multi-class detections, distinguishing temporal **Frame Detections** from persistent **Unique Track Entities** (Car, Motorcycle, Bus, Truck, Person).
- Click **"📥 Download Session CSV"** to export actual frame-by-frame measurements for external research analysis.

### 9. GPU Performance Lab & COA Hardware Deep Dive
- Navigate to **"6. GPU Performance Lab"**.
- Run statistical benchmarks on CPU or CUDA hardware measuring model load latency, mean/median inference time, throughput FPS, and VRAM allocation.
- **Computer Organization & Architecture (COA) Analysis:**
  - **SIMD vs. SIMT:** Analyzes host CPU AVX2/AVX-512 vector pipelines vs. GPU Streaming Multiprocessors executing 32-thread warps.
  - **Memory Hierarchy:** Contrasts DDR4/5 system bandwidth (40–85 GB/s) with GDDR6/HBM GPU memory buses (500–2,000 GB/s).
  - **PCIe Bus Latency:** Details the 11.06 MB transfer cost of 1440p frames and outlines pinned memory (`cudaHostAlloc`) and async CUDA streams (`cudaMemcpyAsync`).
  - **Tensor Cores:** Evaluates mixed-precision matrix multiply-accumulate hardware for FP16 and INT8 inference acceleration.
- **Google Colab GPU Acceleration Guide:** Includes copyable, self-contained Python benchmarking code for executing CrossGuard's harness on free NVIDIA T4 or A100 GPUs in Google Colab.

---

## Automated Test Suite

Run the full automated test suite (47 tests) using `pytest`:

```cmd
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

The test suite covers:
- **Session Data Consistency & Counters:** Safe initialization, persistent tracking IDs vs repeated frame detections, unique pedestrian and vehicle counters, configuration fingerprinting, and reset stability (`test_session_consistency.py`).
- **End-to-End Integration Pipeline:** Full lifecycle from video file ingestion to preview frame extraction, custom zone calibration, sequential frame stepping, full streaming processing, telemetry accumulation, Bloom City SVG sync, and clean session reset (`test_integration_pipeline.py`).
- **FSM Scenario Facility:** Isolated sandbox execution of Cases A through F and Fail-Safe Stop (`test_scenario_facility.py`).
- **Detection & Vision Tuning:** Small-object resolution scaling (640/960/1280) and class-specific confidence cutoff filtering (`test_detection_tuning.py`).
- **Adaptive Controller:** Verification of FSM Cases A, B, C, D, E, F, clearance extensions, and fail-safe shutdown (`test_adaptive_controller.py`).
- **Spatial Zone Engine:** Point-in-polygon containment, resolution scaling, dwell time accumulation, crosswalk hysteresis, and vehicle queue levels (`test_spatial_analysis.py`).
- **AI Vision Pipeline:** YOLO model caching, frame processing, bounding box geometry, and device assignment (`test_vision_pipeline.py`).
- **Bloom City Simulation:** Kinetic synchronization, vehicle queueing, and SVG generation (`test_bloom_city_sim.py`).
- **Benchmark Runner:** Latency profiling, statistical metrics, representative frame input, and CUDA synchronization (`test_benchmark_runner.py`).
- **Telemetry Logger:** Metric aggregation, event audit streams, and CSV encoding (`test_telemetry_manager.py`).
- **Video Processor:** Format validation, size limits, and OpenCV stream metadata extraction (`test_video_processor.py`).
- **Hardware Telemetry:** Authentic hardware probe for OS, CPU, RAM, and PyTorch runtime (`test_hardware_info.py`).
- **Streamlit AppTest:** Multipage routing and verification that zero raw HTML tags (`<tr>`, `<td>`, etc.) leak into markdown across all six pages (`test_dashboard_pages.py`).

---

## Research Simulation Safety Disclaimer

> [!WARNING]
> **RESEARCH SIMULATION USE ONLY:** CrossGuard – Bloom City is an experimental AI research simulator. It is not designed or certified for direct physical connection to municipal traffic signal controllers or public highway infrastructure. The timing parameters and policies are research prototypes. If camera occlusion or detection loss occurs, the simulator automatically initiates a conservative fail-safe state, halting all conflicting movements until the fault is resolved.
