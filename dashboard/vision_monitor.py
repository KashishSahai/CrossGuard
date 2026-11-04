"""
CrossGuard AI Vision Monitor
=============================
Video-driven surveillance processing workspace. Ingests user traffic footage,
provides visual spatial zone polygon calibration on actual video frames,
executes YOLOv8n detection and ByteTrack tracking, and exposes live pipeline diagnostics.
"""

import os
import tempfile
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from config.settings import NormalizedZoneConfig
from core.config import SUPPORTED_VIDEO_EXTENSIONS
from core.session_manager import SessionStatus, VideoSession
from core.video_processor import extract_video_metadata, save_temp_video_file, validate_video_file
from dashboard.components import (
    render_empty_state_card,
    render_html,
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_tech_table,
)


def render_vision_monitor_page() -> None:
    """Renders the central video-processing and AI Vision Monitor workspace."""
    session: VideoSession = st.session_state["video_session"]

    render_hud_header(
        title="AI Vision Monitor // Multi-Target Tracking & Spatial Occupancy",
        subtitle="Ultralytics YOLOv8n detection, ByteTrack tracking, and live crosswalk zone analysis",
        badge_text=session.status.value.replace("_", " "),
        badge_type="active" if session.status in (SessionStatus.COMPLETED, SessionStatus.PROCESSING) else "warning",
    )

    # 1. Footage Ingestion Section
    render_html(
        """
        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #00F0FF; margin-bottom: 8px; letter-spacing: 0.08em;">
            STEP 1: SURVEILLANCE FOOTAGE INGESTION
        </div>
        """
    )

    formats_label = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
    uploaded_file = st.file_uploader(
        f"Select a traffic surveillance video ({formats_label})",
        type=[ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS],
        help="Upload actual intersection or pedestrian crossing footage. Video resolution and codecs are inspected via OpenCV.",
    )

    # Process user uploaded video
    if uploaded_file is not None:
        is_valid, validation_msg = validate_video_file(uploaded_file.name, uploaded_file.size)
        if not is_valid:
            st.error(f"Validation Failure: {validation_msg}")
        else:
            current_loaded = st.session_state.get("uploaded_filename")
            if current_loaded != uploaded_file.name:
                try:
                    saved_path = save_temp_video_file(uploaded_file)
                    extracted_meta = extract_video_metadata(saved_path)
                    if extracted_meta.get("is_valid", False):
                        st.session_state["uploaded_filename"] = uploaded_file.name
                        session.load_video(saved_path, extracted_meta)
                        session.telemetry_manager.add_event(
                            "INFO",
                            f"Uploaded video '{uploaded_file.name}' ({extracted_meta['resolution']} @ {extracted_meta['fps']} FPS).",
                        )
                        st.success(f"Video loaded: {uploaded_file.name} | {extracted_meta['resolution']} | {extracted_meta['duration_formatted']}")
                    else:
                        st.error(f"OpenCV Stream Decoding Error: {extracted_meta.get('error')}")
                except Exception as ex:
                    st.error(f"Error buffering video stream: {str(ex)}")

    # Show warning if no video is loaded
    if not session.video_path or not os.path.exists(session.video_path):
        render_empty_state_card(
            icon_symbol="📹",
            title="No Surveillance Video Uploaded",
            description="Upload an MP4, AVI, MOV, or MKV surveillance video to begin AI detection, ByteTrack tracking, and adaptive signal control.",
            stage_tag="Awaiting File Upload",
        )
        return

    metadata = session.video_metadata or {}

    # Display Video Metadata Bar
    render_html(
        f"""
        <div class="hud-card" style="padding: 10px 18px; margin-bottom: 16px; background: rgba(13, 19, 34, 0.95);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; font-family: var(--font-mono); font-size: 0.8rem;">
                <div><span style="color: #64748B;">SOURCE:</span> <strong style="color: #00F0FF;">{metadata.get('file_name')}</strong></div>
                <div><span style="color: #64748B;">RESOLUTION:</span> <span style="color: #F8FAFC;">{metadata.get('resolution')}</span></div>
                <div><span style="color: #64748B;">FRAME RATE:</span> <span style="color: #10B981;">{metadata.get('fps')} FPS</span></div>
                <div><span style="color: #64748B;">DURATION:</span> <span style="color: #A78BFA;">{metadata.get('duration_formatted')}</span></div>
                <div><span style="color: #64748B;">TOTAL FRAMES:</span> <span style="color: #F8FAFC;">{metadata.get('frame_count')}</span></div>
                <div><span style="color: #64748B;">SESSION ID:</span> <span style="color: #00F0FF;">{session.session_id}</span></div>
            </div>
        </div>
        """
    )

    # 2. Visual Spatial Zone Calibration Expander
    with st.expander("📐 Visual Spatial Zone Calibration (Drawn over actual video frame)", expanded=False):
        st.markdown(
            "Verify and adjust normalized spatial boundaries (0.0 to 1.0) overlaid directly on the uploaded video frame. "
            "Detections centroids are rendered with diagnostic containment checks."
        )

        cal_frame = session.get_calibration_preview_frame(draw_detections=True)
        if cal_frame is not None:
            cal_rgb = cv2.cvtColor(cal_frame, cv2.COLOR_BGR2RGB)
            st.image(cal_rgb, caption="Active Zone Overlay with Detection Centroids on Video Frame", use_container_width=True)

        # Centroid Containment Diagnostics
        if session.latest_result and session.latest_result.tracked_detections:
            dets = session.latest_result.tracked_detections
            in_app = sum(1 for d in dets if d.assigned_zone == "APPROACH_QUEUE")
            in_cross = sum(1 for d in dets if d.assigned_zone == "CROSSWALK")
            in_wait = sum(1 for d in dets if "WAITING" in d.assigned_zone)
            unassigned = sum(1 for d in dets if d.assigned_zone == "UNASSIGNED")

            c_diag1, c_diag2, c_diag3, c_diag4 = st.columns(4)
            c_diag1.metric("In Approach Queue", f"{in_app} vehs")
            c_diag2.metric("In Zebra Crossing", f"{in_cross} peds")
            c_diag3.metric("In Waiting Curbs", f"{in_wait} peds")
            c_diag4.metric("Outside Any Zone", f"{unassigned} targets")

            if unassigned > 0 and in_app == 0 and any(not d.is_pedestrian for d in dets):
                st.warning(
                    "⚠️ **Calibration Overlap Warning:** Vehicles are detected outside calibrated zones! "
                    "Adjust the Approach Range sliders below so that vehicular bounding-box centroids/bumpers fall within the magenta approach corridor."
                )

        # Preset Calibration Buttons
        st.markdown("**Zone Topology Presets**")
        pcol1, pcol2 = st.columns(2)
        apply_aerial = pcol1.button("✈️ Load Aerial Roadway Preset", help="Calibrated for horizontal traffic corridor (Y: 0.30 - 0.68)")
        apply_urban = pcol2.button("🏙️ Load Standard Urban Crosswalk", help="Calibrated for orthogonal urban intersection")

        if apply_aerial:
            session.zone_config.waiting_zone_north = [(0.05, 0.45), (0.30, 0.45), (0.30, 0.78), (0.05, 0.78)]
            session.zone_config.waiting_zone_south = [(0.70, 0.42), (0.96, 0.42), (0.96, 0.75), (0.70, 0.75)]
            session.zone_config.crosswalk_polygon = [(0.28, 0.42), (0.72, 0.42), (0.72, 0.76), (0.28, 0.76)]
            session.zone_config.vehicle_approach = [(0.20, 0.30), (0.80, 0.30), (0.80, 0.68), (0.20, 0.68)]
            w = metadata.get("width", 1920)
            h = metadata.get("height", 1080)
            session.spatial_engine.recalculate_pixel_polygons(w, h)
            session._cached_first_frame = None
            st.success("Loaded Aerial Roadway Preset!")
            st.rerun()

        if apply_urban:
            session.zone_config.waiting_zone_north = [(0.15, 0.45), (0.35, 0.45), (0.35, 0.58), (0.15, 0.58)]
            session.zone_config.waiting_zone_south = [(0.65, 0.45), (0.85, 0.45), (0.85, 0.58), (0.65, 0.58)]
            session.zone_config.crosswalk_polygon = [(0.35, 0.42), (0.65, 0.42), (0.65, 0.62), (0.35, 0.62)]
            session.zone_config.vehicle_approach = [(0.30, 0.05), (0.70, 0.05), (0.70, 0.40), (0.30, 0.40)]
            w = metadata.get("width", 1920)
            h = metadata.get("height", 1080)
            session.spatial_engine.recalculate_pixel_polygons(w, h)
            session._cached_first_frame = None
            st.success("Loaded Standard Urban Crosswalk Preset!")
            st.rerun()

        st.markdown("**Manual Boundary Adjustments [0.0 - 1.0 Normalized Space]**")
        zcol1, zcol2 = st.columns(2)
        with zcol1:
            st.markdown("**North Waiting Zone (Lavender)**")
            wn_x = st.slider(
                "North X-Range", 0.0, 1.0,
                (float(session.zone_config.waiting_zone_north[0][0]), float(session.zone_config.waiting_zone_north[1][0])),
                0.01, key="wn_x_slider"
            )
            wn_y = st.slider(
                "North Y-Range", 0.0, 1.0,
                (float(session.zone_config.waiting_zone_north[0][1]), float(session.zone_config.waiting_zone_north[2][1])),
                0.01, key="wn_y_slider"
            )

            st.markdown("**South Waiting Zone (Lavender)**")
            ws_x = st.slider(
                "South X-Range", 0.0, 1.0,
                (float(session.zone_config.waiting_zone_south[0][0]), float(session.zone_config.waiting_zone_south[1][0])),
                0.01, key="ws_x_slider"
            )
            ws_y = st.slider(
                "South Y-Range", 0.0, 1.0,
                (float(session.zone_config.waiting_zone_south[0][1]), float(session.zone_config.waiting_zone_south[2][1])),
                0.01, key="ws_y_slider"
            )

        with zcol2:
            st.markdown("**Zebra Crossing Corridor (Cyan)**")
            cw_x = st.slider(
                "Crosswalk X-Range", 0.0, 1.0,
                (float(session.zone_config.crosswalk_polygon[0][0]), float(session.zone_config.crosswalk_polygon[1][0])),
                0.01, key="cw_x_slider"
            )
            cw_y = st.slider(
                "Crosswalk Y-Range", 0.0, 1.0,
                (float(session.zone_config.crosswalk_polygon[0][1]), float(session.zone_config.crosswalk_polygon[2][1])),
                0.01, key="cw_y_slider"
            )

            st.markdown("**Vehicle Approach Queue (Magenta)**")
            va_x = st.slider(
                "Approach X-Range", 0.0, 1.0,
                (float(session.zone_config.vehicle_approach[0][0]), float(session.zone_config.vehicle_approach[1][0])),
                0.01, key="va_x_slider"
            )
            va_y = st.slider(
                "Approach Y-Range", 0.0, 1.0,
                (float(session.zone_config.vehicle_approach[0][1]), float(session.zone_config.vehicle_approach[2][1])),
                0.01, key="va_y_slider"
            )

        if st.button("Apply Spatial Calibration", help="Recalculates pixel polygons for active video resolution."):
            session.zone_config.waiting_zone_north = [
                (wn_x[0], wn_y[0]), (wn_x[1], wn_y[0]), (wn_x[1], wn_y[1]), (wn_x[0], wn_y[1])
            ]
            session.zone_config.waiting_zone_south = [
                (ws_x[0], ws_y[0]), (ws_x[1], ws_y[0]), (ws_x[1], ws_y[1]), (ws_x[0], ws_y[1])
            ]
            session.zone_config.crosswalk_polygon = [
                (cw_x[0], cw_y[0]), (cw_x[1], cw_y[0]), (cw_x[1], cw_y[1]), (cw_x[0], cw_y[1])
            ]
            session.zone_config.vehicle_approach = [
                (va_x[0], va_y[0]), (va_x[1], va_y[0]), (va_x[1], va_y[1]), (va_x[0], va_y[1])
            ]
            w = metadata.get("width", 1920)
            h = metadata.get("height", 1080)
            session.spatial_engine.recalculate_pixel_polygons(w, h)
            session._cached_first_frame = None  # Invalidate cached preview to refresh
            st.success("Spatial calibration saved and applied!")
            st.rerun()

    st.write("")

    # 3. Processing Controls & Configuration
    render_html(
        """
        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #00F0FF; margin-bottom: 8px; letter-spacing: 0.08em;">
            STEP 2: PIPELINE EXECUTION & SHARED SESSION PROCESSING
        </div>
        """
    )

    pcfg1, pcfg2, pcfg3, pcfg4 = st.columns([1.1, 1.2, 1.1, 1.0])
    with pcfg1:
        stride_opts = [1, 2, 3, 5]
        stride_idx = stride_opts.index(session.frame_stride) if session.frame_stride in stride_opts else 2
        stride_sel = st.selectbox(
            "Processing Stride",
            stride_opts,
            index=stride_idx,
            help="1: Full sampling. 2 or 3: Fast CPU (stride 3 recommended default for CPU real-time). 5: Maximum speed.",
        )
        session.frame_stride = stride_sel

    with pcfg2:
        res_opts = [960, 640, 1280]
        profile_names = {
            640: "Fast (640px) - Low Latency CPU",
            960: "Balanced (960px) - Recommended for Aerial",
            1280: "Accurate (1280px) - Small-Target Precision",
        }
        curr_res_idx = res_opts.index(session.vision_config.inference_size) if session.vision_config.inference_size in res_opts else 0
        imgsz_sel = st.selectbox(
            "Inference Profile",
            res_opts,
            index=curr_res_idx,
            format_func=lambda x: profile_names.get(x, f"{x}px"),
            help="Configurable inference profile: Fast (640px), Balanced (960px), or Accurate (1280px).",
        )
        session.vision_config.inference_size = imgsz_sel

    with pcfg3:
        ped_conf_slider = st.slider(
            "Pedestrian Conf Cutoff",
            0.10, 0.40,
            float(session.vision_config.pedestrian_conf_threshold),
            0.02,
            help="Specialized threshold for small pedestrian detections (class 0)."
        )
        session.vision_config.pedestrian_conf_threshold = ped_conf_slider
        session.vision_config.confidence_threshold = min(ped_conf_slider, session.vision_config.confidence_threshold)

    with pcfg4:
        st.write("")
        st.write("")
        reset_btn = st.button("🔄 Reset Session", help="Clears detections and restarts session pointer.")
        if reset_btn:
            session.reset_processing()
            st.success("Session state reset.")
            st.rerun()

    # Inform user if current session results are stale due to modified configuration
    if session.is_stale:
        st.warning(
            "⚠️ **Configuration Modified:** Detection resolution, stride, confidence, or zones were altered since the last run. "
            "Click **⚡ Process Uploaded Video** to refresh session results with the updated parameters."
        )

    # Action Buttons: Process Entire Video vs. Batch Stepping
    act_col1, act_col2, act_col3, act_col4 = st.columns([1.5, 1.0, 1.0, 1.0])

    process_all_btn = act_col1.button("⚡ Process Uploaded Video", type="primary", help="Processes the video sequentially with real-time progress.")
    process_chunk_btn = act_col2.button("⏩ Step 10 Frames", help="Processes the next 10 frames incrementally.")
    process_single_btn = act_col3.button("▶ Step 1 Frame", help="Processes the single next frame.")
    stop_btn = act_col4.button("⏹ Stop Processing", help="Halts active video processing.")

    if stop_btn:
        session.status = SessionStatus.STOPPED
        st.info("Processing halted by user.")
        st.rerun()

    # Execute full video processing with progress bar
    if process_all_btn:
        if session.is_stale:
            session.reset_processing()

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(current_f, total_f, pct):
            progress_bar.progress(pct)
            status_text.markdown(f"**Processing Video:** Source Frame {current_f} / {total_f} ({int(pct*100)}%) — Sent to YOLO: {session.processed_frames_count} frames")

        processed = session.process_video_full(
            stride=session.frame_stride,
            progress_callback=update_progress,
        )
        progress_bar.empty()
        status_text.empty()
        st.success(f"Processing complete! Processed {processed} inference frames across {session.current_frame_idx} video frames in {session.total_proc_seconds:.1f}s.")
        st.rerun()

    elif process_chunk_btn:
        with st.spinner("Processing next 10 frames..."):
            processed = session.process_frames_chunk(num_frames=10, stride=session.frame_stride)
            st.success(f"Processed {processed} frames (Current index: {session.current_frame_idx}).")
            st.rerun()

    elif process_single_btn:
        with st.spinner("Processing single frame..."):
            processed = session.process_frames_chunk(num_frames=1, stride=session.frame_stride)
            st.rerun()

    # Progress Telemetry Bar
    total_vid_frames = session.total_source_frames or (metadata.get("frame_count", 1) or 1)
    pct_done = min(100.0, (session.current_frame_idx / total_vid_frames) * 100.0)
    inf_fps_str = f"{session.effective_inference_fps:.1f} FPS" if session.effective_inference_fps > 0 else "--"
    inf_lat_str = f"{session.mean_inference_latency_ms:.1f} ms" if session.mean_inference_latency_ms > 0 else "--"
    render_html(
        f"""
        <div style="background: rgba(13, 19, 34, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 10px 16px; margin: 12px 0; font-family: var(--font-mono); font-size: 0.8rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div><span style="color:#64748B;">SOURCE STREAM:</span> <strong style="color:#00F0FF;">Frame {session.current_frame_idx} / {total_vid_frames} ({pct_done:.1f}%)</strong></div>
            <div><span style="color:#64748B;">INFERENCE FRAMES:</span> <strong style="color:#F8FAFC;">{session.processed_frames_count} frames</strong> <span style="color:#64748B;">(Stride {session.frame_stride})</span></div>
            <div><span style="color:#64748B;">INFERENCE SPEED:</span> <strong style="color:#10B981;">{inf_fps_str}</strong> <span style="color:#64748B;">({inf_lat_str})</span></div>
            <div><span style="color:#64748B;">STATUS:</span> <span class="status-pill cyan">{session.status.value}</span></div>
        </div>
        """
    )

    # 4. Viewport Display: Raw vs. Annotated AI Overlay
    vcol1, vcol2 = st.columns([1.1, 1.1])

    with vcol1:
        render_html(
            """
            <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; margin-bottom: 6px;">
                RAW SURVEILLANCE FEED PREVIEW
            </div>
            """
        )
        st.video(session.video_path)

    with vcol2:
        render_html(
            """
            <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00F0FF; margin-bottom: 6px;">
                YOLOv8n DETECTION & BYTETRACK OVERLAY
            </div>
            """
        )
        if session.latest_result is not None:
            annotated_rgb = cv2.cvtColor(session.latest_result.annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(
                annotated_rgb,
                caption=f"Frame #{session.latest_result.frame_index} | Inference: {session.latest_result.inference_time_ms:.1f} ms | FPS: {session.latest_result.fps:.1f} | Detections: {len(session.latest_result.tracked_detections)}",
                use_container_width=True,
            )
        else:
            render_empty_state_card(
                icon_symbol="👁️",
                title="No Processed Frames Yet",
                description="Click 'Process Uploaded Video' or 'Step 10 Frames' to begin YOLO object detection and ByteTrack tracking.",
                stage_tag="Ready to Process",
            )

    st.write("")

    # 5. Active Telemetry Counters
    render_hud_header(
        title="Live Surveillance Spatial Telemetry",
        subtitle="Measurements derived from active video frame detections",
        badge_text="FRAME METRICS",
        badge_type="cyan",
    )

    snap = session.latest_snapshot
    res = session.latest_result

    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
    with tcol1:
        fps_disp = f"{res.fps:.1f}" if res else "--"
        render_metric_card(
            label="INFERENCE SPEED",
            value=fps_disp,
            unit="FPS",
            subtext=f"{res.inference_time_ms:.1f} ms latency" if res else "Awaiting inference",
            status_color="cyan" if res else "muted",
        )
    with tcol2:
        render_metric_card(
            label="WAITING PEDS",
            value=str(snap.waiting_pedestrians_count),
            unit="PEDS",
            subtext=f"Max Dwell: {snap.max_pedestrian_waiting_time:.1f}s",
            status_color="cyan" if snap.waiting_pedestrians_count > 0 else "muted",
        )
    with tcol3:
        render_metric_card(
            label="CROSSING PEDS",
            value=str(snap.crossing_pedestrians_count),
            unit="ACT",
            subtext="Inside Zebra Corridor",
            status_color="lavender" if snap.crossing_pedestrians_count > 0 else "muted",
        )
    with tcol4:
        render_metric_card(
            label="APPROACH VEHICLES",
            value=str(snap.approach_vehicles_count),
            unit="VEH",
            subtext=f"Queue: {snap.queue_density_level}",
            status_color="cyan" if snap.approach_vehicles_count > 0 else "muted",
        )
    with tcol5:
        c_status = "OCCUPIED" if snap.crosswalk_occupied else "CLEAR"
        render_metric_card(
            label="CROSSWALK OCCUPANCY",
            value=c_status,
            unit="",
            subtext="Hysteresis Debounced",
            status_color="warning" if snap.crosswalk_occupied else "active",
        )

    st.write("")

    # 6. Active Target Tracking Buffer Table
    table_rows = []
    active_peds = sum(1 for det in res.tracked_detections if det.is_pedestrian) if res else 0
    active_vehs = sum(1 for det in res.tracked_detections if not det.is_pedestrian) if res else 0
    total_active = len(res.tracked_detections) if res else 0

    if res and res.tracked_detections:
        for det in res.tracked_detections:
            x1, y1, x2, y2 = det.bbox
            dwell_str = f"{det.dwell_time_seconds:.1f}s" if det.dwell_time_seconds > 0 else "--"
            cls_badge = "cyan" if det.is_pedestrian else "lavender"
            table_rows.append([
                f'<strong style="color: #00F0FF;">#{det.track_id}</strong>',
                f'<span class="status-pill {cls_badge}">{det.class_name.upper()}</span>',
                f'<span style="color: #10B981;">{det.confidence:.2f}</span>',
                f'<span style="font-family: var(--font-mono); color: #94A3B8;">[{x1}, {y1}, {x2}, {y2}]</span>',
                f'<span style="color: #F8FAFC;">{det.assigned_zone}</span>',
                f'<span style="color: #A78BFA;">{dwell_str}</span>',
            ])
    elif res and len(res.tracked_detections) == 0:
        table_rows.append([
            f'<span colspan="6" style="color: #94A3B8;">Zero objects detected in frame #{res.frame_index}. Video frame is clear of targets.</span>',
            "", "", "", "", ""
        ])
    elif session.status == SessionStatus.FAILED:
        table_rows.append([
            f'<span colspan="6" style="color: #EF4444;">Pipeline Inference Error: {session.error_message or "Execution failed"}. Check diagnostics below.</span>',
            "", "", "", "", ""
        ])
    else:
        table_rows.append([
            '<span colspan="6" style="color: #64748B;">Awaiting video processing. Click Process Uploaded Video to inspect targets.</span>',
            "", "", "", "", ""
        ])

    caption_str = (
        f"BYTETRACK TARGET BUFFER — FRAME TARGETS: {total_active} ({active_peds} Peds, {active_vehs} Vehs) | "
        f"SESSION UNIQUE ENTITIES: {len(session.diagnostics.unique_track_ids)} "
        f"({session.unique_pedestrians_count} Peds, {session.unique_vehicles_count} Vehs)"
    )

    render_tech_table(
        headers=["TRACK ID", "CLASS", "CONFIDENCE", "BOUNDING BOX", "ASSIGNED ZONE", "WAITING DWELL"],
        rows=table_rows,
        caption=caption_str,
    )

    st.write("")

    # 7. Pipeline Diagnostics Panel
    with st.expander("🔍 Pipeline Health & Detection Diagnostics", expanded=False):
        diag = session.diagnostics
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.metric("Total Detections", diag.total_detections_count)
            st.metric("Zero-Detection Frames", diag.zero_detection_frames)
        with dcol2:
            st.metric("Unique Track IDs", len(diag.unique_track_ids))
            st.metric("Average Confidence", f"{diag.avg_confidence:.2f}")
        with dcol3:
            st.metric("Person Detections", diag.detections_by_class.get("person", 0))
            st.metric("Car Detections", diag.detections_by_class.get("car", 0))
        with dcol4:
            st.metric("Truck/Bus Detections", diag.detections_by_class.get("truck", 0) + diag.detections_by_class.get("bus", 0))
            st.metric("Inference Errors", len(diag.inference_errors))
