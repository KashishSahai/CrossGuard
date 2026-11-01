"""
CrossGuard AI Vision Monitor
=============================
Handles real traffic video ingestion, OpenCV header inspection, format validation,
video playback preview, and hosts the central YOLO/ByteTrack detection overlay viewport.
"""

from datetime import datetime
from typing import Optional
import streamlit as st

from core.config import SUPPORTED_VIDEO_EXTENSIONS
from core.video_processor import (
    extract_video_metadata,
    save_temp_video_file,
    validate_video_file,
)
from dashboard.components import (
    render_empty_state_card,
    render_hud_header,
    render_metric_card,
    render_status_badge,
)


def render_vision_monitor_page() -> None:
    """Renders the AI Vision Monitor workspace."""
    render_hud_header(
        title="AI Vision Monitor // Surveillance & Spatial Occupancy",
        subtitle="Real-time multi-target tracking and crosswalk zone occupancy analysis",
        badge_text="VISION ENGINE // STAGE 2 PENDING",
        badge_type="lavender",
    )

    # Notice regarding Stage 2 implementation
    st.markdown(
        """
        <div class="hud-card lavender-accent" style="padding: 12px 18px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    <strong style="color: #A78BFA;">STAGE 1 NOTICE:</strong>
                    Video ingestion, format validation, and playback preview are active.
                    YOLOv8/v11 object detection, ByteTrack tracking, and zone polygons are scheduled for <strong>Stage 2</strong>.
                </div>
                <span class="status-pill lavender">MODEL WEIGHTS UNLOADED</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ingestion Controls Section
    st.markdown(
        """
        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #00F0FF; margin-bottom: 8px; letter-spacing: 0.08em;">
            STEP 1: SURVEILLANCE FOOTAGE INGESTION
        </div>
        """,
        unsafe_allow_html=True,
    )

    formats_label = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
    uploaded_file = st.file_uploader(
        f"Select a traffic surveillance video ({formats_label})",
        type=[ext.lstrip(".") for ext in SUPPORTED_VIDEO_EXTENSIONS],
        help="Upload genuine intersection or pedestrian crossing footage for pipeline processing.",
    )

    video_path: Optional[str] = st.session_state.get("video_path")
    metadata = st.session_state.get("video_metadata")

    if uploaded_file is not None:
        # Validate format and size
        is_valid, validation_msg = validate_video_file(
            uploaded_file.name, uploaded_file.size
        )

        if not is_valid:
            st.error(f"Validation Failure: {validation_msg}")
        else:
            # Check if this is a newly uploaded file
            current_loaded = st.session_state.get("uploaded_filename")
            if current_loaded != uploaded_file.name:
                try:
                    saved_path = save_temp_video_file(uploaded_file)
                    extracted_meta = extract_video_metadata(saved_path)

                    if extracted_meta.get("is_valid", False):
                        st.session_state["video_path"] = saved_path
                        st.session_state["video_metadata"] = extracted_meta
                        st.session_state["uploaded_filename"] = uploaded_file.name
                        video_path = saved_path
                        metadata = extracted_meta

                        # Record event in audit log
                        if "event_logs" not in st.session_state:
                            st.session_state["event_logs"] = []
                        st.session_state["event_logs"].append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "level": "INFO",
                                "msg": f"Ingested video '{uploaded_file.name}' ({extracted_meta['resolution']} @ {extracted_meta['fps']} FPS).",
                            }
                        )
                        st.success(
                            f"Footage loaded: {uploaded_file.name} | Resolution: {extracted_meta['resolution']} | Duration: {extracted_meta['duration_formatted']}"
                        )
                    else:
                        st.error(f"OpenCV Stream Error: {extracted_meta.get('error')}")
                except Exception as ex:
                    st.error(f"Error buffering video stream: {str(ex)}")

    # Display Metadata Bar if video is loaded
    if metadata and metadata.get("is_valid", False):
        st.markdown(
            f"""
            <div class="hud-card" style="padding: 10px 18px; margin-bottom: 16px; background: rgba(13, 19, 34, 0.95);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; font-family: var(--font-mono); font-size: 0.8rem;">
                    <div><span style="color: #64748B;">SOURCE:</span> <strong style="color: #00F0FF;">{metadata.get('file_name')}</strong></div>
                    <div><span style="color: #64748B;">RESOLUTION:</span> <span style="color: #F8FAFC;">{metadata.get('resolution')}</span></div>
                    <div><span style="color: #64748B;">FPS:</span> <span style="color: #10B981;">{metadata.get('fps')}</span></div>
                    <div><span style="color: #64748B;">DURATION:</span> <span style="color: #A78BFA;">{metadata.get('duration_formatted')}</span></div>
                    <div><span style="color: #64748B;">FRAMES:</span> <span style="color: #F8FAFC;">{metadata.get('frame_count')}</span></div>
                    <div><span style="color: #64748B;">SIZE:</span> <span style="color: #94A3B8;">{metadata.get('file_size_mb')} MB</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Central Viewport: Real Preview vs. Future Detection Overlay
    st.markdown(
        """
        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: #00F0FF; margin-bottom: 8px; letter-spacing: 0.08em;">
            STEP 2: SURVEILLANCE VIEWPORT & YOLO DETECTION CANVAS
        </div>
        """,
        unsafe_allow_html=True,
    )

    vcol1, vcol2 = st.columns([1.1, 1.1])

    with vcol1:
        st.markdown(
            """
            <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; margin-bottom: 6px;">
                RAW SURVEILLANCE FEED PREVIEW
            </div>
            """,
            unsafe_allow_html=True,
        )
        if video_path and metadata and metadata.get("is_valid", False):
            st.video(video_path)
        else:
            render_empty_state_card(
                icon_symbol="📹",
                title="No Surveillance Video Loaded",
                description="Upload an MP4, AVI, or MOV file using the file selector above to preview raw footage.",
                stage_tag="Awaiting File Upload",
            )

    with vcol2:
        st.markdown(
            """
            <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #00F0FF; margin-bottom: 6px;">
                AI DETECTION & BYTETRACK OVERLAY CANVAS
            </div>
            """,
            unsafe_allow_html=True,
        )
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"""
            <div class="surveillance-viewport">
                <div class="camera-tag">CAM-04 // NORTH ARTERIAL CROSSWALK</div>
                <div class="camera-timestamp">{timestamp_str}</div>
                <div class="viewport-crosshair"></div>
                <div style="text-align: center; z-index: 2; padding: 20px;">
                    <div style="font-family: var(--font-mono); font-size: 1.05rem; font-weight: 700; color: #00F0FF; margin-bottom: 8px; letter-spacing: 0.05em;">
                        [ + ] YOLO DETECTION & BYTETRACK CANVAS
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8; max-width: 380px; margin: 0 auto 12px auto; line-height: 1.4;">
                        Bounding box overlays, spatial trajectory vectors, and curb holding-zone polygon masks will render here in Stage 2.
                    </div>
                    <span class="status-pill lavender">DETECTION ENGINE STAGE 2 PENDING</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Spatial Tracking & Occupancy Status Placeholders
    render_hud_header(
        title="Spatial Zone & Trajectory Telemetry",
        subtitle="Pedestrian waiting-zone saturation and crosswalk occupancy counters",
        badge_text="TELEMETRY PLACEHOLDERS",
        badge_type="neutral",
    )

    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns(5)
    with tcol1:
        render_metric_card(
            label="ACTIVE TRACK IDs",
            value="--",
            unit="OBJS",
            subtext="Stage 2 Tracker",
            status_color="muted",
        )
    with tcol2:
        render_metric_card(
            label="PEDESTRIANS",
            value="--",
            unit="COUNT",
            subtext="Stage 2 Detector",
            status_color="muted",
        )
    with tcol3:
        render_metric_card(
            label="VEHICLES",
            value="--",
            unit="COUNT",
            subtext="Stage 2 Detector",
            status_color="muted",
        )
    with tcol4:
        render_metric_card(
            label="NORTH CURB ZONE",
            value="--",
            unit="PEDS",
            subtext="Polygon Zone A",
            status_color="muted",
        )
    with tcol5:
        render_metric_card(
            label="CROSSWALK LOAD",
            value="--",
            unit="%",
            subtext="Zebra Zone Active",
            status_color="muted",
        )

    st.write("")

    # Active Target Tracking Table Placeholder
    st.markdown(
        """
        <div class="hud-card" style="padding: 14px 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; color: #F8FAFC;">
                    ACTIVE TARGET TRACKING LOG (BYTETRACK BUFFER)
                </span>
                <span class="status-pill lavender">0 ACTIVE TRACKS // STAGE 2</span>
            </div>
            <table class="tech-table">
                <thead>
                    <tr>
                        <th>TRACK ID</th>
                        <th>CLASS</th>
                        <th>CONFIDENCE</th>
                        <th>BOUNDING BOX [X1, Y1, X2, Y2]</th>
                        <th>SPEED VECTOR</th>
                        <th>ZONE ASSIGNMENT</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td colspan="6" style="text-align: center; color: #64748B; padding: 22px;">
                            Vision pipeline stage 2 awaiting YOLOv8/v11 weights and ByteTrack tracker initialization. No targets detected.
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
