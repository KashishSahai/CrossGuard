"""
CrossGuard AI Command Centre: System Overview
============================================
Provides overarching system health, active surveillance mode, video source
telemetry, subsystem operational readiness, and real-time event logs.
"""

from typing import Any, Dict, List
import streamlit as st

from core.config import PROJECT_SUBTITLE, PROJECT_TITLE, SUBSYSTEM_REGISTRY, VERSION
from dashboard.components import (
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_traffic_signal_head,
)


def render_overview_page() -> None:
    """Renders the comprehensive CrossGuard Command Centre overview page."""
    # Top Branding Banner
    st.markdown(
        f"""
        <div class="hud-card cyan-accent" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 1.45rem; font-weight: 800; letter-spacing: 0.08em; color: #F8FAFC;">
                        {PROJECT_TITLE}
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">
                        {PROJECT_SUBTITLE}
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span class="status-pill cyan">{VERSION}</span>
                    <span class="status-pill active">SYSTEM OPERATIONAL</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ingest / Video metadata from session state
    video_info = st.session_state.get("video_metadata")
    video_loaded = video_info is not None and video_info.get("is_valid", False)

    # 4-Column KPI Metric Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mode_val = "INGESTION READY" if video_loaded else "STANDBY"
        render_metric_card(
            label="PROCESSING MODE",
            value=mode_val,
            unit="",
            subtext="Real-time Surveillance Loop",
            status_color="cyan" if video_loaded else "muted",
        )

    with col2:
        if video_loaded:
            v_val = f"{video_info.get('width')}x{video_info.get('height')}"
            v_sub = f"{video_info.get('fps', 30)} FPS | {video_info.get('duration_formatted', '')}"
            v_color = "cyan"
        else:
            v_val = "NO STREAM"
            v_sub = "Awaiting Video Upload"
            v_color = "muted"

        render_metric_card(
            label="CAMERA / VIDEO FEED",
            value=v_val,
            unit="",
            subtext=v_sub,
            status_color=v_color,
        )

    with col3:
        render_metric_card(
            label="AI VISION PIPELINE",
            value="STAGE 2",
            unit="PENDING",
            subtext="YOLOv8/v11 + ByteTrack Unloaded",
            status_color="lavender",
        )

    with col4:
        render_metric_card(
            label="SIGNAL CONTROLLER",
            value="FALLBACK",
            unit="CYCLE",
            subtext="Pre-timed 60s Interlock Cycle",
            status_color="warning",
        )

    st.write("")

    # Main Grid: Left (Subsystem Diagnostics & Metrics), Right (Signal Preview & Event Log)
    left_col, right_col = st.columns([1.55, 1.0])

    with left_col:
        # Subsystem Operational Readiness Panel
        render_hud_header(
            title="Subsystem Architecture & Health Matrix",
            subtitle="Operational verification of research simulator components",
            badge_text="7 NODES REGISTERED",
            badge_type="cyan",
        )

        subsystem_rows_html = ""
        for sub in SUBSYSTEM_REGISTRY:
            if sub.default_status == "READY":
                badge_type = "active"
                status_text = "READY // STAGE 1"
            elif sub.default_status == "STAGE_PENDING":
                badge_type = "lavender"
                status_text = f"SCHEDULED // {sub.target_stage.upper()}"
            else:
                badge_type = "warning"
                status_text = f"STANDBY // {sub.target_stage.upper()}"

            subsystem_rows_html += f"""
            <tr>
                <td style="font-weight: 600; color: #F8FAFC;">{sub.name}</td>
                <td style="color: #94A3B8;">{sub.target_stage}</td>
                <td><span class="status-pill {badge_type}">{status_text}</span></td>
                <td style="color: #64748B; font-size: 0.75rem;">{sub.description}</td>
            </tr>
            """

        table_html = f"""
        <div class="hud-card" style="padding: 10px 14px;">
            <table class="tech-table">
                <thead>
                    <tr>
                        <th>SUBSYSTEM MODULE</th>
                        <th>TARGET</th>
                        <th>STATUS</th>
                        <th>OPERATIONAL SPECIFICATION</th>
                    </tr>
                </thead>
                <tbody>
                    {subsystem_rows_html}
                </tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.write("")

        # Live Pedestrian and Vehicle Metric Panels (Explicitly showing Stage 2 pending)
        render_hud_header(
            title="Live Intersection Telemetry",
            subtitle="Spatial occupancy and queue measurement (Stage 2 Integration)",
            badge_text="AWAITING VISION PIPELINE",
            badge_type="lavender",
        )

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            render_metric_card(
                label="PEDESTRIANS",
                value="--",
                unit="ACT",
                subtext="Stage 2 Pending",
                status_color="muted",
            )
        with mcol2:
            render_metric_card(
                label="VEHICLES",
                value="--",
                unit="CORR",
                subtext="Stage 2 Pending",
                status_color="muted",
            )
        with mcol3:
            render_metric_card(
                label="ZONE SATURATION",
                value="--",
                unit="%",
                subtext="Waiting Zone Polygons",
                status_color="muted",
            )
        with mcol4:
            render_metric_card(
                label="CROSSWALK LOAD",
                value="--",
                unit="%",
                subtext="Zebra Occupancy",
                status_color="muted",
            )

    with right_col:
        # Current Signal Status
        render_hud_header(
            title="Active Signal Phase",
            subtitle="Pre-timed fallback state",
            badge_text="CYCLE ACTIVE",
            badge_type="active",
        )

        render_traffic_signal_head(
            red_active=True,
            amber_active=False,
            green_active=False,
            ped_state="WAIT",
            countdown="18s",
        )

        st.write("")

        # Recent Event Panel
        render_hud_header(
            title="Recent Event Stream",
            subtitle="System audit & pipeline telemetry feed",
            badge_text="REAL-TIME",
            badge_type="cyan",
        )

        events: List[Dict[str, str]] = st.session_state.get(
            "event_logs",
            [
                {
                    "time": "00:00:01",
                    "level": "INFO",
                    "msg": "CrossGuard Command Centre initialized successfully.",
                },
                {
                    "time": "00:00:02",
                    "level": "SYSTEM",
                    "msg": "OpenCV video ingestion engine bound and ready.",
                },
                {
                    "time": "00:00:03",
                    "level": "INFO",
                    "msg": "PyTorch CPU runtime verified. CUDA acceleration standby.",
                },
                {
                    "time": "00:00:04",
                    "level": "NOTICE",
                    "msg": "Vision models and ByteTrack scheduled for Stage 2.",
                },
            ],
        )

        log_items_html = ""
        for ev in reversed(events[-6:]):
            color = (
                "#10B981"
                if ev["level"] == "INFO"
                else ("#00F0FF" if ev["level"] == "SYSTEM" else "#A78BFA")
            )
            log_items_html += f"""
            <div style="padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: var(--font-mono); font-size: 0.75rem; display: flex; gap: 8px;">
                <span style="color: #64748B;">[{ev['time']}]</span>
                <span style="color: {color}; font-weight: 700;">{ev['level']}</span>
                <span style="color: #CBD5E1;">{ev['msg']}</span>
            </div>
            """

        st.markdown(
            f"""
            <div class="hud-card" style="padding: 12px 16px;">
                {log_items_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
