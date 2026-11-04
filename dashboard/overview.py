"""
CrossGuard AI Command Centre: System Overview
============================================
Provides overarching system health, active surveillance mode, video source
telemetry, subsystem operational readiness, and real-time event logs.
Uses clean centralized HTML rendering to eliminate raw markup escaping.
"""

from typing import Any, Dict, List
import streamlit as st

from core.adaptive_controller import ControllerDecision, SignalPhase
from core.config import PROJECT_SUBTITLE, PROJECT_TITLE, SUBSYSTEM_REGISTRY, VERSION
from core.session_manager import SessionStatus, VideoSession
from core.spatial_analysis import SpatialSnapshot
from dashboard.components import (
    render_html,
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_tech_table,
    render_traffic_signal_head,
)


def render_overview_page() -> None:
    """Renders the comprehensive CrossGuard Command Centre overview page."""
    session: VideoSession = st.session_state["video_session"]

    # Top Branding Banner
    status_label = session.status.value.replace("_", " ")
    badge_style = "active" if session.status in (SessionStatus.COMPLETED, SessionStatus.PROCESSING) else "warning"

    render_html(
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
                    <span class="status-pill {badge_style}">{status_label}</span>
                </div>
            </div>
        </div>
        """
    )

    video_info = session.video_metadata
    video_loaded = video_info is not None and session.video_path is not None
    snapshot: SpatialSnapshot = session.latest_snapshot
    decision: ControllerDecision = session.latest_decision
    telemetry = session.telemetry_manager

    # 4-Column KPI Metric Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if session.status == SessionStatus.COMPLETED:
            mode_val = "COMPLETED"
            sub_val = f"Processed {session.processed_frames_count} frames"
            m_col = "active"
        elif session.status == SessionStatus.PROCESSING:
            mode_val = "PROCESSING"
            sub_val = f"Frame {session.current_frame_idx}"
            m_col = "cyan"
        elif session.status == SessionStatus.VIDEO_UPLOADED:
            mode_val = "VIDEO LOADED"
            sub_val = "Ready to Process"
            m_col = "lavender"
        else:
            mode_val = "AWAITING UPLOAD"
            sub_val = "Surveillance Stream Idle"
            m_col = "muted"

        render_metric_card(
            label="SESSION STATE",
            value=mode_val,
            unit="",
            subtext=sub_val,
            status_color=m_col,
        )

    with col2:
        if video_loaded:
            v_val = f"{video_info.get('width')}x{video_info.get('height')}"
            v_sub = f"{video_info.get('fps', 30)} FPS | {video_info.get('duration_formatted', '')}"
            v_color = "cyan"
        else:
            v_val = "NO STREAM"
            v_sub = "Upload Video in Vision Monitor"
            v_color = "muted"

        render_metric_card(
            label="VIDEO FEED",
            value=v_val,
            unit="",
            subtext=v_sub,
            status_color=v_color,
        )

    with col3:
        if session.processed_frames_count > 0:
            pipe_val = "ACTIVE"
            pipe_sub = f"Unique Tracks: {len(session.diagnostics.unique_track_ids)}"
            p_col = "cyan"
        elif video_loaded:
            pipe_val = "STANDBY"
            pipe_sub = "YOLO & ByteTrack Initialized"
            p_col = "lavender"
        else:
            pipe_val = "UNLINKED"
            pipe_sub = "Awaiting Video Stream"
            p_col = "muted"

        render_metric_card(
            label="AI VISION PIPELINE",
            value=pipe_val,
            unit="YOLOv8n",
            subtext=pipe_sub,
            status_color=p_col,
        )

    with col4:
        phase_label = decision.current_phase.value.replace("_", " ")
        render_metric_card(
            label="ADAPTIVE CONTROLLER",
            value=phase_label,
            unit="",
            subtext=f"Rule: {decision.decision_case} ({decision.phase_remaining_seconds:.0f}s left)",
            status_color="active" if decision.current_phase == SignalPhase.VEHICLE_GREEN else "warning",
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

        subsystem_rows = []
        for sub in SUBSYSTEM_REGISTRY:
            if sub.id == "video_ingest":
                status_text = "READY // ACTIVE" if video_loaded else "AWAITING UPLOAD"
                badge_type = "active" if video_loaded else "warning"
            elif sub.id in ("yolo_detector", "bytetrack_tracker", "zone_analyzer"):
                if session.processed_frames_count > 0:
                    status_text = "PROCESSING // ACTIVE"
                    badge_type = "active"
                elif video_loaded:
                    status_text = "READY // STANDBY"
                    badge_type = "cyan"
                else:
                    status_text = "STANDBY"
                    badge_type = "neutral"
            elif sub.id == "adaptive_controller":
                status_text = "ACTIVE // ADAPTIVE" if session.processed_frames_count > 0 else "STANDBY"
                badge_type = "active" if session.processed_frames_count > 0 else "lavender"
            elif sub.id == "bloom_city_bridge":
                status_text = "SYNCHRONIZED" if session.processed_frames_count > 0 else "STANDBY"
                badge_type = "active" if session.processed_frames_count > 0 else "lavender"
            else:
                status_text = "ACTIVE" if len(telemetry.records) > 0 else "STANDBY"
                badge_type = "active" if len(telemetry.records) > 0 else "neutral"

            subsystem_rows.append([
                f'<strong style="color: #F8FAFC;">{sub.name}</strong>',
                f'<span style="color: #94A3B8;">{sub.target_stage}</span>',
                f'<span class="status-pill {badge_type}">{status_text}</span>',
                f'<span style="color: #64748B; font-size: 0.75rem;">{sub.description}</span>',
            ])

        render_tech_table(
            headers=["SUBSYSTEM MODULE", "STAGE", "STATUS", "OPERATIONAL SPECIFICATION"],
            rows=subsystem_rows,
        )

        st.write("")

        # Live Pedestrian and Vehicle Metric Panels
        render_hud_header(
            title="Live Intersection Telemetry",
            subtitle="Real-time spatial occupancy and queue measurement from AI vision pipeline",
            badge_text="SESSION METRICS" if session.processed_frames_count > 0 else "AWAITING RUN",
            badge_type="cyan" if session.processed_frames_count > 0 else "neutral",
        )

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        has_processed = session.processed_frames_count > 0
        with mcol1:
            render_metric_card(
                label="WAITING PEDS",
                value=str(snapshot.waiting_pedestrians_count) if has_processed else "--",
                unit="ACT",
                subtext=f"Max Dwell: {snapshot.max_pedestrian_waiting_time:.1f}s" if has_processed else "Awaiting Video Run",
                status_color="cyan" if has_processed and snapshot.waiting_pedestrians_count > 0 else "muted",
            )
        with mcol2:
            render_metric_card(
                label="CROSSING PEDS",
                value=str(snapshot.crossing_pedestrians_count) if has_processed else "--",
                unit="ACT",
                subtext="In Zebra Corridor" if has_processed else "Awaiting Video Run",
                status_color="lavender" if has_processed and snapshot.crossing_pedestrians_count > 0 else "muted",
            )
        with mcol3:
            render_metric_card(
                label="APPROACH VEHICLES",
                value=str(snapshot.approach_vehicles_count) if has_processed else "--",
                unit="VEH",
                subtext=f"Queue: {snapshot.queue_density_level}" if has_processed else "Awaiting Video Run",
                status_color="cyan" if has_processed and snapshot.approach_vehicles_count > 0 else "muted",
            )
        with mcol4:
            cross_val = ("OCCUPIED" if snapshot.crosswalk_occupied else "CLEAR") if has_processed else "--"
            cross_col = ("warning" if snapshot.crosswalk_occupied else "active") if has_processed else "muted"
            render_metric_card(
                label="CROSSWALK STATUS",
                value=cross_val,
                unit="",
                subtext="Hysteresis Debounced" if has_processed else "Awaiting Video Run",
                status_color=cross_col,
            )

    with right_col:
        # Current Signal Status
        render_hud_header(
            title="Active Signal Phase",
            subtitle="Adaptive controller state & optical preview",
            badge_text=decision.current_phase.value,
            badge_type="active" if decision.current_phase == SignalPhase.VEHICLE_GREEN else "warning",
        )

        is_red = decision.current_phase in (
            SignalPhase.ALL_RED_CLEARANCE,
            SignalPhase.PEDESTRIAN_WALK,
            SignalPhase.PEDESTRIAN_CLEARANCE,
            SignalPhase.FAIL_SAFE_STOP,
        )
        is_yellow = decision.current_phase == SignalPhase.VEHICLE_YELLOW
        is_green = decision.current_phase == SignalPhase.VEHICLE_GREEN
        ped_state = (
            "WALK"
            if decision.current_phase in (SignalPhase.PEDESTRIAN_WALK, SignalPhase.PEDESTRIAN_CLEARANCE)
            else "DONT_WALK"
        )

        render_traffic_signal_head(
            red_active=is_red,
            amber_active=is_yellow,
            green_active=is_green,
            ped_state=ped_state,
            countdown=f"{decision.phase_remaining_seconds:.0f}s",
            phase_label=decision.current_phase.value.replace("_", " "),
        )

        st.write("")

        # Recent Event Panel
        render_hud_header(
            title="Recent Event Stream",
            subtitle="Real-time controller transitions & detection events",
            badge_text="LIVE AUDIT",
            badge_type="cyan",
        )

        events: List[Dict[str, str]] = telemetry.events
        if not events:
            events = [
                {"time": "00:00:01", "level": "SYSTEM", "msg": "CrossGuard initialized."},
            ]

        log_items_html = ""
        for ev in reversed(events[-6:]):
            color = (
                "#10B981"
                if ev["level"] == "INFO"
                else ("#00F0FF" if ev["level"] == "SYSTEM" else ("#F59E0B" if ev["level"] == "DECISION" else "#A78BFA"))
            )
            log_items_html += f"""
            <div style="padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: var(--font-mono); font-size: 0.75rem; display: flex; gap: 8px;">
                <span style="color: #64748B;">[{ev['time']}]</span>
                <span style="color: {color}; font-weight: 700;">{ev['level']}</span>
                <span style="color: #CBD5E1;">{ev['msg']}</span>
            </div>
            """

        render_html(
            f"""
            <div class="hud-card" style="padding: 12px 16px;">
                {log_items_html}
            </div>
            """
        )
