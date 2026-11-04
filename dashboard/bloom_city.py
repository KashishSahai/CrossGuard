"""
CrossGuard - Bloom City Digital Twin Simulation Workspace
=========================================================
Hosts the microscopic 2D intersection digital twin with a distinctive pastel aesthetic.
Synchronizes vehicle queues, pedestrian platoons, and traffic signals directly with
real-time video detection and adaptive FSM decisions.
"""

import streamlit as st

from core.adaptive_controller import ControllerDecision, SignalPhase
from core.session_manager import SessionStatus, VideoSession
from core.spatial_analysis import SpatialSnapshot
from dashboard.components import (
    render_empty_state_card,
    render_html,
    render_hud_header,
    render_metric_card,
    render_status_badge,
)
from simulation.bloom_city_engine import BloomCityEngine


def render_bloom_city_page() -> None:
    """Renders the Bloom City Digital Twin intersection simulation workspace."""
    session: VideoSession = st.session_state["video_session"]

    render_hud_header(
        title="Bloom City // Digital Twin Intersection Simulator",
        subtitle="Microscopic pedestrian-priority urban traffic simulation workspace",
        badge_text="SYNCHRONIZED" if session.processed_frames_count > 0 else "AWAITING TELEMETRY",
        badge_type="active" if session.processed_frames_count > 0 else "neutral",
    )

    bloom_engine: BloomCityEngine = session.bloom_engine
    snapshot: SpatialSnapshot = session.latest_snapshot
    decision: ControllerDecision = session.latest_decision
    has_video = session.video_path is not None
    has_processed = session.processed_frames_count > 0

    # Bridge Connection Telemetry Bar
    if has_processed:
        bridge_badge = "active"
        bridge_status = f"VIDEO TELEMETRY LINKED (Frame #{session.current_frame_idx})"
    elif has_video:
        bridge_badge = "lavender"
        bridge_status = "VIDEO LOADED // AWAITING PROCESSING"
    else:
        bridge_badge = "warning"
        bridge_status = "NO VIDEO STREAM LOADED"

    render_html(
        f"""
        <div class="hud-card cyan-accent" style="padding: 12px 18px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="pulse-dot {'active' if has_processed else 'warning'}"></span>
                    <div>
                        <div style="font-family: var(--font-mono); font-size: 0.85rem; font-weight: 700; color: #F8FAFC;">
                            SIMULATION BRIDGE: {bridge_status}
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">
                            Sector 7: Bloom Ave & Aurora Blvd (Adaptive Pedestrian-Priority Node)
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: center; font-family: var(--font-mono); font-size: 0.78rem;">
                    <div><span style="color: #64748B;">CONTROLLER PHASE:</span> <strong style="color: #00F0FF;">{decision.current_phase.value}</strong></div>
                    <div><span style="color: #64748B;">COUNTDOWN:</span> <strong style="color: #A78BFA;">{decision.phase_remaining_seconds:.0f}s</strong></div>
                    <span class="status-pill {bridge_badge}">BRIDGE {session.status.value}</span>
                </div>
            </div>
        </div>
        """
    )

    # Informative banner if no video processed yet
    if not has_processed:
        render_empty_state_card(
            icon_symbol="🏙️",
            title="Awaiting Video Processing Telemetry",
            description="Bloom City's vehicles, pedestrians, and signals are directly driven by real-world detections. Upload footage and click 'Process Uploaded Video' in the AI Vision Monitor to synchronize the digital twin.",
            stage_tag="Standby Mode",
        )

    # Simulation Controls Toolbar
    scol1, scol2, scol3, scol4 = st.columns([1.2, 1.0, 1.0, 1.2])

    with scol1:
        step_sim = st.button("⏩ Advance Simulation Tick (0.5s)")

    with scol2:
        speed_mult = st.selectbox("Simulation Speed", ["1.0x (Real-time)", "0.5x (Slow-mo)", "2.0x (Fast)"], index=0)

    with scol3:
        night_toggle = st.toggle("🌙 Night Cyber Glow Mode", value=False)

    with scol4:
        if st.button("🔄 Reset Digital Twin"):
            bloom_engine.pedestrians.clear()
            bloom_engine.vehicles.clear()
            st.success("Bloom City entities reset.")

    # Execute simulation tick if requested
    dt = 0.5
    if speed_mult.startswith("0.5"):
        dt = 0.25
    elif speed_mult.startswith("2.0"):
        dt = 1.0

    if step_sim:
        bloom_engine.sync(
            snapshot,
            decision,
            delta_time=dt,
            night_mode=night_toggle,
            unique_vehicle_count=session.diagnostics.unique_vehicles_count,
            unique_pedestrian_count=session.diagnostics.unique_pedestrians_count,
        )
    else:
        # Keep engine synchronized with latest telemetry
        bloom_engine.sync(
            snapshot,
            decision,
            delta_time=0.05,
            night_mode=night_toggle,
            unique_vehicle_count=session.diagnostics.unique_vehicles_count,
            unique_pedestrian_count=session.diagnostics.unique_pedestrians_count,
        )

    st.write("")

    # Dedicated Simulation Viewport
    svg_canvas = bloom_engine.render_svg()
    render_html(
        f"""
        <div class="hud-corner-box" style="padding: 12px; display: flex; flex-direction: column; align-items: center;">
            {svg_canvas}
        </div>
        """
    )

    st.write("")

    # Real-Time Digital Twin Telemetry Counters
    render_hud_header(
        title="Digital Twin Kinetic Telemetry",
        subtitle="Simulated representative agents synchronized with computer vision spatial occupancy",
        badge_text="AGENT TELEMETRY",
        badge_type="cyan",
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card(
            label="SIMULATED VEHICLES",
            value=str(len(bloom_engine.vehicles)),
            unit="CARS",
            subtext=f"Queue density: {snapshot.queue_density_level}",
            status_color="cyan" if len(bloom_engine.vehicles) > 0 else "muted",
        )
    with m2:
        render_metric_card(
            label="SIMULATED PEDESTRIANS",
            value=str(len(bloom_engine.pedestrians)),
            unit="AGENTS",
            subtext=f"Waiting: {snapshot.waiting_pedestrians_count} | Crossing: {snapshot.crossing_pedestrians_count}",
            status_color="lavender" if len(bloom_engine.pedestrians) > 0 else "muted",
        )
    with m3:
        stopped_count = sum(1 for v in bloom_engine.vehicles if v.is_stopped)
        render_metric_card(
            label="VEHICLES STOPPED",
            value=str(stopped_count),
            unit="UNITS",
            subtext="At Stop Bar (Red / Pedestrian Conflict)",
            status_color="warning" if stopped_count > 0 else "active",
        )
    with m4:
        c_status = "OCCUPIED" if snapshot.crosswalk_occupied else "CLEAR"
        render_metric_card(
            label="CROSSING RESERVATION",
            value=c_status,
            unit="",
            subtext="Conflicting Vehicles Held" if snapshot.crosswalk_occupied else "Safe for Transit",
            status_color="warning" if snapshot.crosswalk_occupied else "active",
        )
