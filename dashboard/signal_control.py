"""
CrossGuard Adaptive Signal Control Hub
======================================
Visualizes intersection traffic signal heads, phase transitions, pedestrian
demand thresholds, phase timing envelopes, and explainable arbitration rationale.
Driven directly by the central video processing session.
"""

import streamlit as st

from core.adaptive_controller import AdaptiveSignalController, ControllerDecision, SignalPhase
from core.session_manager import VideoSession
from core.spatial_analysis import SpatialSnapshot
from dashboard.components import (
    render_empty_state_card,
    render_html,
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_tech_table,
    render_traffic_signal_head,
)
from dashboard.scenario_facility import render_scenario_testing_facility


def render_signal_control_page() -> None:
    """Renders the Adaptive Signal Control policy and telemetry dashboard."""
    tab1, tab2 = st.tabs([
        "📹 Surveillance Video Actuation",
        "🧪 FSM Scenario Testing & Demonstration Facility",
    ])

    with tab1:
        _render_surveillance_controller()

    with tab2:
        render_scenario_testing_facility()


def _render_surveillance_controller() -> None:
    """Renders the live video-driven signal controller interface."""
    session: VideoSession = st.session_state["video_session"]
    controller: AdaptiveSignalController = session.controller
    snapshot: SpatialSnapshot = session.latest_snapshot
    decision: ControllerDecision = session.latest_decision
    has_processed = session.processed_frames_count > 0

    render_hud_header(
        title="Adaptive Signal Control // Policy Engine & Actuation",
        subtitle="Dynamic phase extension, safety clearance intervals, and pedestrian priority",
        badge_text="FSM CONTROLLER ACTIVE" if has_processed else "STANDBY // AWAITING STREAM",
        badge_type="active" if has_processed else "neutral",
    )

    # Controller Status Banner
    if controller.fault_active:
        status_badge = "danger"
        status_text = "CONSERVATIVE FAIL-SAFE ACTIVE"
    elif has_processed:
        status_badge = "active"
        status_text = "DYNAMIC ADAPTIVE PRIORITY ACTIVE"
    else:
        status_badge = "warning"
        status_text = "PRE-TIMED FALLBACK // AWAITING VIDEO INPUT"

    render_html(
        f"""
        <div class="hud-card cyan-accent" style="padding: 12px 18px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-family: var(--font-mono); font-size: 0.8rem; color: #00F0FF; letter-spacing: 0.08em;">
                        OPERATIONAL STATUS:
                    </span>
                    <span style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-left: 6px;">
                        {status_text}
                    </span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="status-pill lavender">CASE: {decision.decision_case}</span>
                    <span class="status-pill {status_badge}">FAIL-SAFE INTERLOCK ENGAGED</span>
                </div>
            </div>
        </div>
        """
    )

    if not has_processed:
        render_empty_state_card(
            icon_symbol="🚦",
            title="Awaiting Active Video Detections",
            description="The Adaptive Signal Controller policy engine requires real-time pedestrian waiting and crossing telemetry. Process video footage in the AI Vision Monitor to see dynamic priority transitions.",
            stage_tag="Standby Mode",
        )

    # Top Grid: Left (Visual Signal Head & Phase Ring), Right (Parameters & Decision Rationale)
    col_left, col_right = st.columns([1.0, 1.45])

    with col_left:
        render_hud_header(
            title="Signal Head Visualizer",
            subtitle="Real-time optical display component",
            badge_text=decision.current_phase.value,
            badge_type="cyan",
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

        # Phase Ring Schematic Card
        current_phase_name = decision.current_phase.value
        render_html(
            f"""
            <div class="hud-card" style="padding: 14px 16px;">
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; margin-bottom: 8px; text-transform: uppercase;">
                    PHASE RING & BARRIER ARCHITECTURE
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px; font-family: var(--font-mono); font-size: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; background: rgba(0,240,255,0.08); padding: 5px 8px; border-radius: 4px; border-left: 3px solid #00F0FF;">
                        <span style="color: #00F0FF; font-weight: 700;">CURRENT PHASE:</span>
                        <span style="color: #F8FAFC;">{current_phase_name}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 5px 8px; color: #94A3B8;">
                        <span>PHASE 2 (ARTERIAL N/S):</span>
                        <span>Min Green: {controller.config.min_vehicle_green:.0f}s | Max: {controller.config.max_vehicle_green:.0f}s</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 5px 8px; color: #94A3B8;">
                        <span>PEDESTRIAN PHASE 2P:</span>
                        <span>Walk: {controller.config.pedestrian_walk_duration:.0f}s | Clearance: {controller.config.pedestrian_clearance_duration:.0f}s</span>
                    </div>
                </div>
            </div>
            """
        )

        st.write("")

        # Emergency Fail-Safe Controls
        st.markdown("**Safety & Manual Override Controls**")
        ocol1, ocol2 = st.columns(2)
        with ocol1:
            if st.button("🚨 Trigger Safe Stop", help="Forces conservative all-red fail-safe stop."):
                controller.trigger_fault("Manual research simulation safety trigger", timestamp=snapshot.timestamp)
                session.latest_decision = controller.last_decision
                st.rerun()
        with ocol2:
            if st.button("✅ Clear Safe Stop", help="Restores normal adaptive controller cycle."):
                controller.clear_fault(timestamp=snapshot.timestamp)
                session.latest_decision = controller.last_decision
                st.rerun()

    with col_right:
        # Controller Data Reserves
        render_hud_header(
            title="Dynamic Telemetry Inputs",
            subtitle="Live sensor feed inputs from Vision Pipeline",
            badge_text="REAL TELEMETRY" if has_processed else "AWAITING INPUT",
            badge_type="active" if has_processed else "neutral",
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            render_metric_card(
                label="PEDESTRIAN DEMAND",
                value=f"{snapshot.waiting_pedestrians_count} Peds" if has_processed else "--",
                unit=f"(Threshold: {controller.config.pedestrian_group_threshold})" if has_processed else "",
                subtext="Waiting Zone Saturation",
                status_color="cyan" if has_processed and snapshot.waiting_pedestrians_count > 0 else "muted",
            )
            st.write("")
            render_metric_card(
                label="VEHICLE QUEUE DENSITY",
                value=f"{snapshot.approach_vehicles_count} Units" if has_processed else "--",
                unit=f"({snapshot.queue_density_level})" if has_processed else "",
                subtext="Corridor Approach Lanes",
                status_color="cyan" if has_processed and snapshot.approach_vehicles_count > 0 else "muted",
            )
        with dcol2:
            render_metric_card(
                label="MAX WAITING TIME",
                value=f"{snapshot.max_pedestrian_waiting_time:.1f} s" if has_processed else "--",
                unit=f"(Max: {controller.config.max_waiting_time:.0f}s)" if has_processed else "",
                subtext="Dwell Time Accumulator",
                status_color="lavender" if has_processed and snapshot.max_pedestrian_waiting_time > 0 else "muted",
            )
            st.write("")
            render_metric_card(
                label="EXTENSION COUNTER",
                value=f"{decision.clearance_extensions} / {controller.config.max_clearance_extensions}" if has_processed else "--",
                unit="CYC",
                subtext="Dynamic Clearance Holds",
                status_color="warning" if decision.clearance_extensions > 0 else "muted",
            )

        st.write("")

        # Explainable Decision Rationale Panel (XAI)
        render_html(
            f"""
            <div class="hud-card lavender-accent" style="padding: 16px 20px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-family: var(--font-mono); font-size: 0.85rem; font-weight: 700; color: #A78BFA;">
                        EXPLAINABLE AI (XAI) DECISION RATIONALE
                    </span>
                    <span class="status-pill lavender">{decision.decision_case}</span>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.82rem; color: #F8FAFC; line-height: 1.55; margin-bottom: 8px;">
                    {decision.reason}
                </div>
                <div style="display: flex; gap: 14px; font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <div><span>PREVIOUS:</span> <strong style="color: #F8FAFC;">{decision.previous_phase.value}</strong></div>
                    <div><span>ELAPSED:</span> <strong style="color: #00F0FF;">{decision.phase_elapsed_seconds:.1f}s</strong></div>
                    <div><span>REMAINING:</span> <strong style="color: #10B981;">{decision.phase_remaining_seconds:.1f}s</strong></div>
                    <div><span>SAFETY:</span> <strong style="color: #EF4444;">{'FAULT STOP' if controller.fault_active else 'ACTIVE SAFE'}</strong></div>
                </div>
            </div>
            """
        )

        # Phase Timers and Configuration Envelope
        render_hud_header(
            title="Controller Timing Envelope (Safety Bounds)",
            subtitle="MUTCD pedestrian safety standards & clearance intervals",
            badge_text="SAFETY BOUNDS",
            badge_type="cyan",
        )

        table_rows = [
            [
                "Minimum Vehicular Green",
                f"{controller.config.min_vehicle_green:.1f} s",
                "Fixed Safety Floor",
                "MUTCD Driver Expectancy Standard",
            ],
            [
                "Maximum Vehicular Green",
                f"{controller.config.max_vehicle_green:.1f} s",
                "Dynamic Ceiling",
                "Prevents Pedestrian Corridor Starvation",
            ],
            [
                "Pedestrian Walk Interval",
                f"{controller.config.pedestrian_walk_duration:.1f} s",
                "Protected Walk",
                "Guaranteed Initial Crossing Entry",
            ],
            [
                "Flashing Clearance Interval",
                f"{controller.config.pedestrian_clearance_duration:.1f} s",
                f"Dynamic +{controller.config.extension_seconds:.0f}s Extensions",
                "Crossing Occupancy Hysteresis Protected",
            ],
            [
                "Yellow Vehicular Change",
                f"{controller.config.vehicle_yellow_duration:.1f} s",
                "Non-Overridable",
                "Kinematic Stopping Zone Clearance",
            ],
            [
                "All-Red Clearance Interval",
                f"{controller.config.all_red_clearance_duration:.1f} s",
                "Non-Overridable",
                "Intersection Evacuation Buffer",
            ],
        ]

        render_tech_table(
            headers=["INTERVAL PARAMETER", "CONFIGURED VALUE", "DYNAMIC ADAPTATION", "SAFETY SPECIFICATION"],
            rows=table_rows,
        )
