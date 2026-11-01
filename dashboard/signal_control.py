"""
CrossGuard Adaptive Signal Control Hub
======================================
Visualizes intersection traffic signal heads, phase transitions, pedestrian
demand thresholds, phase timing envelopes, and explainable arbitration rationale.
"""

import streamlit as st
from dashboard.components import (
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_traffic_signal_head,
)


def render_signal_control_page() -> None:
    """Renders the Adaptive Signal Control policy and telemetry dashboard."""
    render_hud_header(
        title="Adaptive Signal Control // Policy Engine & Actuation",
        subtitle="Dynamic phase extension, safety clearance intervals, and pedestrian priority",
        badge_text="CONTROLLER // STAGE 3 PENDING",
        badge_type="warning",
    )

    # Controller Status Banner
    st.markdown(
        """
        <div class="hud-card cyan-accent" style="padding: 12px 18px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-family: var(--font-mono); font-size: 0.8rem; color: #00F0FF; letter-spacing: 0.08em;">
                        OPERATIONAL STATUS:
                    </span>
                    <span style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-left: 6px;">
                        FIXED-TIME SAFETY FALLBACK CYCLE ACTIVE
                    </span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="status-pill warning">ADAPTIVE ENGINE STAGE 3</span>
                    <span class="status-pill active">FAIL-SAFE INTERLOCK ENGAGED</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Grid: Left (Visual Signal Head & Phase Ring), Right (Parameters & Decision Rationale)
    col_left, col_right = st.columns([1.0, 1.45])

    with col_left:
        render_hud_header(
            title="Signal Head Visualizer",
            subtitle="Real-time optical display component",
            badge_text="ACTUATOR PREVIEW",
            badge_type="cyan",
        )

        # Allow user to preview signal aspects in Stage 1
        signal_state_mode = st.radio(
            "Signal Head Aspect Preview (UI Verification):",
            ["Red (Stop)", "Amber (Clearance)", "Green (Go)"],
            horizontal=True,
            index=0,
            help="Stage 1 interactive verification of the visual signal head component.",
        )

        is_red = signal_state_mode.startswith("Red")
        is_amber = signal_state_mode.startswith("Amber")
        is_green = signal_state_mode.startswith("Green")

        ped_state = "DONT_WALK" if is_green else "WALK"
        countdown_display = "18s" if is_red else ("04s" if is_amber else "32s")

        render_traffic_signal_head(
            red_active=is_red,
            amber_active=is_amber,
            green_active=is_green,
            ped_state=ped_state,
            countdown=countdown_display,
        )

        st.write("")

        # Phase Ring Schematic Card
        st.markdown(
            """
            <div class="hud-card" style="padding: 14px 16px;">
                <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; margin-bottom: 8px; text-transform: uppercase;">
                    PHASE RING & BARRIER ARCHITECTURE
                </div>
                <div style="display: flex; flex-direction: column; gap: 6px; font-family: var(--font-mono); font-size: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; background: rgba(0,240,255,0.05); padding: 5px 8px; border-radius: 4px; border-left: 2px solid #00F0FF;">
                        <span style="color: #00F0FF;">PHASE 2 (CURRENT):</span>
                        <span style="color: #F8FAFC;">Arterial N/S Vehicular Through</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 5px 8px; color: #64748B;">
                        <span>PHASE 4:</span>
                        <span>Side Street E/W Vehicular Through</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 5px 8px; color: #64748B;">
                        <span>PEDESTRIAN PHASE 2P:</span>
                        <span>Arterial Crosswalk Exclusive Walk</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        # Controller Data Reserves (Clearly labelled Unavailable)
        render_hud_header(
            title="Dynamic Telemetry Reserves",
            subtitle="Sensor feed inputs from Vision Pipeline (Stage 2 Integration)",
            badge_text="INPUTS UNAVAILABLE",
            badge_type="neutral",
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            render_metric_card(
                label="PEDESTRIAN DEMAND",
                value="Unavailable",
                unit="",
                subtext="Awaiting Zone Analyzer (Stage 2)",
                status_color="muted",
            )
            st.write("")
            render_metric_card(
                label="VEHICLE QUEUE DENSITY",
                value="Unavailable",
                unit="",
                subtext="Awaiting Corridor Detector (Stage 2)",
                status_color="muted",
            )
        with dcol2:
            render_metric_card(
                label="MAX WAITING TIME",
                value="Unavailable",
                unit="",
                subtext="Awaiting Dwell Tracker (Stage 2)",
                status_color="muted",
            )
            st.write("")
            render_metric_card(
                label="EXTENSION COUNTER",
                value="0 / 5",
                unit="CYC",
                subtext="Adaptive Extension Standby",
                status_color="muted",
            )

        st.write("")

        # Phase Timers and Configuration Envelope
        st.markdown(
            """
            <div class="hud-card" style="padding: 14px 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; color: #F8FAFC;">
                        CONTROLLER TIMING ENVELOPE (SAFETY BOUNDS)
                    </span>
                    <span class="status-pill cyan">ACTIVE CONFIG</span>
                </div>
                <table class="tech-table">
                    <thead>
                        <tr>
                            <th>INTERVAL PARAMETER</th>
                            <th>CONFIGURED VALUE</th>
                            <th>DYNAMIC ADAPTATION RANGE</th>
                            <th>SAFETY CONSTRAINT</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Minimum Vehicular Green</td>
                            <td style="color: #00F0FF;">15.0 s</td>
                            <td style="color: #94A3B8;">Fixed Floor</td>
                            <td>MUTCD Minimum Safety Standard</td>
                        </tr>
                        <tr>
                            <td>Maximum Vehicular Green</td>
                            <td style="color: #00F0FF;">60.0 s</td>
                            <td style="color: #94A3B8;">Adaptive Extension Ceiling</td>
                            <td>Prevents Cross-Street Starvation</td>
                        </tr>
                        <tr>
                            <td>Pedestrian Walk Interval</td>
                            <td style="color: #10B981;">7.0 s</td>
                            <td style="color: #94A3B8;">Expandable with High Platoon Demand</td>
                            <td>Guaranteed Minimum Initiation</td>
                        </tr>
                        <tr>
                            <td>Flashing Don't Walk (Clearance)</td>
                            <td style="color: #EF4444;">18.0 s</td>
                            <td style="color: #94A3B8;">Dynamic based on Occupancy</td>
                            <td>3.5 ft/s Walking Speed Standard</td>
                        </tr>
                        <tr>
                            <td>Yellow Vehicular Change</td>
                            <td style="color: #F59E0B;">4.0 s</td>
                            <td style="color: #94A3B8;">Non-Overridable</td>
                            <td>Kinematic Stopping Zone Clearance</td>
                        </tr>
                        <tr>
                            <td>All-Red Clearance Interval</td>
                            <td style="color: #EF4444;">2.0 s</td>
                            <td style="color: #94A3B8;">Non-Overridable</td>
                            <td>Intersection Evacuation Buffer</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Explainable Decision Rationale Panel
        st.markdown(
            """
            <div class="hud-card lavender-accent" style="padding: 14px 18px;">
                <div style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 700; color: #A78BFA; margin-bottom: 6px;">
                    EXPLAINABLE AI (XAI) DECISION RATIONALE
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #CBD5E1; line-height: 1.5;">
                    <p style="margin-bottom: 6px;">
                        [STATE EXPLANATION]: The signal controller is executing in fail-safe fallback cycle mode.
                        Fixed 60-second cycle interval is serving North/South arterial movement.
                    </p>
                    <p style="margin-bottom: 0; color: #94A3B8;">
                        [ADAPTIVE POLICY TRIGGER]: Adaptive green extensions and pedestrian priority preemption
                        will be unlocked upon Stage 3 engine integration with real-time zone occupancy metrics.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
