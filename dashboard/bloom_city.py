"""
CrossGuard - Bloom City Digital Twin Simulation Workspace
=========================================================
Hosts the digital twin intersection simulator viewport, communication bridge
status, viewport camera selection, and environmental controls for Stage 3.
"""

import streamlit as st
from dashboard.components import (
    render_empty_state_card,
    render_hud_header,
    render_metric_card,
    render_status_badge,
)


def render_bloom_city_page() -> None:
    """Renders the Bloom City Digital Twin intersection simulation workspace."""
    render_hud_header(
        title="Bloom City // Digital Twin Intersection Simulator",
        subtitle="Microscopic pedestrian-priority urban traffic simulation workspace",
        badge_text="SIMULATOR // STAGE 3 PENDING",
        badge_type="lavender",
    )

    # Simulator Connection Status Bar
    st.markdown(
        """
        <div class="hud-card lavender-accent" style="padding: 14px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span class="pulse-dot warning"></span>
                    <div>
                        <div style="font-family: var(--font-mono); font-size: 0.82rem; font-weight: 700; color: #F8FAFC;">
                            SIMULATION ENGINE BRIDGE: DISCONNECTED
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">
                            Sector 7: Bloom Ave & Aurora Blvd (4-Way Dual Crosswalk Node)
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: center; font-family: var(--font-mono); font-size: 0.78rem;">
                    <div><span style="color: #64748B;">PROTOCOL:</span> <span style="color: #CBD5E1;">WebSocket / Local Socket</span></div>
                    <div><span style="color: #64748B;">SYNC CLOCK:</span> <span style="color: #00F0FF;">00:00:00 (HALTED)</span></div>
                    <span class="status-pill warning">STAGE 3 INTEGRATION</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Simulation Viewport Toolbar (Controls for perspective, time of day, weather)
    tcol1, tcol2, tcol3 = st.columns([1.2, 1.0, 1.0])
    with tcol1:
        view_angle = st.selectbox(
            "Simulation Viewport Mode",
            ["Overhead 2D Orthographic (Tactical)", "Isometric 3D Perspective", "Driver Eye Level", "Pedestrian Crossing Eye"],
            index=0,
            help="Select the digital twin rendering perspective for Stage 3.",
        )
    with tcol2:
        lighting = st.selectbox(
            "Lighting & Time of Day",
            ["Night Surveillance (Cyber Glow)", "Daylight Clear", "Dusk / Sunset", "Precipitation / Fog"],
            index=0,
        )
    with tcol3:
        playback_rate = st.selectbox(
            "Simulation Step Rate",
            ["1.0x (Real-Time Synchronous)", "0.5x (Slow-Motion Audit)", "2.0x (Accelerated Scenario)", "Step-by-Step"],
            index=0,
        )

    st.write("")

    # Dedicated Large Simulation Workspace Canvas
    st.markdown(
        f"""
        <div class="hud-corner-box" style="min-height: 440px; display: flex; flex-direction: column; justify-content: center; align-items: center; background: radial-gradient(circle at center, rgba(17, 24, 39, 0.9) 0%, rgba(8, 12, 20, 0.98) 100%);">
            <!-- Simulated Intersection Schematic Wireframe -->
            <div style="position: absolute; inset: 0; pointer-events: none; opacity: 0.45; background-image: 
                radial-gradient(circle at 50% 50%, rgba(0, 240, 255, 0.15) 0%, transparent 60%),
                linear-gradient(rgba(0, 240, 255, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 240, 255, 0.05) 1px, transparent 1px);
                background-size: 100% 100%, 30px 30px, 30px 30px;">
            </div>

            <!-- Central HUD Graphic -->
            <div style="z-index: 2; text-align: center; max-width: 620px; padding: 24px;">
                <div style="font-family: var(--font-mono); font-size: 0.85rem; color: #00F0FF; letter-spacing: 0.15em; margin-bottom: 8px;">
                    // BLOOM CITY DIGITAL TWIN WORKSPACE
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC; margin-bottom: 12px; letter-spacing: 0.04em;">
                    SECTOR 7: INTERSECTION KINETIC SIMULATOR
                </div>
                <p style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6; margin-bottom: 20px;">
                    This dedicated simulation canvas is prepared for real-time 2D/3D intersection animation in Stage 3.
                    It will visualize vehicles approaching stop lines, pedestrian agent clusters gathering at waiting curbs,
                    and dynamic signal phase transitions commanded by CrossGuard's adaptive policy engine.
                </p>
                <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <span class="status-pill cyan">VIEW: {view_angle.split('(')[0].strip().upper()}</span>
                    <span class="status-pill lavender">LIGHTING: {lighting.split('(')[0].strip().upper()}</span>
                    <span class="status-pill warning">SIMULATION ENGINE: STAGE 3</span>
                </div>
            </div>

            <!-- Corner Telemetry Markers -->
            <div style="position: absolute; bottom: 12px; left: 16px; font-family: var(--font-mono); font-size: 0.72rem; color: #64748B;">
                LAT/LON: 37.7749° N, 122.4194° W | NODE ID: BC-SEC07-INT01
            </div>
            <div style="position: absolute; bottom: 12px; right: 16px; font-family: var(--font-mono); font-size: 0.72rem; color: #64748B;">
                PHYSICS STEP: 60Hz | AGENTS: 0 ACTIVE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Digital Twin Real-Time Entity Counts (Strictly Unavailable / No Fake Data)
    render_hud_header(
        title="Simulation Entity & Safety Assertions",
        subtitle="Kinetic physics telemetry (Awaiting Stage 3 Simulator Engine)",
        badge_text="TELEMETRY OFFLINE",
        badge_type="neutral",
    )

    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        render_metric_card(
            label="SIMULATED VEHICLES",
            value="--",
            unit="FLOW",
            subtext="Stage 3 Physics Engine",
            status_color="muted",
        )
    with scol2:
        render_metric_card(
            label="SIMULATED PEDESTRIANS",
            value="--",
            unit="PEDS",
            subtext="Stage 3 Agent Spawner",
            status_color="muted",
        )
    with scol3:
        render_metric_card(
            label="CLEARANCE TIME MARGIN",
            value="--",
            unit="SEC",
            subtext="Safety Interlock Buffer",
            status_color="muted",
        )
    with scol4:
        render_metric_card(
            label="CONFLICT SEVERITY",
            value="--",
            unit="TTC",
            subtext="Time-to-Collision Index",
            status_color="muted",
        )
