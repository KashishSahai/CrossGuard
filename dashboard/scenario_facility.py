"""
CrossGuard Controller Scenario Testing & Demonstration Facility
==============================================================
Isolated sandbox testing environment for validating all 7 finite state machine (FSM)
operational policies (Cases A-F and Fail-Safe Stop) without contaminating real session telemetry.
"""

from typing import Dict, List, Optional
import streamlit as st

from core.adaptive_controller import AdaptiveSignalController, ControllerDecision, SignalPhase
from core.spatial_analysis import SpatialSnapshot
from dashboard.components import (
    render_html,
    render_hud_header,
    render_metric_card,
    render_tech_table,
    render_traffic_signal_head,
)

SCENARIOS: Dict[str, Dict] = {
    "CASE_A": {
        "title": "Case A: Arterial Priority (Zero Pedestrians)",
        "description": "Continuous vehicular traffic flow. With zero pedestrians in the waiting zones, the controller maintains vehicular green indefinitely beyond min green.",
        "peds_waiting": 0,
        "max_dwell": 0.0,
        "crosswalk_occupied": False,
        "vehicles": 8,
        "initial_phase": SignalPhase.VEHICLE_GREEN,
        "initial_elapsed": 15.0,
    },
    "CASE_B": {
        "title": "Case B: Sparse Arrival (1-2 Pedestrians Accumulating)",
        "description": "A small group below the group threshold (1-2 pedestrians) arrives. The controller begins an accumulation interval timer without immediately disrupting vehicular traffic.",
        "peds_waiting": 2,
        "max_dwell": 3.0,
        "crosswalk_occupied": False,
        "vehicles": 6,
        "initial_phase": SignalPhase.VEHICLE_GREEN,
        "initial_elapsed": 8.0,
    },
    "CASE_C": {
        "title": "Case C: Platoon Arrival (>= 3 Pedestrians Priority Trigger)",
        "description": "A platoon of 3 or more pedestrians arrives at the curb. The group threshold is met, triggering an immediate transition to vehicular yellow and pedestrian walk phase once min green is satisfied.",
        "peds_waiting": 4,
        "max_dwell": 2.0,
        "crosswalk_occupied": False,
        "vehicles": 5,
        "initial_phase": SignalPhase.VEHICLE_GREEN,
        "initial_elapsed": 13.0,
    },
    "CASE_D": {
        "title": "Case D: Starvation Prevention (Single Pedestrian >= 25s Dwell)",
        "description": "A single pedestrian has waited longer than the maximum tolerable delay (25.0s). The controller overrides vehicular priority to prevent pedestrian starvation.",
        "peds_waiting": 1,
        "max_dwell": 26.5,
        "crosswalk_occupied": False,
        "vehicles": 12,
        "initial_phase": SignalPhase.VEHICLE_GREEN,
        "initial_elapsed": 20.0,
    },
    "CASE_E": {
        "title": "Case E: Dynamic Clearance Extension (Occupied Crosswalk)",
        "description": "During pedestrian clearance, pedestrians are still detected in the zebra crossing corridor. The controller grants dynamic extensions (+3.0s) up to 3 times to prevent trapped pedestrians.",
        "peds_waiting": 0,
        "max_dwell": 0.0,
        "crosswalk_occupied": True,
        "vehicles": 4,
        "initial_phase": SignalPhase.PEDESTRIAN_CLEARANCE,
        "initial_elapsed": 12.0,
    },
    "CASE_F": {
        "title": "Case F: Safe Arterial Return (Clear Crosswalk Debounce)",
        "description": "The crosswalk is confirmed clear for 3 consecutive observation cycles after nominal clearance interval. The controller safely returns green to vehicular traffic.",
        "peds_waiting": 0,
        "max_dwell": 0.0,
        "crosswalk_occupied": False,
        "vehicles": 6,
        "initial_phase": SignalPhase.PEDESTRIAN_CLEARANCE,
        "initial_elapsed": 12.5,
    },
    "FAIL_SAFE": {
        "title": "Fail-Safe Conservative Interlock (Optical Occlusion / Sensor Fault)",
        "description": "Critical camera failure or sensor anomaly detected. The controller immediately forces an all-red conservative stop, halting all vehicular and conflicting pedestrian movements.",
        "peds_waiting": 2,
        "max_dwell": 5.0,
        "crosswalk_occupied": False,
        "vehicles": 7,
        "initial_phase": SignalPhase.FAIL_SAFE_STOP,
        "initial_elapsed": 0.0,
    },
}


def _get_sandbox_state() -> Dict:
    """Initializes or retrieves the isolated testing sandbox state from Streamlit session."""
    if "sandbox_ctrl" not in st.session_state:
        ctrl = AdaptiveSignalController()
        st.session_state["sandbox_ctrl"] = ctrl
        st.session_state["sandbox_time"] = 15.0
        st.session_state["sandbox_decision"] = ctrl.step(
            SpatialSnapshot(
                timestamp=15.0,
                waiting_pedestrians_count=0,
                waiting_track_ids=[],
                max_pedestrian_waiting_time=0.0,
                crosswalk_occupied=False,
                crossing_pedestrians_count=0,
                crossing_track_ids=[],
                approach_vehicles_count=5,
                vehicle_track_ids=[],
                queue_density_level="MEDIUM",
            ),
            timestamp=15.0,
        )
        st.session_state["sandbox_history"] = []
        st.session_state["active_scenario_key"] = "CASE_A"
    return st.session_state


def load_scenario_preset(scenario_key: str) -> None:
    """Configures the sandbox controller into the initial conditions of the chosen scenario."""
    sc = SCENARIOS[scenario_key]
    ctrl = AdaptiveSignalController()
    st.session_state["active_scenario_key"] = scenario_key
    st.session_state["sandbox_time"] = sc["initial_elapsed"]

    # Pre-condition the controller state
    ctrl.current_phase = sc["initial_phase"]
    ctrl.phase_start_time = 0.0
    ctrl.phase_elapsed = sc["initial_elapsed"]

    if scenario_key == "FAIL_SAFE":
        ctrl.trigger_fault("Simulated camera occlusion and tracking loss in sandbox", timestamp=sc["initial_elapsed"])
    elif scenario_key == "CASE_E":
        ctrl.current_nominal_duration = ctrl.config.pedestrian_clearance_duration
    elif scenario_key == "CASE_F":
        ctrl.current_nominal_duration = ctrl.config.pedestrian_clearance_duration
        ctrl.consecutive_clear_observations = 1

    snap = SpatialSnapshot(
        timestamp=sc["initial_elapsed"],
        waiting_pedestrians_count=sc["peds_waiting"],
        waiting_track_ids=[101, 102][: sc["peds_waiting"]],
        max_pedestrian_waiting_time=sc["max_dwell"],
        crosswalk_occupied=sc["crosswalk_occupied"],
        crossing_pedestrians_count=1 if sc["crosswalk_occupied"] else 0,
        crossing_track_ids=[201] if sc["crosswalk_occupied"] else [],
        approach_vehicles_count=sc["vehicles"],
        vehicle_track_ids=list(range(10, 10 + sc["vehicles"])),
        queue_density_level="HIGH" if sc["vehicles"] >= 5 else "LOW",
    )

    decision = ctrl.step(snap, timestamp=sc["initial_elapsed"])
    st.session_state["sandbox_ctrl"] = ctrl
    st.session_state["sandbox_decision"] = decision
    st.session_state["sandbox_history"] = [
        {
            "time": f"{sc['initial_elapsed']:.1f}s",
            "phase": decision.current_phase.value,
            "case": decision.decision_case,
            "reason": decision.reason,
            "peds": sc["peds_waiting"],
            "cross_occ": "YES" if sc["crosswalk_occupied"] else "NO",
        }
    ]


def step_sandbox(
    dt: float,
    waiting_peds: int,
    max_dwell: float,
    crosswalk_occ: bool,
    approach_vehs: int,
) -> None:
    """Advances the isolated controller by dt seconds with current manual or preset inputs."""
    ctrl: AdaptiveSignalController = st.session_state["sandbox_ctrl"]
    st.session_state["sandbox_time"] += dt
    t = st.session_state["sandbox_time"]

    snap = SpatialSnapshot(
        timestamp=t,
        waiting_pedestrians_count=waiting_peds,
        waiting_track_ids=list(range(100, 100 + waiting_peds)),
        max_pedestrian_waiting_time=max_dwell,
        crosswalk_occupied=crosswalk_occ,
        crossing_pedestrians_count=1 if crosswalk_occ else 0,
        crossing_track_ids=[201] if crosswalk_occ else [],
        approach_vehicles_count=approach_vehs,
        vehicle_track_ids=list(range(10, 10 + approach_vehs)),
        queue_density_level="HIGH" if approach_vehs >= 5 else "LOW",
    )

    decision = ctrl.step(snap, timestamp=t)
    st.session_state["sandbox_decision"] = decision

    hist = st.session_state.get("sandbox_history", [])
    hist.insert(
        0,
        {
            "time": f"{t:.1f}s",
            "phase": decision.current_phase.value,
            "case": decision.decision_case,
            "reason": decision.reason,
            "peds": waiting_peds,
            "cross_occ": "YES" if crosswalk_occ else "NO",
        },
    )
    if len(hist) > 30:
        hist.pop()
    st.session_state["sandbox_history"] = hist


def render_scenario_testing_facility() -> None:
    """Renders the interactive scenario demonstration sandbox tab."""
    _get_sandbox_state()
    active_key = st.session_state.get("active_scenario_key", "CASE_A")

    render_hud_header(
        title="FSM Scenario Demonstration & Safety Testing Sandbox",
        subtitle="Validate all 7 formal controller operating policies in an isolated sandbox environment",
        badge_text="ISOLATED SANDBOX MODE",
        badge_type="lavender",
    )

    # Scenario Selection
    sc_keys = list(SCENARIOS.keys())
    selected_key = st.selectbox(
        "Select Test Scenario Configuration",
        sc_keys,
        index=sc_keys.index(active_key) if active_key in sc_keys else 0,
        format_func=lambda k: SCENARIOS[k]["title"],
        help="Select a scenario to load its initial conditions into the isolated sandbox controller.",
    )

    sc_info = SCENARIOS[selected_key]
    load_col1, load_col2 = st.columns([3, 1])
    with load_col1:
        st.info(f"**{sc_info['title']}**: {sc_info['description']}")
    with load_col2:
        if st.button("📥 Load Selected Preset", type="primary", use_container_width=True):
            load_scenario_preset(selected_key)
            st.rerun()

    st.write("")

    ctrl: AdaptiveSignalController = st.session_state["sandbox_ctrl"]
    dec: ControllerDecision = st.session_state["sandbox_decision"]
    current_time: float = st.session_state["sandbox_time"]

    # Interactive Sandbox Layout
    s_col_left, s_col_right = st.columns([1.0, 1.45])

    with s_col_left:
        render_hud_header(
            title="Sandbox Signal Head",
            subtitle=f"Phase: {dec.current_phase.value}",
            badge_text=f"T = {current_time:.1f}s",
            badge_type="cyan",
        )

        is_red = dec.current_phase in (
            SignalPhase.ALL_RED_CLEARANCE,
            SignalPhase.PEDESTRIAN_WALK,
            SignalPhase.PEDESTRIAN_CLEARANCE,
            SignalPhase.FAIL_SAFE_STOP,
        )
        is_yellow = dec.current_phase == SignalPhase.VEHICLE_YELLOW
        is_green = dec.current_phase == SignalPhase.VEHICLE_GREEN
        ped_state = (
            "WALK"
            if dec.current_phase in (SignalPhase.PEDESTRIAN_WALK, SignalPhase.PEDESTRIAN_CLEARANCE)
            else "DONT_WALK"
        )

        render_traffic_signal_head(
            red_active=is_red,
            amber_active=is_yellow,
            green_active=is_green,
            ped_state=ped_state,
            countdown=f"{dec.phase_remaining_seconds:.0f}s",
            phase_label=dec.current_phase.value.replace("_", " "),
        )

        st.write("")

        # Step Controls
        st.markdown("**Sandbox Stepping Controls**")
        step_c1, step_c2, step_c3 = st.columns(3)
        do_step_1 = step_c1.button("▶ Step +1.0s", use_container_width=True)
        do_step_5 = step_c2.button("⏩ Step +5.0s", use_container_width=True)
        do_reset = step_c3.button("🔄 Reset", use_container_width=True)

        # Fault Injection Controls
        st.markdown("**Fault Injection**")
        f_c1, f_c2 = st.columns(2)
        if f_c1.button("🚨 Inject Fault", help="Triggers conservative fail-safe stop"):
            ctrl.trigger_fault("Manual fault injected in sandbox testing", timestamp=current_time)
            st.session_state["sandbox_decision"] = ctrl.last_decision
            st.rerun()
        if f_c2.button("✅ Clear Fault", help="Restores normal cycle"):
            ctrl.clear_fault(timestamp=current_time)
            st.session_state["sandbox_decision"] = ctrl.last_decision
            st.rerun()

    with s_col_right:
        # Dynamic telemetry inputs injection sliders
        st.markdown("**Live Input Injector (Overrides Scenario Default Inputs)**")
        inj_c1, inj_c2 = st.columns(2)
        with inj_c1:
            in_peds = st.slider("Waiting Pedestrians", 0, 10, int(sc_info["peds_waiting"]), 1, key="sb_peds")
            in_dwell = st.slider("Max Ped Dwell Time (s)", 0.0, 40.0, float(sc_info["max_dwell"]), 0.5, key="sb_dwell")
        with inj_c2:
            in_cross = st.checkbox("Crosswalk Occupied", value=bool(sc_info["crosswalk_occupied"]), key="sb_cross")
            in_vehs = st.slider("Approach Vehicles", 0, 15, int(sc_info["vehicles"]), 1, key="sb_vehs")

        if do_step_1:
            step_sandbox(1.0, in_peds, in_dwell, in_cross, in_vehs)
            st.rerun()
        elif do_step_5:
            step_sandbox(5.0, in_peds, in_dwell, in_cross, in_vehs)
            st.rerun()
        elif do_reset:
            load_scenario_preset(selected_key)
            st.rerun()

        st.write("")

        # Decision Rationale Card
        render_html(
            f"""
            <div class="hud-card lavender-accent" style="padding: 16px 20px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-family: var(--font-mono); font-size: 0.85rem; font-weight: 700; color: #A78BFA;">
                        DECISION RATIONALE: {dec.decision_case}
                    </span>
                    <span class="status-pill lavender">PHASE: {dec.current_phase.value}</span>
                </div>
                <div style="font-family: var(--font-mono); font-size: 0.82rem; color: #F8FAFC; line-height: 1.55; margin-bottom: 8px;">
                    {dec.reason}
                </div>
                <div style="display: flex; gap: 14px; font-family: var(--font-mono); font-size: 0.75rem; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <div><span>ELAPSED:</span> <strong style="color: #00F0FF;">{dec.phase_elapsed_seconds:.1f}s</strong></div>
                    <div><span>REMAINING:</span> <strong style="color: #10B981;">{dec.phase_remaining_seconds:.1f}s</strong></div>
                    <div><span>EXTENSIONS:</span> <strong style="color: #A78BFA;">{dec.clearance_extensions}</strong></div>
                    <div><span>FAULT ACTIVE:</span> <strong style="color: {'#EF4444' if ctrl.fault_active else '#10B981'};">{'YES' if ctrl.fault_active else 'NO'}</strong></div>
                </div>
            </div>
            """
        )

        # Audit History Table
        hist_rows = [
            [
                h["time"],
                f"<strong style='color:#00F0FF;'>{h['phase']}</strong>",
                f"<span class='status-pill lavender'>{h['case']}</span>",
                f"{h['peds']} peds",
                h["cross_occ"],
                h["reason"][:80] + "..." if len(h["reason"]) > 80 else h["reason"],
            ]
            for h in st.session_state.get("sandbox_history", [])[:6]
        ]

        if not hist_rows:
            hist_rows = [["--", "--", "--", "--", "--", "Awaiting sandbox steps"]]

        render_tech_table(
            headers=["TIMESTAMP", "CURRENT PHASE", "CASE", "WAITING", "CROSSING", "DECISION RATIONALE"],
            rows=hist_rows,
            caption="SANDBOX TRANSITION AUDIT TRAIL (ISOLATED FROM SESSION TELEMETRY)",
        )
