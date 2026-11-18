"""Unit and integration tests for the Controller -> Bloom City bridge (Step 6)."""
from __future__ import annotations

import json
import re
import pytest

from core.adaptive_controller import AdaptiveController
from core.models import (
    ControllerConfig,
    ControllerState,
    Decision,
    DemandSnapshot,
    PedestrianSignal,
    Phase,
    VehicleSignal,
)
from simulation.bloom_city_engine import BloomCityEngine, build_frames
from simulation.bloom_city_renderer import render_bloom_city_html
from simulation.controller_bridge import (
    BloomCitySignalState,
    ControllerBridge,
    SimulationMode,
    map_controller_state,
)


@pytest.fixture
def bridge() -> ControllerBridge:
    return ControllerBridge()


@pytest.fixture
def controller() -> AdaptiveController:
    return AdaptiveController()


def _get_html_payload(html: str) -> dict:
    """Extract embedded JSON payload from rendered Bloom City HTML."""
    match = re.search(r'<script id="bcdata" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None, "Could not find bcdata script tag in HTML"
    return json.loads(match.group(1).replace(r"<\/", "</"))


# -----------------------------------------------------------------------------
# Test 1 — VEHICLE_GREEN produces expected Bloom City signal state
# -----------------------------------------------------------------------------
def test_vehicle_green_signal_state(bridge: ControllerBridge, controller: AdaptiveController):
    controller.force_phase(Phase.VEHICLE_GREEN)
    sig = bridge.map_state(controller.state)

    assert isinstance(sig, BloomCitySignalState)
    assert sig.phase == Phase.VEHICLE_GREEN
    assert sig.vehicle_signal == VehicleSignal.GREEN
    assert sig.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig.vehicle_signal_index == 0
    assert sig.pedestrian_signal_index == 0
    assert sig.vehicle_lens_color == "#38d889"
    assert sig.pedestrian_lens_color == "#ff5d6c"
    assert sig.pedestrian_glyph == "✋"
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 2 — VEHICLE_YELLOW produces expected yellow signal
# -----------------------------------------------------------------------------
def test_vehicle_yellow_signal_state(bridge: ControllerBridge, controller: AdaptiveController):
    controller.force_phase(Phase.VEHICLE_YELLOW)
    sig = bridge.map_state(controller.state)

    assert sig.phase == Phase.VEHICLE_YELLOW
    assert sig.vehicle_signal == VehicleSignal.YELLOW
    assert sig.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig.vehicle_signal_index == 1
    assert sig.pedestrian_signal_index == 0
    assert sig.vehicle_lens_color == "#ffd23f"
    assert sig.pedestrian_lens_color == "#ff5d6c"
    assert sig.pedestrian_glyph == "✋"
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 3 — ALL_RED_CLEARANCE produces vehicle red and pedestrian DON'T WALK
# -----------------------------------------------------------------------------
def test_all_red_clearance_signal_state(bridge: ControllerBridge, controller: AdaptiveController):
    controller.force_phase(Phase.ALL_RED_CLEARANCE)
    sig = bridge.map_state(controller.state)

    assert sig.phase == Phase.ALL_RED_CLEARANCE
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig.vehicle_signal_index == 2
    assert sig.pedestrian_signal_index == 0
    assert sig.vehicle_lens_color == "#ff5d6c"
    assert sig.pedestrian_lens_color == "#ff5d6c"
    assert sig.pedestrian_glyph == "✋"
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 4 — PEDESTRIAN_WALK produces vehicle red and pedestrian WALK
# -----------------------------------------------------------------------------
def test_pedestrian_walk_signal_state(bridge: ControllerBridge, controller: AdaptiveController):
    controller.force_phase(Phase.PEDESTRIAN_WALK)
    sig = bridge.map_state(controller.state)

    assert sig.phase == Phase.PEDESTRIAN_WALK
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.WALK
    assert sig.vehicle_signal_index == 2
    assert sig.pedestrian_signal_index == 1
    assert sig.vehicle_lens_color == "#ff5d6c"
    assert sig.pedestrian_lens_color == "#38d889"
    assert sig.pedestrian_glyph == "🚶"
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 5 — PEDESTRIAN_CLEARANCE produces expected clearance representation
# -----------------------------------------------------------------------------
def test_pedestrian_clearance_signal_state(bridge: ControllerBridge, controller: AdaptiveController):
    controller.force_phase(Phase.PEDESTRIAN_CLEARANCE)
    sig = bridge.map_state(controller.state)

    assert sig.phase == Phase.PEDESTRIAN_CLEARANCE
    assert sig.vehicle_signal == VehicleSignal.RED
    assert sig.pedestrian_signal == PedestrianSignal.CLEARING
    assert sig.vehicle_signal_index == 2
    assert sig.pedestrian_signal_index == 2
    assert sig.vehicle_lens_color == "#ff5d6c"
    assert sig.pedestrian_lens_color == "#ffb020"
    assert sig.pedestrian_glyph == "⏳"
    assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 6 — Changing controller state changes mapped Bloom City signal state
# -----------------------------------------------------------------------------
def test_changing_controller_state_updates_signals(bridge: ControllerBridge, controller: AdaptiveController):
    # Sequence of phases stepped sequentially
    expected_sequence = [
        (Phase.VEHICLE_GREEN, VehicleSignal.GREEN, PedestrianSignal.DONT_WALK),
        (Phase.VEHICLE_YELLOW, VehicleSignal.YELLOW, PedestrianSignal.DONT_WALK),
        (Phase.ALL_RED_CLEARANCE, VehicleSignal.RED, PedestrianSignal.DONT_WALK),
        (Phase.PEDESTRIAN_WALK, VehicleSignal.RED, PedestrianSignal.WALK),
        (Phase.PEDESTRIAN_CLEARANCE, VehicleSignal.RED, PedestrianSignal.CLEARING),
    ]

    for phase, exp_vsig, exp_psig in expected_sequence:
        controller.force_phase(phase)
        sig = bridge.map_state(controller.state)
        assert sig.phase == phase
        assert sig.vehicle_signal == exp_vsig
        assert sig.pedestrian_signal == exp_psig
        assert sig.is_safe is True


# -----------------------------------------------------------------------------
# Test 7 — Invalid/unknown controller state fails safely
# -----------------------------------------------------------------------------
def test_invalid_controller_state_fails_safely(bridge: ControllerBridge):
    # 1. None passed
    sig_none = bridge.map_state(None)
    assert sig_none.phase == Phase.ALL_RED_CLEARANCE
    assert sig_none.vehicle_signal == VehicleSignal.RED
    assert sig_none.pedestrian_signal == PedestrianSignal.DONT_WALK
    assert sig_none.is_safe is True
    assert sig_none.fail_safe_active is True

    # 2. Corrupted object
    sig_bad = bridge.map_state("not_a_controller_state")  # type: ignore
    assert sig_bad.phase == Phase.ALL_RED_CLEARANCE
    assert sig_bad.vehicle_signal == VehicleSignal.RED
    assert sig_bad.is_safe is True


# -----------------------------------------------------------------------------
# Test 8 — Existing Bloom City demo/scenario behavior remains intact
# -----------------------------------------------------------------------------
def test_existing_demo_scenario_behavior_intact():
    frames, reasons, engine = build_frames("balanced", seconds=10.0, warmup=5.0)
    assert len(frames) > 0
    assert len(reasons) > 0
    assert isinstance(engine, BloomCityEngine)

    html = render_bloom_city_html(frames, reasons, mode="SCENARIO DEMO")
    assert "<svg" in html
    assert "SCENARIO DEMO" in html
    assert "Representative agents derived from scenario telemetry" in html

    # Verify frame fields in payload
    payload = _get_html_payload(html)
    f0 = payload["frames"][0]
    assert "vs" in f0
    assert "ps" in f0
    assert "ph" in f0


# -----------------------------------------------------------------------------
# Test 9 — Rendered signal reflection in HTML frames
# -----------------------------------------------------------------------------
def test_rendered_signals_match_controller_phases():
    # Verify that each phase's rendered HTML frame contains the exact signal codes
    for phase in Phase:
        frames, reasons, _ = build_frames("quiet", seconds=2.0, warmup=2.0, hold_phase=phase)
        html = render_bloom_city_html(frames, reasons)
        payload = _get_html_payload(html)
        f0 = payload["frames"][0]

        expected_vs = {
            Phase.VEHICLE_GREEN: 0,
            Phase.VEHICLE_YELLOW: 1,
            Phase.ALL_RED_CLEARANCE: 2,
            Phase.PEDESTRIAN_WALK: 2,
            Phase.PEDESTRIAN_CLEARANCE: 2,
        }[phase]

        expected_ps = {
            Phase.VEHICLE_GREEN: 0,
            Phase.VEHICLE_YELLOW: 0,
            Phase.ALL_RED_CLEARANCE: 0,
            Phase.PEDESTRIAN_WALK: 1,
            Phase.PEDESTRIAN_CLEARANCE: 2,
        }[phase]

        assert f0["vs"] == expected_vs, f"Phase {phase} had vs={f0['vs']}, expected {expected_vs}"
        assert f0["ps"] == expected_ps, f"Phase {phase} had ps={f0['ps']}, expected {expected_ps}"


# -----------------------------------------------------------------------------
# Test 10 — Live controller mode frame generation and rendering via Bridge
# -----------------------------------------------------------------------------
def test_live_controller_bridge_frames(bridge: ControllerBridge, controller: AdaptiveController):
    engine = BloomCityEngine(scenario="quiet", seed=42)

    # 10 frames with no demand, then 10 frames with pedestrian demand
    demands = [DemandSnapshot(source="vision") for _ in range(10)]
    demands += [DemandSnapshot(pedestrians_waiting=3, source="vision") for _ in range(20)]

    frames, reasons = bridge.build_controller_frames(engine, controller, demands, dt=0.1, fps=10)
    assert len(frames) == len(demands)
    assert len(reasons) > 0

    # Render via bridge in LIVE CONTROLLER mode
    html = bridge.render_live_html(frames, reasons, title="Live Pedestrian Priority")
    assert SimulationMode.LIVE_CONTROLLER.value in html
    assert "Live Pedestrian Priority" in html
    assert "Signals driven directly by AdaptiveController" in html
