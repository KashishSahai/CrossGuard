import math
import random
import sys

import pytest

from core.adaptive_controller import AdaptiveController
from core.models import (PHASE_ORDER, ControllerConfig, Decision, DemandSnapshot,
                         PedestrianSignal, Phase, VehicleSignal)


def demand(waiting=0, wait=0.0, crossing=0, vehicles=0):
    return DemandSnapshot(pedestrians_waiting=waiting, longest_pedestrian_wait_s=wait,
                          pedestrians_in_crosswalk=crossing, vehicle_demand=vehicles,
                          vehicle_queue=0, source="test")


def run(ctrl, seconds, d, dt=0.1):
    trace = []
    for _ in range(int(round(seconds / dt))):
        trace.append(ctrl.step(dt, d))
    return trace


def first_phase_time(trace, phase):
    for s in trace:
        if s.current_phase == phase:
            return s.total_time
    return None


def test_controller_has_no_ui_dependency():
    assert "streamlit" not in sys.modules


def test_signal_mapping_for_every_phase():
    expected = {
        Phase.VEHICLE_GREEN: (VehicleSignal.GREEN, PedestrianSignal.DONT_WALK),
        Phase.VEHICLE_YELLOW: (VehicleSignal.YELLOW, PedestrianSignal.DONT_WALK),
        Phase.ALL_RED_CLEARANCE: (VehicleSignal.RED, PedestrianSignal.DONT_WALK),
        Phase.PEDESTRIAN_WALK: (VehicleSignal.RED, PedestrianSignal.WALK),
        Phase.PEDESTRIAN_CLEARANCE: (VehicleSignal.RED, PedestrianSignal.CLEARING),
    }
    c = AdaptiveController()
    for phase, (v, p) in expected.items():
        c.force_phase(phase, hold=True)
        s = c.state
        assert (s.vehicle_signal, s.pedestrian_signal) == (v, p)


def test_full_sequence_order_and_durations():
    cfg = ControllerConfig()
    c = AdaptiveController(cfg)
    trace = run(c, 60, demand(waiting=3, vehicles=1))
    seq = [trace[0].current_phase]
    for s in trace:
        if s.current_phase != seq[-1]:
            seq.append(s.current_phase)
    assert seq[:6] == list(PHASE_ORDER) + [Phase.VEHICLE_GREEN]
    t_y = first_phase_time(trace, Phase.VEHICLE_YELLOW)
    t_r = first_phase_time(trace, Phase.ALL_RED_CLEARANCE)
    t_w = first_phase_time(trace, Phase.PEDESTRIAN_WALK)
    t_c = first_phase_time(trace, Phase.PEDESTRIAN_CLEARANCE)
    assert math.isclose(t_y, cfg.min_vehicle_green, abs_tol=0.2)
    assert math.isclose(t_r - t_y, cfg.yellow_time, abs_tol=0.2)
    assert math.isclose(t_w - t_r, cfg.all_red_time, abs_tol=0.2)
    assert math.isclose(t_c - t_w, cfg.pedestrian_walk_time, abs_tol=0.2)


def test_minimum_green_is_never_cut_short():
    c = AdaptiveController()
    trace = run(c, 9.5, demand(waiting=9, wait=99, vehicles=0))
    assert all(s.current_phase == Phase.VEHICLE_GREEN for s in trace)


def test_no_pedestrians_vehicles_keep_green_past_max_green():
    c = AdaptiveController()
    trace = run(c, 120, demand(waiting=0, vehicles=3))
    assert all(s.current_phase == Phase.VEHICLE_GREEN for s in trace)
    assert trace[-1].decision == Decision.VEHICLE_PRIORITY


def test_pedestrian_demand_with_low_vehicle_demand_selects_pedestrian_priority():
    c = AdaptiveController()
    trace = run(c, 15, demand(waiting=3, wait=5, vehicles=1))
    assert any(s.current_phase == Phase.VEHICLE_YELLOW for s in trace)
    s = next(s for s in trace if s.current_phase == Phase.VEHICLE_YELLOW)
    assert s.decision == Decision.PEDESTRIAN_PRIORITY
    assert "3 pedestrians waiting" in s.decision_reason


def test_heavy_vehicle_demand_defers_pedestrians_until_wait_limit():
    cfg = ControllerConfig()
    c = AdaptiveController(cfg)
    trace = run(c, 20, demand(waiting=4, wait=10, vehicles=cfg.vehicle_release_threshold))
    assert all(s.current_phase == Phase.VEHICLE_GREEN for s in trace)
    assert "vehicle demand is heavy" in trace[-1].decision_reason
    trace = run(c, 1, demand(waiting=4, wait=cfg.max_pedestrian_wait,
                             vehicles=cfg.vehicle_release_threshold))
    assert trace[-1].current_phase == Phase.VEHICLE_YELLOW


def test_max_vehicle_green_forces_pedestrian_service():
    cfg = ControllerConfig()
    c = AdaptiveController(cfg)
    trace = run(c, cfg.max_vehicle_green + 1, demand(waiting=2, wait=1, vehicles=99))
    assert first_phase_time(trace, Phase.VEHICLE_YELLOW) is not None


def test_single_pedestrian_served_after_single_wait():
    cfg = ControllerConfig()
    c = AdaptiveController(cfg)
    trace = run(c, 15, demand(waiting=1, wait=5, vehicles=0))
    assert all(s.current_phase == Phase.VEHICLE_GREEN for s in trace)
    trace = run(c, 1, demand(waiting=1, wait=cfg.single_pedestrian_wait, vehicles=0))
    assert trace[-1].current_phase == Phase.VEHICLE_YELLOW


def test_clearance_extends_only_while_crosswalk_occupied_and_is_bounded():
    cfg = ControllerConfig()
    c = AdaptiveController(cfg)
    c.force_phase(Phase.PEDESTRIAN_CLEARANCE)
    trace = run(c, cfg.pedestrian_clearance_time + cfg.max_clearance_extension + 5,
                demand(crossing=2))
    t_end = first_phase_time(trace, Phase.VEHICLE_GREEN)
    assert t_end is not None
    assert t_end <= cfg.pedestrian_clearance_time + cfg.max_clearance_extension + 0.3
    assert t_end > cfg.pedestrian_clearance_time + 1.0

    c2 = AdaptiveController(cfg)
    c2.force_phase(Phase.PEDESTRIAN_CLEARANCE)
    t2 = first_phase_time(run(c2, 12, demand(crossing=0)), Phase.VEHICLE_GREEN)
    assert math.isclose(t2, cfg.pedestrian_clearance_time, abs_tol=0.2)


def test_vehicles_released_after_clearance_with_green_and_cycle_count():
    c = AdaptiveController()
    run(c, 60, demand(waiting=3, vehicles=0))
    assert c.state.cycles_completed >= 1
    assert c.state.pedestrian_phases_served >= 1


def test_failsafe_on_missing_telemetry_uses_fixed_time_cycle():
    cfg = ControllerConfig()
    c = AdaptiveController(cfg)
    trace = run(c, cfg.failsafe_after + 0.5, None)
    assert trace[-1].fail_safe_active
    assert trace[-1].decision == Decision.FIXED_TIME_FALLBACK
    trace += run(c, cfg.fixed_green_time + 1, None)
    assert first_phase_time(trace, Phase.VEHICLE_YELLOW) is not None   # still goes via yellow
    seq = []
    for s in trace:
        if not seq or seq[-1] != s.current_phase:
            seq.append(s.current_phase)
    assert seq[:3] == [Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW, Phase.ALL_RED_CLEARANCE]


def test_failsafe_triggers_on_invalid_values_and_recovers():
    c = AdaptiveController()
    bad = demand(waiting=float("nan"))
    assert not bad.is_valid()
    run(c, 4, bad)
    assert c.state.fail_safe_active
    run(c, 0.5, demand())
    assert not c.state.fail_safe_active


def test_short_telemetry_dropout_does_not_trigger_failsafe():
    c = AdaptiveController()
    run(c, 1.0, None)
    assert not c.state.fail_safe_active


def test_ignores_bad_dt():
    c = AdaptiveController()
    for dt in (0, -1, float("nan"), float("inf")):
        before = c.state.total_time
        assert c.step(dt, demand()).total_time == before


def test_random_demand_never_breaks_sequence_or_skips_yellow():
    allowed = {Phase.VEHICLE_GREEN: {Phase.VEHICLE_YELLOW},
               Phase.VEHICLE_YELLOW: {Phase.ALL_RED_CLEARANCE},
               Phase.ALL_RED_CLEARANCE: {Phase.PEDESTRIAN_WALK},
               Phase.PEDESTRIAN_WALK: {Phase.PEDESTRIAN_CLEARANCE},
               Phase.PEDESTRIAN_CLEARANCE: {Phase.VEHICLE_GREEN}}
    rng = random.Random(3)
    c = AdaptiveController()
    prev = c.state.current_phase
    for _ in range(20000):
        d = None if rng.random() < 0.02 else demand(rng.randint(0, 8), rng.uniform(0, 40),
                                                    rng.randint(0, 3), rng.randint(0, 14))
        s = c.step(0.1, d)
        if s.current_phase != prev:
            assert s.current_phase in allowed[prev]
            prev = s.current_phase


def test_state_is_serialisable():
    d = AdaptiveController().state.to_dict()
    assert d["current_phase"] == "VEHICLE_GREEN" and d["vehicle_signal"] == "GREEN"


def test_config_validation():
    for bad in (dict(yellow_time=0), dict(min_vehicle_green=30, max_vehicle_green=10),
                dict(all_red_time=float("nan")), dict(pedestrian_demand_threshold=0),
                dict(fixed_green_time=1)):
        with pytest.raises(ValueError):
            ControllerConfig(**bad)
