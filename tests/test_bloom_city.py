import json
import re

import pytest

from core.adaptive_controller import AdaptiveController
from core.models import (PHASE_OUTPUTS, PHASE_ORDER, ControllerConfig, PedestrianSignal, Phase,
                         VehicleSignal)
from simulation import geometry as G
from simulation.bloom_city_engine import (DT, MIN_GAP, SCENARIOS, BloomCityEngine, build_frames)
from simulation.bloom_city_renderer import render_bloom_city_html
from simulation.simulation_agents import (PedestrianAgent, PedestrianState, VehicleAgent,
                                          VehicleKind, VehicleState)


def held(phase, scenario="quiet"):
    """Engine with no random traffic, frozen in one controller phase."""
    e = BloomCityEngine(scenario=SCENARIOS[scenario].__class__(
        "t", "t", "t", (0, 0, 0, 0), 0, 0))
    e.controller.force_phase(phase, hold=True)
    e.step()
    return e


# ------------------------------------------------------------------ signal mapping
def test_vehicle_green_maps_to_green_and_dont_walk():
    e = held(Phase.VEHICLE_GREEN)
    assert e.ctrl.vehicle_signal == VehicleSignal.GREEN
    assert e.ctrl.pedestrian_signal == PedestrianSignal.DONT_WALK


def test_pedestrian_walk_maps_to_red_and_walk():
    e = held(Phase.PEDESTRIAN_WALK)
    assert e.ctrl.vehicle_signal == VehicleSignal.RED
    assert e.ctrl.pedestrian_signal == PedestrianSignal.WALK


def test_every_phase_reaches_the_simulation_unchanged():
    for ph, (v, p) in PHASE_OUTPUTS.items():
        e = held(ph)
        assert (e.ctrl.vehicle_signal, e.ctrl.pedestrian_signal) == (v, p)


# ------------------------------------------------------------------ vehicles
def test_vehicle_moves_on_green():
    e = held(Phase.VEHICLE_GREEN)
    v = e.place_vehicle(0, 160, speed=40)
    s0 = v.s
    e.run(3.0)
    assert v.s > s0 + 100 and v.speed > 30


def test_vehicle_stops_before_the_line_on_red():
    for approach in range(4):
        e = held(Phase.PEDESTRIAN_WALK)
        v = e.place_vehicle(approach, 200, speed=85)
        e.run(8.0)
        assert v.state == VehicleState.STOPPED and v.speed == 0
        assert v.front <= G.STOP_LINE_S[approach]


def test_vehicle_stops_on_all_red_and_remains_stopped_during_clearance():
    for ph in (Phase.ALL_RED_CLEARANCE, Phase.PEDESTRIAN_CLEARANCE):
        e = held(ph)
        v = e.place_vehicle(1, 150, speed=60)
        e.run(10.0)
        assert v.speed == 0 and v.front <= G.STOP_LINE_S[1]


def test_vehicles_queue_with_spacing_and_resume_on_green():
    e = held(Phase.PEDESTRIAN_WALK)
    cars = [e.place_vehicle(2, 10)]
    for kind in (VehicleKind.BUS, VehicleKind.CAR, VehicleKind.TRUCK):
        behind_of = G.STOP_LINE_S[2] - cars[-1].rear          # distance of leader's rear to the line
        cars.append(e.place_vehicle(2, behind_of + 30, kind, speed=60))
    e.run(10.0)
    ordered = sorted(cars, key=lambda v: -v.s)
    for ahead, behind in zip(ordered, ordered[1:]):
        assert ahead.rear - behind.front >= MIN_GAP - 0.5
    assert all(v.speed == 0 for v in cars)
    e.controller.force_phase(Phase.VEHICLE_GREEN, hold=True)
    e.run(6.0)
    assert all(v.speed > 20 for v in cars[:2])
    assert all(v.front > G.STOP_LINE_S[2] for v in cars[:2])


def test_yellow_stops_vehicles_that_can_stop_and_clears_those_that_cannot():
    e = held(Phase.VEHICLE_YELLOW)
    far = e.place_vehicle(0, 190, speed=80)
    near = e.place_vehicle(2, 6, speed=85)
    e.run(1.0)
    e.run(6.0)
    assert far.front <= G.STOP_LINE_S[0] and far.speed == 0
    assert near.front > G.STOP_LINE_S[2]            # committed vehicle cleared through


def test_cross_traffic_does_not_enter_the_box_together():
    e = held(Phase.VEHICLE_GREEN)
    for a in range(4):
        e.place_vehicle(a, 30 + 5 * a, speed=60)
    for _ in range(int(14 / DT)):
        e.step()
        for i, v in enumerate(e.vehicles):
            for w in e.vehicles[i + 1:]:
                if v.axis != w.axis:
                    assert not G.rects_overlap(v.bbox(), w.bbox())


# ------------------------------------------------------------------ pedestrians
def test_pedestrian_waits_on_dont_walk():
    for ph in (Phase.VEHICLE_GREEN, Phase.VEHICLE_YELLOW, Phase.ALL_RED_CLEARANCE):
        e = held(ph)
        p = e.place_pedestrian(0, 1, 0)
        pos = p.position
        e.run(5.0)
        assert p.state == PedestrianState.WAITING and p.progress == 0 and p.position == pos
        assert p.wait_time > 4.0


def test_pedestrian_crosses_on_walk_with_vehicles_stopped():
    e = held(Phase.PEDESTRIAN_WALK)
    p = e.place_pedestrian(2, -1, 1)
    start = p.position
    e.run(2.5)
    assert p.state == PedestrianState.CROSSING and p.progress > 0
    assert p.position[0] < start[0]                       # moving west along the south zebra


def test_pedestrian_finishes_during_clearance_but_none_start_then():
    e = held(Phase.PEDESTRIAN_WALK)
    crossing = e.place_pedestrian(0, 1, 0)
    e.run(1.0)
    assert crossing.state == PedestrianState.CROSSING
    e.controller.force_phase(Phase.PEDESTRIAN_CLEARANCE, hold=True)
    late = e.place_pedestrian(0, -1, 0)
    e.run(8.0)
    assert crossing not in e.pedestrians                  # reached the far curb and left
    assert late.state == PedestrianState.WAITING and late.progress == 0


def test_pedestrian_does_not_step_out_in_front_of_a_vehicle_on_the_crosswalk():
    e = held(Phase.PEDESTRIAN_WALK)
    z = G.ZEBRAS[0]
    car = e.place_vehicle(0, 0)
    car.s = G.STOP_LINE_S[0] + 30                         # sitting on the zebra
    p = e.place_pedestrian(0, 1, 0)
    e.controller.force_phase(Phase.PEDESTRIAN_WALK, hold=True)
    e.step()
    assert p.state == PedestrianState.WAITING


def test_vehicle_will_not_enter_crossing_with_a_pedestrian_on_it():
    e = held(Phase.VEHICLE_GREEN)
    p = e.place_pedestrian(0, 1, 0)
    p.state = PedestrianState.CROSSING
    p.progress = 0.4
    v = e.place_vehicle(0, 120, speed=60)
    e.run(1.0)
    p.progress = 0.4                                       # keep the walker on the zebra
    for _ in range(int(4 / DT)):
        p.progress = 0.4
        e.step()
    assert v.front <= G.STOP_LINE_S[0] and v.speed == 0


# ------------------------------------------------------------------ controller -> simulation, end to end
def test_changing_controller_state_changes_simulation_behaviour():
    e = held(Phase.VEHICLE_GREEN)
    v = e.place_vehicle(3, 220, speed=70)
    p = e.place_pedestrian(1, 1, 0)
    e.run(1.0)
    assert v.speed > 0 and p.state == PedestrianState.WAITING
    e.controller.force_phase(Phase.PEDESTRIAN_WALK, hold=True)
    e.run(8.0)
    assert v.speed == 0 and v.front <= G.STOP_LINE_S[3]
    assert p.state == PedestrianState.CROSSING or p not in e.pedestrians


def test_full_cycle_safety_invariants_over_long_runs():
    for key in ("ped_surge", "heavy_traffic", "balanced"):
        e = BloomCityEngine(scenario=key, seed=11)
        prev_front = {}
        since_red = 99.0
        for _ in range(int(240 / DT)):
            st = e.step()
            ph = st.current_phase
            red_age = st.phase_elapsed if ph == Phase.ALL_RED_CLEARANCE else (
                99.0 if ph in (Phase.PEDESTRIAN_WALK, Phase.PEDESTRIAN_CLEARANCE) else -1)
            crossing = {p.crosswalk for p in e.pedestrians if p.state == PedestrianState.CROSSING}
            for v in e.vehicles:
                # no vehicle sits on a crosswalk while a pedestrian is crossing it
                for cw in crossing:
                    assert not G.rects_overlap(v.bbox(), G.ZEBRAS[cw]), (key, e.time)
                # no vehicle crosses its stop line well after the light went red
                line = G.STOP_LINE_S[v.approach]
                was = prev_front.get(v.id)
                if was is not None and was < line - 1.0 <= v.front and red_age > 1.0:
                    raise AssertionError(f"{key}: vehicle ran the red at t={e.time:.1f}")
                prev_front[v.id] = v.front
            # no two vehicles overlap
            for i, v in enumerate(e.vehicles):
                for w in e.vehicles[i + 1:]:
                    assert not G.rects_overlap(v.bbox(), w.bbox()), (key, e.time)
        assert e.ctrl.cycles_completed >= 3


def test_recorded_frame_times_start_at_zero_and_increase():
    frames, _, _ = build_frames("balanced", 12, 9)      # non-zero warm-up must not leak into t
    ts = [f["t"] for f in frames]
    assert 0.0 <= ts[0] <= 0.1 and all(b > a for a, b in zip(ts, ts[1:])) and 11.0 < ts[-1] < 12.1


def test_simulation_is_deterministic():
    a, _, _ = build_frames("balanced", 20, 5, seed=5)
    b, _, _ = build_frames("balanced", 20, 5, seed=5)
    assert a == b


def test_agents_are_representative_objects_not_detections():
    e = BloomCityEngine(scenario="balanced")
    e.run(30)
    assert e.vehicles and all(isinstance(v, VehicleAgent) for v in e.vehicles)
    assert all(isinstance(p, PedestrianAgent) for p in e.pedestrians)


def test_demand_is_measured_from_representative_agents():
    e = held(Phase.VEHICLE_GREEN)
    for i in range(3):
        e.place_pedestrian(0, 1, i)
    e.place_vehicle(0, 100)
    e.run(2.0)
    assert e.demand.pedestrians_waiting == 3 and e.demand.vehicle_demand >= 1
    assert e.demand.longest_pedestrian_wait_s >= 1.9


# ------------------------------------------------------------------ rendering
def _payload(html):
    m = re.search(r'<script id="bcdata" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_renderer_output_is_non_empty_and_has_expected_elements():
    frames, reasons, _ = build_frames("balanced", 10, 5)
    html = render_bloom_city_html(frames, reasons)
    assert len(html) > 20_000 and "<svg" in html
    for needle in ('id="vehicles"', 'id="peds"', 'id="vl0r"', 'id="vl3g"', 'id="pl3"',
                   "CAFÉ BLOOM", "NORTH", "Representative agents derived from",
                   "SCENARIO DEMO", 'id="phases"', 'id="dec"'):
        assert needle in html, needle
    # zebra stripes, stop lines and 4 signal heads present
    assert html.count('fill="#fffdfb"') >= 40   # 40 zebra bars + 4 stop lines
    assert all(f'id="vh{i}"' in html for i in range(4))


def test_frame_signals_match_the_controller_phase():
    frames, reasons, _ = build_frames("ped_surge", 70, 5)
    colors = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    peds = {"DONT_WALK": 0, "WALK": 1, "CLEARING": 2}
    seen = set()
    for f in _payload(render_bloom_city_html(frames, reasons))["frames"]:
        ph = PHASE_ORDER[f["ph"]]
        v, p = PHASE_OUTPUTS[ph]
        assert f["vs"] == colors[v.value] and f["ps"] == peds[p.value]
        seen.add(ph)
    assert seen == set(PHASE_ORDER)


def test_day_and_night_themes_differ_and_label_mode():
    frames, reasons, _ = build_frames("quiet", 5, 2)
    day = render_bloom_city_html(frames, reasons, theme="day")
    night = render_bloom_city_html(frames, reasons, theme="night")
    assert 'fill="url(#glow)"' not in day and 'fill="url(#glow)"' in night
    live = render_bloom_city_html(frames, reasons, mode="LIVE VIDEO",
                                  source_note="Representative agents derived from video telemetry")
    assert "LIVE VIDEO" in live and "derived from video telemetry" in live


def test_renderer_rejects_empty_input():
    with pytest.raises(ValueError):
        render_bloom_city_html([], [])
