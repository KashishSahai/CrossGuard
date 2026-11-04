"""
Bloom City Digital Twin Simulation Engine
=========================================
Microscopic 2D intersection simulator with a distinctive pastel aesthetic.
Synchronizes kinematic vehicles, pedestrian platoons, and traffic signal heads
directly with real-time video spatial analysis and adaptive FSM decisions.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config.settings import BloomCityPalette
from core.adaptive_controller import ControllerDecision, SignalPhase
from core.spatial_analysis import SpatialSnapshot


@dataclass
class SimVehicle:
    """A simulated vehicle in the Bloom City intersection."""
    id: int
    y: float  # Vertical position along roadway (0.0 to 600.0)
    target_y: float
    lane: int  # 0: Left northbound, 1: Right northbound
    speed: float
    color: str
    length: float = 38.0
    width: float = 22.0
    is_stopped: bool = False


@dataclass
class SimPedestrian:
    """A simulated pedestrian moving across or waiting at the crosswalk."""
    id: int
    x: float  # Horizontal position along crosswalk (0.0 to 600.0)
    y: float  # Vertical position
    target_x: float
    speed: float
    state: str  # "WAITING_NORTH", "WAITING_SOUTH", "CROSSING", "COMPLETED"
    color: str = "#7C3AED"


class BloomCityEngine:
    """
    Kinetic simulation engine rendering a pastel 2D digital twin intersection.
    Drives vehicles, pedestrians, and signals directly from vision & FSM telemetry.
    """

    def __init__(self, palette: Optional[BloomCityPalette] = None):
        self.palette = palette or BloomCityPalette()
        self.vehicles: List[SimVehicle] = []
        self.pedestrians: List[SimPedestrian] = []
        self.current_phase: SignalPhase = SignalPhase.VEHICLE_GREEN
        self.countdown_seconds: float = 12.0
        self.is_daylight: bool = True
        self.sim_time: float = 0.0

        # Predefined cute vehicle colors
        self.car_colors = [
            self.palette.car_mint,
            self.palette.car_coral,
            self.palette.car_sky,
            self.palette.car_cream,
        ]

        # Stop bar position for vehicles (y-coordinate)
        self.stop_bar_y = 210.0
        self.crosswalk_y = 260.0

    def sync(
        self,
        snapshot: Optional[SpatialSnapshot],
        decision: Optional[ControllerDecision] = None,
        delta_time: float = 0.1,
        night_mode: bool = False,
        unique_vehicle_count: Optional[int] = None,
        unique_pedestrian_count: Optional[int] = None,
    ) -> None:
        """
        Synchronizes simulation state with vision detections, controller decisions,
        and cumulative tracked entities.
        """
        self.is_daylight = not night_mode
        self.sim_time += delta_time

        if decision is not None:
            self.current_phase = decision.current_phase
            self.countdown_seconds = decision.phase_remaining_seconds
        else:
            self.current_phase = SignalPhase.VEHICLE_GREEN
            self.countdown_seconds = 15.0

        # 1. Synchronize Pedestrians with detected waiting & crossing entities + session unique tracks
        target_waiting_count = snapshot.waiting_pedestrians_count if snapshot else 0
        target_crossing_count = snapshot.crossing_pedestrians_count if snapshot else 0
        active_peds = target_waiting_count + target_crossing_count

        if active_peds > 0:
            total_target_peds = min(active_peds, 6)
        elif unique_pedestrian_count is not None and unique_pedestrian_count > 0:
            total_target_peds = min(unique_pedestrian_count, 3)
        else:
            total_target_peds = 0

        # Adjust pedestrian population to match telemetry
        current_peds = len(self.pedestrians)
        if current_peds < total_target_peds:
            for i in range(current_peds, total_target_peds):
                # Spawn in waiting zone
                pid = i + 1
                spawn_x = 130.0 + (i % 4) * 22.0
                spawn_y = 240.0 + (i // 4) * 20.0
                p = SimPedestrian(
                    id=pid,
                    x=spawn_x,
                    y=spawn_y,
                    target_x=450.0,
                    speed=25.0 + (i % 3) * 5.0,
                    state="WAITING_NORTH",
                )
                self.pedestrians.append(p)
        elif current_peds > total_target_peds:
            self.pedestrians = self.pedestrians[:total_target_peds]

        # Update pedestrian positions based on signal state
        can_walk = self.current_phase in (
            SignalPhase.PEDESTRIAN_WALK,
            SignalPhase.PEDESTRIAN_CLEARANCE,
        )

        for p in self.pedestrians:
            if can_walk:
                p.state = "CROSSING"
                if p.x < p.target_x:
                    p.x += p.speed * delta_time
            else:
                if p.x >= p.target_x:
                    p.state = "COMPLETED"
                elif p.state == "CROSSING":
                    # Mid-crossing when light changed -> continue clearing
                    p.x += p.speed * delta_time * 1.3
                else:
                    p.state = "WAITING_NORTH"

        # 2. Synchronize Vehicles with approach queue density and session vehicle tracks
        approach_vehs = snapshot.approach_vehicles_count if snapshot else 0
        if approach_vehs > 0:
            target_veh_count = min(approach_vehs, 6)
        elif unique_vehicle_count is not None and unique_vehicle_count > 0:
            target_veh_count = min(unique_vehicle_count, 4)
        else:
            target_veh_count = 0

        current_vehs = len(self.vehicles)
        if current_vehs < target_veh_count:
            for i in range(current_vehs, target_veh_count):
                vid = i + 1
                color = self.car_colors[i % len(self.car_colors)]
                lane = i % 2
                start_y = 60.0 - i * 65.0
                v = SimVehicle(
                    id=vid,
                    y=start_y,
                    target_y=self.stop_bar_y - 20.0 - (i // 2) * 48.0,
                    lane=lane,
                    speed=70.0,
                    color=color,
                )
                self.vehicles.append(v)
        elif current_vehs > target_veh_count:
            self.vehicles = self.vehicles[:target_veh_count]

        # Vehicle movement logic
        is_veh_green = self.current_phase == SignalPhase.VEHICLE_GREEN
        has_crossing_ped = any(p.state == "CROSSING" and 190.0 < p.x < 390.0 for p in self.pedestrians)

        for i, v in enumerate(self.vehicles):
            lane_x = 260.0 if v.lane == 0 else 300.0

            # If vehicle green and crosswalk clear -> proceed through intersection
            if is_veh_green and not has_crossing_ped:
                v.y += v.speed * delta_time
                v.is_stopped = False
                if v.y > 560.0:
                    v.y = 20.0  # Loop back to approach
            else:
                # Must stop behind stop bar
                desired_stop_y = self.stop_bar_y - 25.0 - (i // 2) * 50.0
                if v.y < desired_stop_y:
                    v.y = min(desired_stop_y, v.y + v.speed * delta_time)
                    v.is_stopped = False
                else:
                    v.is_stopped = True

    def render_svg(self) -> str:
        """
        Renders a high-resolution 2D overhead SVG intersection with the pastel Bloom City theme.
        """
        bg_col = "#0F172A" if not self.is_daylight else self.palette.background
        road_col = "#334155" if not self.is_daylight else self.palette.road
        grass_col = "#1E293B" if not self.is_daylight else "#E2ECE9"
        curb_col = "#475569" if not self.is_daylight else self.palette.curb

        # Signal state optics
        red_lit = self.current_phase in (
            SignalPhase.ALL_RED_CLEARANCE,
            SignalPhase.PEDESTRIAN_WALK,
            SignalPhase.PEDESTRIAN_CLEARANCE,
            SignalPhase.FAIL_SAFE_STOP,
        )
        yellow_lit = self.current_phase == SignalPhase.VEHICLE_YELLOW
        green_lit = self.current_phase == SignalPhase.VEHICLE_GREEN

        ped_walk_lit = self.current_phase == SignalPhase.PEDESTRIAN_WALK
        ped_clear_lit = self.current_phase == SignalPhase.PEDESTRIAN_CLEARANCE

        r_glow = "filter='url(#glow-red)'" if red_lit else "opacity='0.25'"
        y_glow = "filter='url(#glow-amber)'" if yellow_lit else "opacity='0.25'"
        g_glow = "filter='url(#glow-green)'" if green_lit else "opacity='0.25'"

        # Pedestrian head colors
        ped_light_col = "#10B981" if (ped_walk_lit or (ped_clear_lit and int(self.sim_time * 4) % 2 == 0)) else "#EF4444"
        ped_icon = "🚶 WALK" if (ped_walk_lit or ped_clear_lit) else "✋ WAIT"

        # Build vehicles SVG
        vehicles_svg = []
        for v in self.vehicles:
            lane_x = 255.0 if v.lane == 0 else 295.0
            car_svg = f"""
            <g transform="translate({lane_x}, {v.y})">
                <!-- Drop shadow -->
                <rect x="-10" y="-18" width="{v.width}" height="{v.length}" rx="6" fill="rgba(0,0,0,0.18)" transform="translate(2, 3)" />
                <!-- Car body -->
                <rect x="-10" y="-18" width="{v.width}" height="{v.length}" rx="6" fill="{v.color}" stroke="#FFFFFF" stroke-width="1.5" />
                <!-- Windshield -->
                <rect x="-8" y="-12" width="18" height="8" rx="2" fill="#334155" opacity="0.8" />
                <!-- Roof -->
                <rect x="-7" y="-2" width="16" height="12" rx="3" fill="#FFFFFF" opacity="0.4" />
                <!-- Rear window -->
                <rect x="-8" y="11" width="18" height="5" rx="1.5" fill="#334155" opacity="0.8" />
                <!-- Headlights -->
                <circle cx="-6" cy="-17" r="2.5" fill="#FEF08A" />
                <circle cx="8" cy="-17" r="2.5" fill="#FEF08A" />
                <!-- Brake lights -->
                <circle cx="-6" cy="19" r="2" fill="{'#EF4444' if v.is_stopped else '#991B1B'}" />
                <circle cx="8" cy="19" r="2" fill="{'#EF4444' if v.is_stopped else '#991B1B'}" />
            </g>
            """
            vehicles_svg.append(car_svg)

        # Build pedestrians SVG
        peds_svg = []
        for p in self.pedestrians:
            ped_svg = f"""
            <g transform="translate({p.x}, {p.y})">
                <circle cx="1" cy="2" r="7" fill="rgba(0,0,0,0.2)" />
                <circle cx="0" cy="0" r="7" fill="{p.color}" stroke="#FFFFFF" stroke-width="1.5" />
                <circle cx="0" cy="-2" r="3" fill="#F8FAFC" />
            </g>
            """
            peds_svg.append(ped_svg)

        # Build Zebra stripes (vertical road, horizontal zebra crosswalk)
        stripes_svg = []
        for sx in range(235, 335, 16):
            stripes_svg.append(
                f'<rect x="{sx}" y="248" width="10" height="26" fill="{self.palette.crosswalk_stripe}" opacity="0.9" rx="2" />'
            )

        svg = f"""
        <svg viewBox="0 0 580 480" width="100%" height="440" xmlns="http://www.w3.org/2000/svg" style="border-radius: 8px; background: {bg_col};">
            <defs>
                <filter id="glow-red" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="glow-amber" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="glow-green" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <!-- Background Grass / City blocks -->
            <rect x="0" y="0" width="580" height="480" fill="{grass_col}" />

            <!-- Curbs and Sidewalks -->
            <rect x="220" y="0" width="130" height="480" fill="{curb_col}" rx="4" />
            <rect x="0" y="235" width="580" height="52" fill="{curb_col}" rx="4" />

            <!-- Road Surface (Arterial North/South) -->
            <rect x="230" y="0" width="110" height="480" fill="{road_col}" />

            <!-- Pedestrian Waiting Curb Zones (Pastel Lavender) -->
            <rect x="120" y="238" width="105" height="46" rx="6" fill="{self.palette.waiting_zone}" stroke="#A78BFA" stroke-width="1.5" stroke-dasharray="4 3" />
            <text x="172" y="265" font-family="Inter, sans-serif" font-size="10" font-weight="700" fill="#6D28D9" text-anchor="middle">WAITING CURB (N)</text>

            <rect x="345" y="238" width="115" height="46" rx="6" fill="{self.palette.waiting_zone}" stroke="#A78BFA" stroke-width="1.5" stroke-dasharray="4 3" />
            <text x="402" y="265" font-family="Inter, sans-serif" font-size="10" font-weight="700" fill="#6D28D9" text-anchor="middle">WAITING CURB (S)</text>

            <!-- Lane Divider Dashed Line -->
            <line x1="285" y1="0" x2="285" y2="210" stroke="{self.palette.lane_marking}" stroke-width="2.5" stroke-dasharray="10 8" opacity="0.75" />
            <line x1="285" y1="280" x2="285" y2="480" stroke="{self.palette.lane_marking}" stroke-width="2.5" stroke-dasharray="10 8" opacity="0.75" />

            <!-- Vehicle Stop Line -->
            <line x1="230" y1="{self.stop_bar_y}" x2="340" y2="{self.stop_bar_y}" stroke="{self.palette.vehicle_stop_bar}" stroke-width="4.5" stroke-linecap="round" />

            <!-- Zebra Crosswalk Corridor -->
            <rect x="230" y="244" width="110" height="34" fill="{self.palette.crossing_zone}" />
            {''.join(stripes_svg)}

            <!-- Vehicles -->
            {''.join(vehicles_svg)}

            <!-- Pedestrians -->
            {''.join(peds_svg)}

            <!-- Traffic Signal Pole & Head (East-side of stop line) -->
            <g transform="translate(355, 175)">
                <!-- Housing -->
                <rect x="0" y="0" width="30" height="74" rx="8" fill="#0B0F19" stroke="#334155" stroke-width="2" />
                <!-- Red Light -->
                <circle cx="15" cy="14" r="8" fill="#EF4444" {r_glow} />
                <!-- Amber Light -->
                <circle cx="15" cy="37" r="8" fill="#F59E0B" {y_glow} />
                <!-- Green Light -->
                <circle cx="15" cy="60" r="8" fill="#10B981" {g_glow} />
            </g>

            <!-- Pedestrian Signal Indicator Box -->
            <g transform="translate(355, 260)">
                <rect x="0" y="0" width="85" height="42" rx="6" fill="#0B0F19" stroke="#334155" stroke-width="1.5" />
                <text x="42" y="18" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="700" fill="{ped_light_col}" text-anchor="middle">{ped_icon}</text>
                <text x="42" y="34" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="700" fill="#00F0FF" text-anchor="middle">{self.countdown_seconds:.0f}s</text>
            </g>

            <!-- Intersection Label -->
            <rect x="14" y="14" width="200" height="36" rx="6" fill="rgba(15, 23, 42, 0.75)" />
            <text x="24" y="30" font-family="Inter, sans-serif" font-size="11" font-weight="800" fill="#F8FAFC">BLOOM CITY // SECTOR 7</text>
            <text x="24" y="43" font-family="'JetBrains Mono', monospace" font-size="9" fill="#A78BFA">PHASE: {self.current_phase.value}</text>
        </svg>
        """
        return svg

    def render_intersection_svg(
        self,
        snapshot: Optional[SpatialSnapshot] = None,
        decision: Optional[ControllerDecision] = None,
    ) -> str:
        """Convenience method syncing state if provided and returning rendered SVG."""
        if snapshot is not None and decision is not None:
            self.sync(snapshot, decision)
        return self.render_svg()
