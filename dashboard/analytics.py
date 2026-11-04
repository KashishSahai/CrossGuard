"""
CrossGuard Traffic Analytics Hub
================================
Presents historical spatial density, pedestrian waiting-time trends, vehicle
queue density, and crosswalk occupancy graphs with live Plotly visualization and CSV export.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.session_manager import VideoSession
from core.telemetry_manager import TelemetryManager
from dashboard.components import (
    get_command_centre_plotly_layout,
    render_empty_state_card,
    render_html,
    render_hud_header,
    render_metric_card,
    render_status_badge,
    render_tech_table,
)


def _create_empty_plotly_chart(title: str, x_label: str, y_label: str, watermark: str) -> go.Figure:
    """Creates a futuristic dark Plotly chart with clean gridlines and an explicit empty watermark."""
    fig = go.Figure()
    layout_cfg = get_command_centre_plotly_layout(title=title, height=270)
    fig.update_layout(**layout_cfg)
    fig.update_xaxes(title_text=x_label, showgrid=True, zeroline=True)
    fig.update_yaxes(title_text=y_label, showgrid=True, zeroline=True)

    fig.add_annotation(
        text=f"<b>[ {watermark} ]</b><br><span style='font-size:10px; color:#64748B;'>Upload footage and run the Vision Monitor to populate measurements</span>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=12, color="rgba(0, 240, 255, 0.6)", family="JetBrains Mono, monospace"),
        align="center",
        bgcolor="rgba(13, 19, 34, 0.7)",
        bordercolor="rgba(0, 240, 255, 0.25)",
        borderwidth=1,
        borderpad=10,
    )
    return fig


def render_analytics_page() -> None:
    """Renders the Traffic Analytics historical telemetry dashboard."""
    session: VideoSession = st.session_state["video_session"]
    telemetry: TelemetryManager = session.telemetry_manager
    df = telemetry.get_dataframe()
    stats = telemetry.get_summary_stats()
    diag = session.diagnostics
    has_data = stats["has_data"] and len(df) > 0

    render_hud_header(
        title="Traffic Analytics // Historical Telemetry & Spatial Trends",
        subtitle="Longitudinal performance metrics, waiting times, and intersection throughput",
        badge_text="TELEMETRY DB ACTIVE" if has_data else "AWAITING TELEMETRY",
        badge_type="active" if has_data else "warning",
    )

    # Filter & Export Toolbar
    fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 1.0])
    with fcol1:
        time_window = st.selectbox(
            "Temporal Window",
            ["Full Session History", "Last 50 Recorded Frames", "Last 20 Recorded Frames"],
            index=0,
        )
    with fcol2:
        zone_filter = st.selectbox(
            "Spatial Zone Filter",
            ["All Intersection Approaches", "Pedestrian Waiting Zones", "Zebra Crossing Corridor"],
            index=0,
        )
    with fcol3:
        st.write("")
        csv_bytes = telemetry.get_csv_bytes()
        st.download_button(
            label="📥 Download Session CSV",
            data=csv_bytes,
            file_name=f"crossguard_telemetry_{len(df)}_records.csv",
            mime="text/csv",
            disabled=df.empty,
            help="Download recorded frame-by-frame telemetry as CSV.",
        )

    st.write("")

    # Filter dataframe if needed
    if not df.empty:
        df_filtered = df.copy()

        # 1. Temporal filter
        if time_window == "Last 20 Recorded Frames":
            df_filtered = df_filtered.tail(20)
        elif time_window == "Last 50 Recorded Frames":
            df_filtered = df_filtered.tail(50)

        # 2. Spatial filter
        if zone_filter == "Pedestrian Waiting Zones":
            df_filtered = df_filtered[df_filtered["pedestrians_waiting"] > 0]
        elif zone_filter == "Zebra Crossing Corridor":
            df_filtered = df_filtered[(df_filtered["pedestrians_crossing"] > 0) | (df_filtered["crosswalk_occupied"] == 1)]

        df_plot = df_filtered
        if df_plot.empty and has_data:
            st.info(f"ℹ️ No recorded frames match spatial filter '{zone_filter}' within '{time_window}'. Reverting to temporal window.")
            df_plot = df.tail(20) if time_window == "Last 20 Recorded Frames" else (df.tail(50) if time_window == "Last 50 Recorded Frames" else df)
    else:
        df_plot = pd.DataFrame()

    max_wait_time = float(df["max_wait_seconds"].max()) if (has_data and "max_wait_seconds" in df) else 0.0

    # Analytics KPI Grid: 6 Real Session Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        served_cnt = stats["total_served"]
        observed_cnt = stats.get("total_observed_peds", 0)
        peds_disp = str(served_cnt) if served_cnt > 0 else (str(observed_cnt) if has_data else "N/A")
        subtext_ped = "Crossing Completed" if served_cnt > 0 else ("Observed & Tracked" if observed_cnt > 0 else "Awaiting video processing")
        render_metric_card(
            label="PEDS SERVED",
            value=peds_disp,
            unit="PEDS" if has_data else "",
            subtext=subtext_ped,
            status_color="cyan" if has_data else "muted",
        )
    with kpi2:
        render_metric_card(
            label="MAX WAIT TIME",
            value=f"{max_wait_time:.1f}" if has_data else "N/A",
            unit="SEC" if has_data else "",
            subtext="Peak Dwell Time Observed" if has_data else "Awaiting video processing",
            status_color="lavender" if has_data else "muted",
        )
    with kpi3:
        render_metric_card(
            label="AVG WAIT TIME",
            value=f"{stats['avg_wait_time']:.1f}" if has_data else "N/A",
            unit="SEC" if has_data else "",
            subtext="Mean Dwell Across Platoons" if has_data else "Awaiting video processing",
            status_color="cyan" if has_data else "muted",
        )
    with kpi4:
        render_metric_card(
            label="PEAK QUEUE",
            value=str(stats["peak_queue"]) if has_data else "N/A",
            unit="VEH" if has_data else "",
            subtext="Max Vehicles in Approach" if has_data else "Awaiting video processing",
            status_color="active" if has_data else "muted",
        )
    with kpi5:
        render_metric_card(
            label="PRIORITY CALLS",
            value=str(stats["priority_activations"]) if has_data else "N/A",
            unit="ACTS" if has_data else "",
            subtext="FSM Cases C & D Preempts" if has_data else "Awaiting video processing",
            status_color="cyan" if has_data else "muted",
        )
    with kpi6:
        render_metric_card(
            label="CLEARANCE EXT",
            value=str(stats["clearance_extensions"]) if has_data else "N/A",
            unit="HOLDS" if has_data else "",
            subtext="FSM Case E Occupancy Holds" if has_data else "Awaiting video processing",
            status_color="warning" if stats.get("clearance_extensions", 0) > 0 else ("active" if has_data else "muted"),
        )

    st.write("")

    # Vehicle Detections by Class Breakdown
    render_hud_header(
        title="Intersection Fleet & Object Classification",
        subtitle="Cumulative objects detected and tracked by YOLOv8n & ByteTrack",
        badge_text=f"{diag.total_detections_count} FRAME INSTANCES | {len(diag.unique_track_ids)} UNIQUE TRACKS",
        badge_type="cyan" if has_data else "neutral",
    )

    class_counts = diag.detections_by_class
    classes_meta = [
        ("car", "Car", "Vehicle Approach Corridor"),
        ("motorcycle", "Motorcycle", "Vehicle Approach Corridor"),
        ("bus", "Bus", "Vehicle Approach Corridor"),
        ("truck", "Truck", "Vehicle Approach Corridor"),
        ("person", "Person (Pedestrian)", "North / South Waiting & Crosswalk"),
    ]

    fleet_rows = []
    for c_id, label, zone_str in classes_meta:
        inst_cnt = class_counts.get(c_id, 0)
        uniq_cnt = len(diag.unique_tracks_by_class.get(c_id, set()))
        fleet_rows.append([
            f"<strong style='color:#F8FAFC;'>{label}</strong>",
            f"{inst_cnt} detections",
            f"<strong style='color:#00F0FF;'>{uniq_cnt} unique tracks</strong>",
            zone_str,
            f"<span class='status-pill {'cyan' if inst_cnt > 0 else 'neutral'}'>{'TRACKED' if inst_cnt > 0 else 'STANDBY'}</span>",
        ])

    render_tech_table(
        headers=["OBJECT CLASS", "FRAME DETECTIONS", "UNIQUE TRACK ENTITIES", "ZONE CORRIDOR", "STATUS"],
        rows=fleet_rows,
        caption="Multi-class detections distinguishing temporal frame instances from persistent ByteTrack unique entities",
    )

    st.write("")

    # If no data is recorded yet, render informative guidance
    if df_plot.empty:
        render_empty_state_card(
            icon_symbol="📊",
            title="No Session Telemetry Recorded Yet",
            description="Run the AI Vision Monitor on a video stream or sample clip to record live tracking coordinates, FSM transitions, and dwell times.",
            stage_tag="Awaiting Vision Pipeline Processing",
        )

    # 4 Chart Grid: 2 rows of 2 charts
    chart_row1_col1, chart_row1_col2 = st.columns(2)

    with chart_row1_col1:
        if not df_plot.empty and "pedestrians_waiting" in df_plot:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["pedestrians_waiting"],
                mode="lines+markers",
                name="Waiting Peds",
                line=dict(color="#00F0FF", width=2.5),
                marker=dict(size=5, color="#00F0FF"),
            ))
            fig1.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["pedestrians_crossing"],
                mode="lines+markers",
                name="Crossing Peds",
                line=dict(color="#A78BFA", width=2, dash="dash"),
                marker=dict(size=4, color="#A78BFA"),
            ))
            layout1 = get_command_centre_plotly_layout("PEDESTRIAN VOLUME OVER TIME (WAITING VS. CROSSING)", height=270)
            fig1.update_layout(**layout1)
            fig1.update_xaxes(title_text="Video Timestamp (s)")
            fig1.update_yaxes(title_text="Pedestrians")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            fig1 = _create_empty_plotly_chart(
                title="PEDESTRIAN VOLUME OVER TIME (WAITING VS. CROSSING)",
                x_label="SURVEILLANCE TIMECODE (s)",
                y_label="PEDESTRIAN COUNT",
                watermark="NO PEDESTRIAN TELEMETRY",
            )
            st.plotly_chart(fig1, use_container_width=True)

    with chart_row1_col2:
        if not df_plot.empty and "max_wait_seconds" in df_plot:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_plot["timestamp"],
                y=df_plot["max_wait_seconds"],
                name="Max Dwell Time",
                marker_color="#F43F5E",
                opacity=0.85,
            ))
            layout2 = get_command_centre_plotly_layout("PEDESTRIAN WAITING DURATION (MAX DWELL TIME)", height=270)
            fig2.update_layout(**layout2)
            fig2.update_xaxes(title_text="Video Timestamp (s)")
            fig2.update_yaxes(title_text="Dwell Time (Seconds)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = _create_empty_plotly_chart(
                title="PEDESTRIAN WAITING DURATION (MAX DWELL TIME)",
                x_label="SURVEILLANCE TIMECODE (s)",
                y_label="DWELL TIME (SECONDS)",
                watermark="NO DWELL TIME TELEMETRY",
            )
            st.plotly_chart(fig2, use_container_width=True)

    chart_row2_col1, chart_row2_col2 = st.columns(2)

    with chart_row2_col1:
        if not df_plot.empty and "vehicles_approach" in df_plot:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["vehicles_approach"],
                mode="lines+markers",
                name="Approach Vehicles",
                line=dict(color="#10B981", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.15)",
            ))
            layout3 = get_command_centre_plotly_layout("VEHICLE QUEUE DENSITY (APPROACH LANES)", height=270)
            fig3.update_layout(**layout3)
            fig3.update_xaxes(title_text="Video Timestamp (s)")
            fig3.update_yaxes(title_text="Vehicles in Queue")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3 = _create_empty_plotly_chart(
                title="VEHICLE QUEUE DENSITY (APPROACH LANES)",
                x_label="SURVEILLANCE TIMECODE (s)",
                y_label="VEHICLES IN QUEUE",
                watermark="NO QUEUE TELEMETRY",
            )
            st.plotly_chart(fig3, use_container_width=True)

    with chart_row2_col2:
        if not df_plot.empty and "crosswalk_occupied" in df_plot:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["crosswalk_occupied"],
                mode="lines",
                name="Crosswalk Occupied",
                line=dict(color="#F59E0B", width=2, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.15)",
            ))
            layout4 = get_command_centre_plotly_layout("CROSSWALK OCCUPANCY TIMELINE (ACTIVE RESERVATIONS)", height=270)
            fig4.update_layout(**layout4)
            fig4.update_xaxes(title_text="Video Timestamp (s)")
            fig4.update_yaxes(title_text="Occupied (1) / Clear (0)", tickvals=[0, 1])
            st.plotly_chart(fig4, use_container_width=True)
        else:
            fig4 = _create_empty_plotly_chart(
                title="CROSSWALK OCCUPANCY TIMELINE (ACTIVE RESERVATIONS)",
                x_label="SURVEILLANCE TIMECODE (s)",
                y_label="OCCUPIED (1) / CLEAR (0)",
                watermark="NO OCCUPANCY TELEMETRY",
            )
            st.plotly_chart(fig4, use_container_width=True)
