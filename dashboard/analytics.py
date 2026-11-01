"""
CrossGuard Traffic Analytics Hub
================================
Presents historical spatial density, pedestrian waiting-time trends, vehicle
queue density, and crosswalk occupancy graphs with informative empty states.
"""

import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    get_command_centre_plotly_layout,
    render_empty_state_card,
    render_hud_header,
    render_metric_card,
    render_status_badge,
)


def _create_empty_plotly_chart(title: str, x_label: str, y_label: str, watermark: str) -> go.Figure:
    """Creates a futuristic dark Plotly chart with clean gridlines and an explicit empty watermark."""
    fig = go.Figure()

    # Configure layout with cyber styling
    layout_cfg = get_command_centre_plotly_layout(title=title, height=280)
    fig.update_layout(**layout_cfg)

    fig.update_xaxes(title_text=x_label, showgrid=True, zeroline=True)
    fig.update_yaxes(title_text=y_label, showgrid=True, zeroline=True)

    # Informative cyberpunk watermark annotation
    fig.add_annotation(
        text=f"<b>[ {watermark} ]</b><br><span style='font-size:10px; color:#64748B;'>Awaiting real-time data logging in Stage 2</span>",
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
    render_hud_header(
        title="Traffic Analytics // Historical Telemetry & Spatial Trends",
        subtitle="Longitudinal performance metrics, waiting times, and intersection throughput",
        badge_text="TELEMETRY DB // STANDBY",
        badge_type="lavender",
    )

    # Filter Toolbar
    fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 1.0])
    with fcol1:
        time_range = st.selectbox(
            "Temporal Window",
            ["Last 15 Minutes", "Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Full Scenario History"],
            index=1,
        )
    with fcol2:
        zone_filter = st.selectbox(
            "Spatial Intersection Filter",
            ["All Crosswalks & Approaches", "North Arterial Crosswalk (Zebra-N)", "South Arterial Crosswalk (Zebra-S)", "Vehicular Approaches (N/S Arterial)"],
            index=0,
        )
    with fcol3:
        st.write("")
        st.button("📥 Export Telemetry (CSV)", disabled=True, help="Database logging will activate in Stage 2.")

    st.write("")

    # Analytics KPI Row (Explicitly Awaiting Data)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card(
            label="TOTAL PEDESTRIANS SERVED",
            value="--",
            unit="PEDS",
            subtext="Awaiting Live Telemetry",
            status_color="muted",
        )
    with kpi2:
        render_metric_card(
            label="MEAN PEDESTRIAN DELAY",
            value="--",
            unit="SEC",
            subtext="Awaiting Dwell Tracker",
            status_color="muted",
        )
    with kpi3:
        render_metric_card(
            label="PEAK QUEUE SATURATION",
            value="--",
            unit="VEH",
            subtext="Awaiting Corridor Detector",
            status_color="muted",
        )
    with kpi4:
        render_metric_card(
            label="CROSSWALK OCCUPANCY RATE",
            value="--",
            unit="%",
            subtext="Awaiting Zone Analyzer",
            status_color="muted",
        )

    st.write("")

    # Explanatory Notice
    render_empty_state_card(
        icon_symbol="📊",
        title="No Live Telemetry Currently Logged",
        description="The 4 analytical trend charts below will populate dynamically once the Stage 2 AI Vision Pipeline processes video frames and records spatial track coordinates to the telemetry store.",
        stage_tag="Telemetry Inactive - Stage 2 Vision Required",
    )

    st.write("")

    # 4 Chart Grid: 2 rows of 2 charts
    chart_row1_col1, chart_row1_col2 = st.columns(2)

    with chart_row1_col1:
        fig1 = _create_empty_plotly_chart(
            title="PEDESTRIAN DENSITY OVER TIME (BY CURB ZONE)",
            x_label="SURVEILLANCE TIMECODE",
            y_label="PEDESTRIAN COUNT",
            watermark="PEDESTRIAN DENSITY EMPTY",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_row1_col2:
        fig2 = _create_empty_plotly_chart(
            title="PEDESTRIAN WAITING-TIME DISTRIBUTION (DWELL DURATION)",
            x_label="WAITING TIME (SECONDS)",
            y_label="FREQUENCY (EVENTS)",
            watermark="WAITING-TIME DISTRIBUTION EMPTY",
        )
        st.plotly_chart(fig2, use_container_width=True)

    chart_row2_col1, chart_row2_col2 = st.columns(2)

    with chart_row2_col1:
        fig3 = _create_empty_plotly_chart(
            title="VEHICULAR QUEUE SATURATION (CORRIDOR FLOW)",
            x_label="SURVEILLANCE TIMECODE",
            y_label="APPROACH VEHICLE QUEUE (UNITS)",
            watermark="VEHICLE QUEUE DENSITY EMPTY",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart_row2_col2:
        fig4 = _create_empty_plotly_chart(
            title="CROSSWALK OCCUPANCY & CLEARANCE DURATION",
            x_label="CYCLE NUMBER",
            y_label="ACTIVE SECONDS IN CROSSWALK",
            watermark="CROSSING OCCUPANCY EMPTY",
        )
        st.plotly_chart(fig4, use_container_width=True)
