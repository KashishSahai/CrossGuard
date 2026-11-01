"""
CrossGuard Reusable Presentation Components
===========================================
High-tech command center visual primitives: illuminated headers, metric cards,
radar pulses, traffic signal heads, empty-state containers, and Plotly cyber themes.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st


def inject_custom_theme() -> None:
    """Loads and injects the futuristic command centre CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "..", "assets", "styles", "theme.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_top_command_bar(
    active_page_title: str,
    video_loaded: bool = False,
    video_name: Optional[str] = None,
) -> None:
    """
    Renders a compact top surveillance status bar with live pulse indicator,
    current system timestamp, active section, and video stream status.
    """
    pulse_class = "pulse-dot" if video_loaded else "pulse-dot warning"
    video_status_str = f"FEED: {video_name}" if video_loaded else "FEED: NO STREAM LOADED"
    video_badge_class = "active" if video_loaded else "warning"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <div class="top-command-bar">
        <div class="top-bar-left">
            <span class="{pulse_class}"></span>
            <span style="font-weight: 700; letter-spacing: 0.08em; font-size: 0.88rem; color: #F8FAFC;">
                CROSSGUARD // COMMAND CENTRE
            </span>
            <span style="color: #64748B;">|</span>
            <span style="font-family: var(--font-mono); font-size: 0.78rem; color: #00F0FF; letter-spacing: 0.05em;">
                SECTION: {active_page_title.upper()}
            </span>
        </div>
        <div class="top-bar-right">
            <span class="status-pill {video_badge_class}">{video_status_str}</span>
            <span class="status-pill cyan">STAGE 1 FOUNDATION</span>
            <span style="color: #94A3B8; font-size: 0.8rem;">{now_str}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_hud_header(
    title: str,
    subtitle: str = "",
    badge_text: Optional[str] = None,
    badge_type: str = "cyan",
) -> None:
    """Renders a section header with neon cyber accents and optional status badge."""
    badge_html = (
        f'<span class="status-pill {badge_type}">{badge_text}</span>'
        if badge_text
        else ""
    )
    html = f"""
    <div class="hud-header">
        <div>
            <div class="hud-title">
                <span class="glow">▶</span> {title}
            </div>
            {f'<div class="hud-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    unit: str = "",
    subtext: str = "",
    status_color: str = "cyan",
    delta: Optional[str] = None,
) -> None:
    """
    Renders a compact information-rich HUD metric container.
    """
    delta_html = ""
    if delta:
        delta_html = f'<span style="font-size: 0.75rem; color: #10B981; margin-left: 6px;">{delta}</span>'

    html = f"""
    <div class="metric-box">
        <div class="label">
            <span>{label}</span>
            {delta_html}
        </div>
        <div class="value {status_color}">
            {value} <span style="font-size: 0.95rem; font-weight: 500; color: #94A3B8;">{unit}</span>
        </div>
        <div class="subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(text: str, badge_type: str = "neutral") -> str:
    """Returns HTML for an inline status pill badge."""
    return f'<span class="status-pill {badge_type}">{text}</span>'


def render_traffic_signal_head(
    red_active: bool = True,
    amber_active: bool = False,
    green_active: bool = False,
    ped_state: str = "WAIT",
    countdown: str = "--",
) -> None:
    """
    Renders a realistic digital traffic signal head with glowing lenses,
    illuminated visors, and pedestrian clearance walk indicator.
    """
    r_class = "signal-lens red active" if red_active else "signal-lens red"
    a_class = "signal-lens amber active" if amber_active else "signal-lens amber"
    g_class = "signal-lens green active" if green_active else "signal-lens green"

    ped_color = "#EF4444" if ped_state in ("WAIT", "DONT_WALK") else "#10B981"
    ped_icon = "✋" if ped_state in ("WAIT", "DONT_WALK") else "🚶"
    ped_label = "DON'T WALK" if ped_state in ("WAIT", "DONT_WALK") else "WALK"

    html = f"""
    <div class="hud-card" style="display: flex; flex-direction: column; align-items: center; padding: 18px;">
        <div style="font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.1em; color: #94A3B8; margin-bottom: 12px; text-transform: uppercase;">
            SIGNAL HEAD // PHASE 2 ARTERIAL
        </div>
        <div class="traffic-signal-box">
            <div class="{r_class}" title="Red Phase"></div>
            <div class="{a_class}" title="Amber Phase"></div>
            <div class="{g_class}" title="Green Phase"></div>
        </div>
        <div style="margin-top: 14px; width: 100%; display: flex; flex-direction: column; gap: 8px;">
            <div class="pedestrian-signal">
                <span style="font-size: 1.1rem;">{ped_icon}</span>
                <span style="color: {ped_color};">{ped_label}</span>
            </div>
            <div style="background: #080C14; border: 1px solid #1F2937; border-radius: 6px; padding: 6px; text-align: center; font-family: var(--font-mono); font-size: 0.75rem;">
                <span style="color: #64748B;">COUNTDOWN:</span>
                <span style="color: #00F0FF; font-weight: 700; margin-left: 6px;">{countdown}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_empty_state_card(
    icon_symbol: str,
    title: str,
    description: str,
    stage_tag: str = "Stage 2 Pending",
) -> None:
    """
    Renders an informative empty state container when real data/model is not yet integrated.
    """
    html = f"""
    <div class="empty-state-container">
        <div class="empty-state-icon">{icon_symbol}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-desc">{description}</div>
        <span class="status-pill lavender">{stage_tag}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_command_centre_plotly_layout(title: str = "", height: int = 300) -> Dict[str, Any]:
    """
    Generates a uniform futuristic Plotly dark layout conforming to the
    CrossGuard neon cyan/charcoal design specification.
    """
    return {
        "title": {
            "text": title,
            "font": {"color": "#F8FAFC", "size": 13, "family": "Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        "height": height,
        "margin": {"l": 40, "r": 20, "t": 40, "b": 35},
        "paper_bgcolor": "rgba(13, 19, 34, 0.4)",
        "plot_bgcolor": "rgba(8, 12, 20, 0.5)",
        "font": {"color": "#94A3B8", "family": "JetBrains Mono, monospace", "size": 11},
        "xaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zerolinecolor": "rgba(0, 240, 255, 0.2)",
            "showline": True,
            "linecolor": "rgba(0, 240, 255, 0.2)",
            "tickfont": {"size": 10},
        },
        "yaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.06)",
            "zerolinecolor": "rgba(0, 240, 255, 0.2)",
            "showline": True,
            "linecolor": "rgba(0, 240, 255, 0.2)",
            "tickfont": {"size": 10},
        },
        "legend": {
            "font": {"color": "#CBD5E1", "size": 10},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    }
