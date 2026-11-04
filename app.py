"""
CrossGuard - Bloom City: AI-Powered Adaptive Pedestrian-Priority Traffic Management
==================================================================================
Main application entry point orchestrating sidebar navigation, theme injection,
top surveillance command bar, and routing to the six integrated modules.
"""

from datetime import datetime
import streamlit as st

from core.adaptive_controller import AdaptiveSignalController, ControllerDecision, SignalPhase
from core.config import APP_NAME, NAV_PAGES, PROJECT_SUBTITLE, PROJECT_TITLE, VERSION
from core.hardware_info import get_hardware_telemetry
from core.session_manager import SessionStatus, VideoSession
from core.spatial_analysis import SpatialSnapshot, SpatialZoneEngine
from core.telemetry_manager import TelemetryManager
from dashboard.analytics import render_analytics_page
from dashboard.bloom_city import render_bloom_city_page
from dashboard.components import inject_custom_theme, render_html, render_top_command_bar
from dashboard.gpu_performance import render_gpu_performance_page
from dashboard.overview import render_overview_page
from dashboard.signal_control import render_signal_control_page
from dashboard.vision_monitor import render_vision_monitor_page
from simulation.bloom_city_engine import BloomCityEngine


# 1. Streamlit Application Configuration
st.set_page_config(
    page_title=f"{APP_NAME} // Bloom City AI Command Centre",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Futuristic Command Centre CSS Theme
inject_custom_theme()

# 3. Central Video Session State Initialization
if "video_session" not in st.session_state:
    st.session_state["video_session"] = VideoSession()
    st.session_state["video_session"].telemetry_manager.add_event(
        "SYSTEM", "CrossGuard AI Command Centre initialized with central video session."
    )
    st.session_state["video_session"].telemetry_manager.add_event(
        "INFO", "YOLOv8n and ByteTrack inference pipeline ready."
    )

session: VideoSession = st.session_state["video_session"]

# Synchronize backward-compatible session state references
st.session_state["telemetry_manager"] = session.telemetry_manager
st.session_state["controller"] = session.controller
st.session_state["spatial_engine"] = session.spatial_engine
st.session_state["bloom_engine"] = session.bloom_engine
st.session_state["latest_snapshot"] = session.latest_snapshot
st.session_state["latest_decision"] = session.latest_decision
st.session_state["video_metadata"] = session.video_metadata
st.session_state["video_path"] = session.video_path

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "overview"


# 4. Sidebar Navigation & Global System Controls
with st.sidebar:
    render_html(
        f"""
        <div style="padding: 10px 4px 16px 4px; border-bottom: 1px solid rgba(0, 240, 255, 0.2); margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="pulse-dot"></span>
                <span style="font-family: var(--font-mono); font-size: 0.72rem; letter-spacing: 0.12em; color: #00F0FF; text-transform: uppercase;">
                    SURVEILLANCE NODE 01
                </span>
            </div>
            <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: 0.05em; color: #F8FAFC; margin-top: 4px;">
                CROSSGUARD
            </div>
            <div style="font-size: 0.78rem; font-weight: 600; color: #A78BFA; letter-spacing: 0.08em;">
                BLOOM CITY RESEARCH SIMULATOR
            </div>
            <div style="font-size: 0.72rem; color: #64748B; margin-top: 4px; line-height: 1.3;">
                Adaptive Pedestrian-Priority Traffic Management
            </div>
            <div style="margin-top: 8px;">
                <span class="status-pill cyan">{VERSION}</span>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: #64748B; letter-spacing: 0.1em; margin-bottom: 8px;">
            NAVIGATION MODULES
        </div>
        """
    )

    nav_options = [
        ("overview", "  1. Command Overview"),
        ("vision_monitor", "  2. AI Vision Monitor"),
        ("bloom_city", "  3. Bloom City Twin"),
        ("signal_control", "  4. Adaptive Signal Control"),
        ("analytics", "  5. Traffic Analytics"),
        ("gpu_performance", "  6. GPU Performance Lab"),
    ]

    label_to_id = {label: pid for pid, label in nav_options}
    current_idx = [pid for pid, _ in nav_options].index(st.session_state["active_page"])
    selected_label = st.radio(
        "Navigation",
        options=[label for _, label in nav_options],
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state["active_page"] = label_to_id[selected_label]

    st.markdown("---")

    # Sidebar Hardware & Pipeline Status Summary
    hw_info = get_hardware_telemetry()
    accel_badge = "active" if hw_info["cuda_available"] else "warning"
    accel_label = "CUDA ACCEL" if hw_info["cuda_available"] else "CPU MODE"

    render_html(
        f"""
        <div style="background: rgba(13, 19, 34, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 12px; font-family: var(--font-mono); font-size: 0.75rem;">
            <div style="color: #94A3B8; font-weight: 700; margin-bottom: 6px; letter-spacing: 0.05em;">
                SYSTEM TELEMETRY
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748B;">ACCELERATION:</span>
                <span class="status-pill {accel_badge}">{accel_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748B;">CPU CORES:</span>
                <span style="color: #F8FAFC;">{hw_info['cpu_cores_logical']} LOGICAL</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #64748B;">RAM AVAIL:</span>
                <span style="color: #00F0FF;">{hw_info['ram_available_gb']} GB FREE</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #64748B;">PYTORCH:</span>
                <span style="color: #A78BFA;">{hw_info['torch_version'].split('+')[0]}</span>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div style="margin-top: 14px; text-align: center; font-size: 0.72rem; color: #475569; font-family: var(--font-mono);">
            CrossGuard Research Simulator<br>&copy; 2026 Bloom City Project
        </div>
        """
    )


# 5. Top Surveillance Status Bar
video_meta = session.video_metadata
has_video = video_meta is not None and session.video_path is not None
video_display_name = video_meta.get("file_name") if has_video else None

active_page_title = NAV_PAGES[st.session_state["active_page"]]["title"]
latest_dec: ControllerDecision = session.latest_decision

# Compute truthful AI status and badge
if not has_video:
    ai_status = "AWAITING FOOTAGE"
    ai_badge = "neutral"
    fsm_label = "STANDBY"
elif session.status == SessionStatus.PROCESSING:
    ai_status = "AI PROCESSING"
    ai_badge = "cyan"
    fsm_label = latest_dec.current_phase.value
elif session.status == SessionStatus.COMPLETED:
    ai_status = "AI COMPLETED"
    ai_badge = "active"
    fsm_label = latest_dec.current_phase.value
else:
    ai_status = "AI READY (UPLOADED)"
    ai_badge = "warning"
    fsm_label = latest_dec.current_phase.value

render_top_command_bar(
    active_page_title=active_page_title,
    video_loaded=has_video,
    video_name=video_display_name,
    fsm_state_label=fsm_label,
    ai_status_label=ai_status,
    ai_badge_class=ai_badge,
)

# 6. Page Routing
page_id = st.session_state["active_page"]

try:
    if page_id == "overview":
        render_overview_page()
    elif page_id == "vision_monitor":
        render_vision_monitor_page()
    elif page_id == "bloom_city":
        render_bloom_city_page()
    elif page_id == "signal_control":
        render_signal_control_page()
    elif page_id == "analytics":
        render_analytics_page()
    elif page_id == "gpu_performance":
        render_gpu_performance_page()
    else:
        st.error(f"Unknown page route: '{page_id}'")
except Exception as ex:
    st.error(f"Command Centre Alert: An unexpected exception occurred while rendering '{active_page_title}': {type(ex).__name__} - {str(ex)}")
    with st.expander("Technical Diagnostic Details & Traceback", expanded=False):
        import traceback
        st.code(traceback.format_exc(), language="python")
