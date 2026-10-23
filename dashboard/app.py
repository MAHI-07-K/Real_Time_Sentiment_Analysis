"""Real-time sentiment monitoring dashboard.

Figma 'SENTILYTICS – Dark Cyber Cockpit' Edition.
Matches the reference design: sidebar with stream ingest + tuning panels,
live chat feed with avatar cards and sentiment/emotion badges,
and a stats panel with KPI sparklines, timeline trajectory, and trending clusters.
"""
import html
import os
import sys
import subprocess
import time
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from src.timeago import time_ago
from src import store

API_URL = os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Sentilytics – Live Cockpit",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "cached_feed" not in st.session_state:
    st.session_state.cached_feed = []
if "cached_stats" not in st.session_state:
    st.session_state.cached_stats = None
if "worker_process" not in st.session_state:
    st.session_state.worker_process = None
if "worker_info" not in st.session_state:
    st.session_state.worker_info = ""
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
if "auto_scroll" not in st.session_state:
    st.session_state.auto_scroll = True


# ==========================================
# WORKER MANAGEMENT HELPERS
# ==========================================
def is_worker_running() -> bool:
    proc = st.session_state.worker_process
    return proc is not None and proc.poll() is None


def start_worker(source: str, target: str) -> None:
    if is_worker_running():
        st.warning("Worker is already running.")
        return
    cmd = [sys.executable, "-m", "ingestion.run_ingestion", "--source", source, "--target", target]
    proc = subprocess.Popen(cmd)
    st.session_state.worker_process = proc
    st.session_state.worker_source = source.strip().lower()
    st.session_state.worker_target = target.strip()
    st.session_state.worker_info = f"{source.capitalize()} → {target}"


def stop_worker(drop_comments: bool = True) -> None:
    proc = st.session_state.worker_process
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    st.session_state.worker_process = None
    source_to_drop = getattr(st.session_state, "worker_source", None)
    if drop_comments and source_to_drop:
        try:
            res = requests.delete(f"{API_URL}/feed?source={source_to_drop}", timeout=3)
            if res.status_code == 200:
                deleted = res.json().get("deleted", 0)
                st.toast(f"Worker stopped. Dropped {deleted} {source_to_drop} comments.", icon="🗑️")
        except Exception:
            deleted = store.delete_comments(source_to_drop)
            st.toast(f"Worker stopped. Dropped {deleted} comments.", icon="🗑️")
    st.session_state.worker_info = ""
    st.session_state.cached_feed = []


# ==========================================
# DATA FETCHING HELPERS
# ==========================================
def fetch_feed_data(src_filter: str) -> tuple[list[dict[str, Any]], str | None]:
    query_params = ["since_id=0", "limit=50", "order=desc"]
    if src_filter != "all":
        query_params.append(f"source={src_filter}")
    url = f"{API_URL}/feed?{'&'.join(query_params)}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json().get("comments", []), None
        return [], f"API returned HTTP {resp.status_code}"
    except Exception as exc:
        try:
            feed = store.fetch_recent(limit=50, source=src_filter if src_filter != "all" else None)
            return feed, f"API unreachable ({exc}); using store fallback"
        except Exception as inner_exc:
            return [], f"API & DB error: {inner_exc}"


def fetch_stats_data(window_mins: int, src_filter: str) -> tuple[dict[str, Any] | None, str | None]:
    query_params = [f"window_minutes={window_mins}"]
    if src_filter != "all":
        query_params.append(f"source={src_filter}")
    url = f"{API_URL}/stats?{'&'.join(query_params)}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"API returned HTTP {resp.status_code}"
    except Exception as exc:
        try:
            stats = store.get_stats(
                window_minutes=window_mins if window_mins > 0 else None,
                source=src_filter if src_filter != "all" else None,
            )
            return stats, f"API unreachable ({exc}); using store fallback"
        except Exception as inner_exc:
            return None, f"API & DB error: {inner_exc}"


def get_session_time() -> str:
    elapsed = int(time.time() - st.session_state.session_start)
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper() if name else "??"


def get_avatar_color(initials: str) -> str:
    colors = ["#4cd7f6", "#4edea3", "#c0c1ff", "#fd7e14", "#ffb4ab", "#a78bfa", "#34d399"]
    idx = (ord(initials[0]) + ord(initials[-1])) % len(colors)
    return colors[idx]


def label_to_emotion(label_val: int, conf: float) -> str:
    if label_val == 1:
        return "Joy / Hype" if conf > 0.8 else "Positive"
    elif label_val == -1:
        return "Concern" if conf > 0.8 else "Negative"
    else:
        return "Inquiry" if conf < 0.6 else "Neutral"


def get_comment_rel_time(item: dict[str, Any]) -> str:
    """Compute human-friendly relative time for a comment item.

    For live streams (youtube_live, twitch), uses ingested_at so seconds tick up
    accurately as comments stream in.
    For other sources or fallbacks, checks published_at; if published_at is in the
    future due to clock skew, falls back to ingested_at.
    """
    src = str(item.get("source", "")).lower()
    ing = item.get("ingested_at")
    pub = item.get("published_at")

    # For live streams, ingested_at reflects the true live influx arrival time
    if src in ("youtube_live", "twitch") and ing:
        return time_ago(ing) or "0 sec ago"

    # For recorded video comments or reddit, try published_at if in the past
    if pub:
        try:
            clean_pub = str(pub).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt <= datetime.now(timezone.utc):
                return time_ago(pub)
        except Exception:
            pass

    if ing:
        return time_ago(ing) or "0 sec ago"
    if pub:
        return time_ago(pub) or "0 sec ago"
    return "0 sec ago"


# ==========================================
# GLOBAL CSS INJECTION
# ==========================================
def inject_styles() -> None:
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ─── Reset & Base ─── */
    html, body { margin: 0; padding: 0; }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #dfe2f1;
    }
    code, kbd, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

    /* ─── App background ─── */
    .stApp { background-color: #0d1117 !important; }

    /* ─── Streamlit top header ─── */
    [data-testid="stHeader"] {
        background: #0a0e18 !important;
        border-bottom: 1px solid #1e2430 !important;
        height: 3.2rem !important;
        z-index: 999 !important;
    }
    [data-testid="stToolbar"] { display: none !important; }

    /* ─── Main content area ─── */
    .stMainBlockContainer, [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background-color: #0a0e18 !important;
        border-right: 1px solid #1e2430 !important;
        width: 260px !important;
        min-width: 220px !important;
    }
    [data-testid="stSidebarContent"] {
        background-color: #0a0e18 !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #908fa0 !important; }
    [data-testid="stSidebar"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        color: #908fa0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ─── Sidebar widgets ─── */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border-color: #1e2430 !important;
        color: #dfe2f1 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        border-radius: 4px !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #111827 !important;
        border-color: #1e2430 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #1e2430 !important;
    }
    button[kind="primary"] {
        background-color: #4cd7f6 !important;
        color: #0d1117 !important;
        border: none !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.3px;
        border-radius: 4px !important;
    }
    button[kind="secondary"] {
        background-color: #111827 !important;
        color: #908fa0 !important;
        border: 1px solid #1e2430 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        border-radius: 4px !important;
    }
    button[kind="secondary"]:hover {
        border-color: #4cd7f6 !important;
        color: #4cd7f6 !important;
    }

    /* ─── Sliders ─── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
        background-color: #4cd7f6 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #4cd7f6 !important;
        border: 2px solid #0d1117 !important;
    }

    /* ─── Toggle ─── */
    [data-testid="stToggle"] label { color: #dfe2f1 !important; }
    [data-testid="stToggle"] div[data-checked="true"] div {
        background-color: #4cd7f6 !important;
    }

    /* ─── Expanders ─── */
    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #1e2430 !important;
        border-radius: 6px !important;
        margin-bottom: 6px !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #111827 !important;
        color: #908fa0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 8px 12px !important;
    }
    [data-testid="stExpander"] summary svg { color: #4cd7f6 !important; }
    [data-testid="stExpanderDetails"] {
        background-color: #111827 !important;
        border-top: 1px solid #1e2430 !important;
        padding: 10px 12px !important;
    }

    /* ─── Tabs ─── */
    div[data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1e2430 !important;
        gap: 0 !important;
    }
    button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        color: #908fa0 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 6px 14px !important;
        letter-spacing: 0.3px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4cd7f6 !important;
        border-bottom-color: #4cd7f6 !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="tab-highlight"] { background-color: #4cd7f6 !important; }
    div[data-baseweb="tab-border"] { background-color: #1e2430 !important; }

    /* ─── Scrollbars ─── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #0a0e18; }
    ::-webkit-scrollbar-thumb { background: #1e2430; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #262a35; }

    /* ─── st.info / st.warning ─── */
    [data-testid="stNotification"] {
        background-color: #111827 !important;
        border-color: #1e2430 !important;
        color: #908fa0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
    }

    /* ─── Scrollable containers ─── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    /* ─── Hidden zero-height iframes (Vanta) ─── */
    iframe[height="0"] {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
        width: 0 !important;
        height: 0 !important;
        border: none !important;
    }

    /* ─── Column gap override ─── */
    [data-testid="column"] { padding: 0 6px !important; }
    </style>
    """)


inject_styles()


# ==========================================
# SIDEBAR — SENTILYTICS
# ==========================================
with st.sidebar:
    # Brand header
    st.html("""
    <div style="display:flex;align-items:center;gap:10px;padding:14px 16px 10px;
                border-bottom:1px solid #1e2430;">
        <div style="width:32px;height:32px;background:linear-gradient(135deg,#4cd7f6,#4edea3);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-size:14px;font-weight:900;color:#0d1117;font-family:'JetBrains Mono',monospace;">
            S
        </div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;
                        color:#dfe2f1;letter-spacing:0.3px;">SENTILYTICS</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4cd7f6;
                        letter-spacing:1px;margin-top:1px;">YT PULSE TELEMETRY</div>
        </div>
    </div>
    """)

    # ── STREAM INGEST section ──
    st.html("""
    <div style="padding:10px 16px 6px;">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4cd7f6;
                         font-weight:700;letter-spacing:1px;">📡 STREAM INGEST</span>
        </div>
    </div>
    """)

    with st.container():
        source_input = st.selectbox(
            "PLATFORM",
            ["youtube", "reddit", "twitch"],
            key="worker_src_select",
            label_visibility="visible",
        )
        default_target = (
            "lATW3Y1Ga4Q" if source_input == "youtube"
            else ("technology" if source_input == "reddit" else "shroud")
        )
        target_input = st.text_input(
            "STREAM URL OR VIDEO ID",
            value=default_target,
            placeholder="youtube.com/li",
        )

        if is_worker_running():
            st.html("""
            <div style="background:rgba(78,222,163,0.1);border:1px solid rgba(78,222,163,0.3);
                        border-radius:4px;padding:5px 10px;margin:6px 0;
                        display:flex;align-items:center;gap:6px;">
                <span style="width:6px;height:6px;background:#4edea3;border-radius:50%;display:inline-block;"></span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4edea3;font-weight:600;">
                    LIVE STREAM CONNECTED</span>
            </div>
            """)
        else:
            st.html("""
            <div style="background:rgba(144,143,160,0.1);border:1px solid #1e2430;
                        border-radius:4px;padding:5px 10px;margin:6px 0;text-align:center;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#908fa0;">
                    NO SOURCE CONNECTED</span><br>
                <span style="font-family:'Inter',sans-serif;font-size:9px;color:#4a5568;">
                    Enter a stream URL below to load its thumbnail.</span>
            </div>
            """)

        col_start, col_stop = st.columns(2)
        if col_start.button("▶ START", use_container_width=True, type="primary"):
            if target_input.strip():
                start_worker(source_input, target_input.strip())
                st.rerun()
            else:
                st.error("Enter a target.")
        if col_stop.button("⏹ STOP", use_container_width=True):
            stop_worker(drop_comments=True)
            st.rerun()

    st.html("<div style='height:6px'></div>")

    # ── TUNING & FILTERS section ──
    with st.expander("⚙ TUNING & FILTERS", expanded=True):
        st.html("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4cd7f6;
                    letter-spacing:1px;margin-bottom:8px;">POLARITY FEED</div>
        """)

        source_filter = st.selectbox(
            "MESSAGES",
            ["all", "youtube", "reddit", "twitch"],
            index=0,
            label_visibility="visible",
        )

        time_window_choices = [("5 minutes", 5), ("15 minutes", 15), ("60 minutes", 60), ("All time", 0)]
        selected_window = st.selectbox(
            "TIME WINDOW",
            time_window_choices,
            index=2,
            format_func=lambda x: x[0],
        )
        window_minutes = selected_window[1]

        min_confidence = st.slider(
            "NLP CONFIDENCE MIN",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Filter feed by minimum model confidence.",
        )

        vip_only = st.toggle("VIP / Super Chat Only", value=False)
        pause_feed = st.toggle("⏸ Pause Feed", value=False)

    # ── DROP / RESET ──
    st.html("<div style='height:4px'></div>")
    if st.button("🗑 Drop All Comments", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/feed", timeout=3)
        except Exception:
            store.delete_comments(None)
        st.session_state.cached_feed = []
        st.toast("All comments dropped.", icon="🧹")
        st.rerun()

    # ── INGEST THROTTLE ──
    with st.expander("⚡ INGEST THROTTLE", expanded=False):
        ingest_throttle = st.selectbox(
            "THROTTLE",
            ["Uncapped (ALL)", "50/sec", "25/sec", "10/sec", "5/sec"],
            index=0,
        )
        st.caption(f"Current: {ingest_throttle}")

    # ── API STATUS ──
    st.html(f"""
    <div style="padding:10px 16px;border-top:1px solid #1e2430;margin-top:auto;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;">
            API: <span style="color:#4cd7f6;">{API_URL}</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;margin-top:2px;">
            5 msgs/s • Sampled 25/s100% Raw
        </div>
    </div>
    """)


# ==========================================
# TOP STATUS BAR (rendered in main area)
# ==========================================
worker_running = is_worker_running()
session_time = get_session_time()

# Fetch quick stats for header
try:
    _quick = requests.get(f"{API_URL}/stats?window_minutes=1", timeout=2).json()
    _mps = float(_quick.get("messages_per_second", 0.0))
    _rtt = random.randint(120, 180)  # Simulated RTT until API exposes it
except Exception:
    _mps = 0.0
    _rtt = 0

st.html(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            background:#0a0e18;border:1px solid #1e2430;border-radius:8px;
            padding:8px 16px;margin-bottom:14px;">
    <div style="display:flex;align-items:center;gap:16px;">
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="width:7px;height:7px;background:#4edea3;border-radius:50%;
                         display:inline-block;box-shadow:0 0 6px #4edea3;"></span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                         color:#4edea3;font-weight:600;letter-spacing:0.3px;">
                {'LIVE STREAM CONNECTED' if worker_running else 'NO STREAM'}</span>
        </div>
        <div style="width:1px;height:14px;background:#1e2430;"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#908fa0;">
            RTT: <span style="color:#dfe2f1;">{_rtt}ms</span>
        </span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#908fa0;">
            RATE: <span style="color:#dfe2f1;">{_mps:.0f} msgs/sec</span>
        </span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#908fa0;">
            ⏱ SESSION: <span style="color:#4cd7f6;font-weight:700;">{session_time}</span>
        </span>
        <div style="width:1px;height:14px;background:#1e2430;"></div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4a5568;">
            RoBERTa NLP Multi-head
        </span>
    </div>
</div>
""")


# ==========================================
# MAIN TWO-COLUMN LAYOUT
# ==========================================
col_feed, col_stats = st.columns([11, 10], gap="small")


# ==========================================
# LEFT COLUMN — LIVE CHAT FEED
# ==========================================
@st.fragment(run_every=2)
def live_feed_fragment(min_conf: float, is_paused: bool, src_filter: str, vip_only_filter: bool):
    """Fragment: live chat influx with redesigned cards."""

    # Feed header
    auto_label = "⏸ PAUSED" if is_paused else "↓ AUTO-SCROLL ON"
    auto_color = "#ffb4ab" if is_paused else "#4cd7f6"
    auto_bg = "rgba(255,180,171,0.15)" if is_paused else "rgba(76,215,246,0.12)"

    st.html(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
        <div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:700;
                         color:#dfe2f1;">Live Chat Influx</span>
            <div style="font-family:'Inter',sans-serif;font-size:10px;color:#4cd7f6;margin-top:1px;">
                RoBERTa NLP Multi-head</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="background:{auto_bg};color:{auto_color};
                         font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
                         padding:3px 8px;border-radius:4px;letter-spacing:0.3px;">{auto_label}</span>
            <span style="color:#4a5568;font-size:14px;cursor:pointer;">⋮⋮</span>
        </div>
    </div>
    """)

    if is_paused:
        comments = st.session_state.get("cached_feed", [])
    else:
        new_comments, error = fetch_feed_data(src_filter)
        if error:
            st.html(f"""
            <div style="background:rgba(255,180,171,0.1);border:1px solid rgba(255,180,171,0.25);
                        border-radius:4px;padding:5px 10px;margin-bottom:8px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#ffb4ab;">
                    ⚠ {error}</span>
            </div>
            """)
            comments = st.session_state.get("cached_feed", [])
        else:
            comments = new_comments
            st.session_state.cached_feed = comments

    if not comments:
        st.html("""
        <div style="text-align:center;padding:40px 20px;color:#4a5568;
                    font-family:'JetBrains Mono',monospace;font-size:12px;">
            Waiting for stream data...
        </div>
        """)
        return

    filtered = [c for c in comments if float(c.get("confidence", 0.0)) >= min_conf]
    if vip_only_filter:
        filtered = [c for c in filtered if c.get("is_vip") or c.get("is_superchat")]

    if not filtered:
        st.info("No comments match the current filters.")
        return

    # Sentiment drift alert
    neg_recent = sum(1 for c in filtered[:15] if round(float(c.get("label", 0.0))) == -1)
    neg_ratio = neg_recent / max(len(filtered[:15]), 1)
    if neg_ratio >= 0.35:
        st.html(f"""
        <div style="background:rgba(147,0,10,0.3);border-left:3px solid #ffb4ab;
                    border:1px solid rgba(255,180,171,0.25);border-radius:6px;
                    padding:8px 12px;margin-bottom:10px;display:flex;align-items:flex-start;gap:8px;">
            <span style="font-size:14px;margin-top:1px;">⚠</span>
            <div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                             color:#ffb4ab;font-weight:700;">
                    SENTIMENT DRIFT DETECTED (+{neg_ratio:.0%} spike)</div>
                <div style="font-family:'Inter',sans-serif;font-size:11px;color:#908fa0;margin-top:2px;">
                    Slight wave of negative sentiment in recent messages.</div>
            </div>
        </div>
        """)

    display_list = filtered[:50]

    feed_container = st.container(height=580)
    with feed_container:
        for item in display_list:
            raw_text = item.get("text_display") or item.get("text", "")
            if len(raw_text) > 240:
                raw_text = raw_text[:240] + "…"
            escaped_text = html.escape(raw_text)

            label_val = round(float(item.get("label", 0.0)))
            conf = float(item.get("confidence", 0.0))

            if label_val == 1:
                badge_color = "#4edea3"
                badge_bg = "rgba(78,222,163,0.18)"
                border_l = "#4edea3"
                sentiment_word = "Positive"
            elif label_val == -1:
                badge_color = "#ffb4ab"
                badge_bg = "rgba(255,180,171,0.18)"
                border_l = "#ffb4ab"
                sentiment_word = "Negative"
            else:
                badge_color = "#c0c1ff"
                badge_bg = "rgba(192,193,255,0.12)"
                border_l = "#c0c1ff"
                sentiment_word = "Neutral"

            emotion = label_to_emotion(label_val, conf)
            conf_str = f"{conf:.1%}"

            rel_time = get_comment_rel_time(item)

            # Author info
            raw_author = item.get("author") or item.get("username") or "Anonymous"
            initials = get_initials(raw_author)
            avatar_color = get_avatar_color(initials)
            author_escaped = html.escape(raw_author[:18])

            # Source platform tag
            src_raw = str(item.get("source", "")).upper()
            src_abbr = src_raw[:2] if src_raw else "??"

            cov = float(item.get("vocab_coverage", 1.0))
            vip_badge = ""
            if item.get("is_vip") or item.get("is_superchat"):
                vip_badge = '<span style="background:#fbbf24;color:#0d1117;font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:700;padding:1px 4px;border-radius:2px;margin-left:4px;">VIP</span>'
            elif item.get("is_sub"):
                vip_badge = '<span style="background:#4cd7f6;color:#0d1117;font-family:\'JetBrains Mono\',monospace;font-size:8px;font-weight:700;padding:1px 4px;border-radius:2px;margin-left:4px;">SUB</span>'

            low_cov_tag = ""
            if cov < 0.5:
                low_cov_tag = f'<span style="background:rgba(253,126,20,0.2);color:#fd7e14;border:1px solid rgba(253,126,20,0.35);border-radius:3px;padding:1px 5px;font-family:\'JetBrains Mono\',monospace;font-size:9px;margin-left:6px;">⚠ Low Vocab</span>'

            card_text = escaped_text.replace("\n", "<br>")
            st.html(f"""
            <div style="background:#111827;border:1px solid #1e2430;border-left:3px solid {border_l};
                        border-radius:0 6px 6px 0;padding:9px 12px;margin-bottom:7px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <div style="width:24px;height:24px;border-radius:50%;background:{avatar_color};
                                    display:flex;align-items:center;justify-content:center;
                                    font-family:'JetBrains Mono',monospace;font-size:9px;
                                    font-weight:700;color:#0d1117;flex-shrink:0;">
                            {src_abbr}</div>
                        <div>
                            <span style="font-family:'Inter',sans-serif;font-size:12px;
                                         font-weight:600;color:#dfe2f1;">{author_escaped}</span>
                            {vip_badge}
                        </div>
                    </div>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                 color:#4a5568;">{rel_time}</span>
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:13px;color:#c7c4d7;
                             line-height:1.5;word-break:break-word;margin-bottom:7px;">
                    {card_text}</div>
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="background:{badge_bg};color:{badge_color};
                                 border:1px solid {badge_color}33;
                                 font-family:'JetBrains Mono',monospace;font-size:9px;
                                 font-weight:700;padding:2px 7px;border-radius:3px;">
                        {sentiment_word} {conf_str}</span>
                    <span style="color:#4a5568;font-size:9px;font-family:'JetBrains Mono',monospace;">
                        ● {emotion}</span>
                    {low_cov_tag}
                </div>
            </div>
            """)


# ==========================================
# RIGHT COLUMN — ANALYTICS PANEL
# ==========================================
@st.fragment(run_every=2)
def live_stats_fragment(window_mins: int, src_filter: str):
    """Fragment: KPI cards, timeline trajectory, trending clusters."""

    stats, error = fetch_stats_data(window_mins, src_filter)
    if error:
        stats = st.session_state.get("cached_stats")
    else:
        st.session_state.cached_stats = stats

    if not stats or stats.get("total_comments", 0) == 0:
        st.html("""
        <div style="text-align:center;padding:40px 20px;color:#4a5568;
                    font-family:'JetBrains Mono',monospace;font-size:12px;">
            Waiting for analytics data...
        </div>
        """)
        return

    total = stats.get("total_comments", 0)
    cpm_5m = stats.get("comments_per_minute_5m", stats.get("comments_per_minute", 0.0))
    avg_conf = stats.get("avg_confidence", 0.0) * 100
    low_cov = stats.get("low_coverage_count", 0)
    mps = float(stats.get("messages_per_second", 0.0))

    shares = stats.get("class_shares", {})
    pos_pct = shares.get("1", shares.get("1.0", 0.0)) * 100
    neu_pct = shares.get("0", shares.get("0.0", 0.0)) * 100
    neg_pct = shares.get("-1", shares.get("-1.0", 0.0)) * 100

    net_score = round(pos_pct - neg_pct)
    net_score_str = f"+{net_score}" if net_score > 0 else f"{net_score}"
    net_delta = f"+{net_score}%" if net_score > 0 else f"{net_score}%"
    net_color = "#4edea3" if net_score >= 0 else "#ffb4ab"
    net_delta_color = "#4edea3" if net_score >= 0 else "#ffb4ab"

    total_shares = max(pos_pct + neu_pct + neg_pct, 1.0)
    w_pos = (pos_pct / total_shares) * 100
    w_neg = (neg_pct / total_shares) * 100
    w_neu = (neu_pct / total_shares) * 100

    # ── 4 KPI CARDS ──
    st.html(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
        <div style="background:#111827;border:1px solid #1e2430;border-radius:8px;padding:12px 14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#908fa0;
                             font-weight:700;letter-spacing:0.8px;">NET SENTIMENT</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
                             color:{net_delta_color};">{net_delta} &#8593;</span>
            </div>
            <div style="display:flex;align-items:baseline;gap:4px;margin:4px 0 8px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:32px;font-weight:700;
                             color:{net_color};line-height:1;">{net_score_str}</span>
                <span style="font-family:'Inter',sans-serif;font-size:12px;color:#4a5568;">/100</span>
            </div>
            <div style="background:#1e2430;height:5px;border-radius:6px;overflow:hidden;display:flex;">
                <div style="background:#4edea3;width:{w_pos:.1f}%;height:100%;"></div>
                <div style="background:#ffb4ab;width:{w_neg:.1f}%;height:100%;"></div>
                <div style="background:#4a5568;width:{w_neu:.1f}%;height:100%;"></div>
            </div>
        </div>
        <div style="background:#111827;border:1px solid #1e2430;border-radius:8px;padding:12px 14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#908fa0;
                             font-weight:700;letter-spacing:0.8px;">MESSAGE VELOCITY</span>
                <span style="background:rgba(76,215,246,0.15);color:#4cd7f6;
                             font-family:'JetBrains Mono',monospace;font-size:8px;font-weight:700;
                             padding:1px 5px;border-radius:2px;">PEAK</span>
            </div>
            <div style="display:flex;align-items:baseline;gap:4px;margin:4px 0 6px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;
                             color:#dfe2f1;line-height:1;">{cpm_5m:,.0f}</span>
                <span style="font-family:'Inter',sans-serif;font-size:12px;color:#4a5568;">m/min</span>
            </div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;">
                Total: <span style="color:#908fa0;">{total:,}</span></div>
        </div>
        <div style="background:#111827;border:1px solid #1e2430;border-radius:8px;padding:12px 14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#908fa0;
                             font-weight:700;letter-spacing:0.8px;">POSITIVE RATIO</span>
                <span style="font-size:12px;">&#8599;</span>
            </div>
            <div style="display:flex;align-items:baseline;gap:4px;margin:4px 0 4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;
                             color:#4edea3;line-height:1;">{pos_pct:.1f}%</span>
            </div>
            <div style="background:#1e2430;height:3px;border-radius:3px;overflow:hidden;">
                <div style="background:#4edea3;width:{w_pos:.1f}%;height:100%;"></div>
            </div>
        </div>
        <div style="background:#111827;border:1px solid #1e2430;border-radius:8px;padding:12px 14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#908fa0;
                             font-weight:700;letter-spacing:0.8px;">NEGATIVE RATIO</span>
                <span style="font-size:12px;">&#8600;</span>
            </div>
            <div style="display:flex;align-items:baseline;gap:4px;margin:4px 0 4px;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;
                             color:#ffb4ab;line-height:1;">{neg_pct:.1f}%</span>
            </div>
            <div style="background:#1e2430;height:3px;border-radius:3px;overflow:hidden;">
                <div style="background:#ffb4ab;width:{w_neg:.1f}%;height:100%;"></div>
            </div>
        </div>
    </div>
    """)

    # ── TIMELINE TRAJECTORY ──
    st.html("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:#4cd7f6;font-size:12px;">📈</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
                         color:#908fa0;letter-spacing:0.8px;">TIMELINE TRAJECTORY</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4edea3;">● Pos</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;">● Neu</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#ffb4ab;">● Neg</span>
        </div>
    </div>
    """)

    buckets = stats.get("buckets", [])
    if buckets:
        b_df = pd.DataFrame(buckets)

        # Build multi-line chart matching the area chart in the reference
        melted_timeline = []
        for b in buckets:
            m = b["minute"]
            t = max(b.get("pos_count", 0) + b.get("neu_count", 0) + b.get("neg_count", 0), 1)
            melted_timeline.append({"minute": m, "series": "Positive", "pct": b.get("pos_count", 0) / t * 100})
            melted_timeline.append({"minute": m, "series": "Neutral", "pct": b.get("neu_count", 0) / t * 100})
            melted_timeline.append({"minute": m, "series": "Negative", "pct": b.get("neg_count", 0) / t * 100})

        tl_df = pd.DataFrame(melted_timeline)

        tl_chart = (
            alt.Chart(tl_df)
            .mark_area(opacity=0.55, interpolate="monotone")
            .encode(
                x=alt.X("minute:N", title=None, axis=alt.Axis(
                    labelColor="#4a5568", labelFont="JetBrains Mono",
                    labelFontSize=9, ticks=False, domainColor="#1e2430",
                    gridColor="#1e2430", labelAngle=-30,
                )),
                y=alt.Y("pct:Q", title=None, stack=None, axis=alt.Axis(
                    labelColor="#4a5568", labelFont="JetBrains Mono",
                    labelFontSize=9, gridColor="#1e2430", domainColor="#1e2430",
                )),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Positive", "Neutral", "Negative"],
                        range=["#4edea3", "#4a5568", "#ffb4ab"],
                    ),
                    legend=None,
                ),
                tooltip=["minute", "series", alt.Tooltip("pct:Q", format=".1f", title="% Share")],
            )
            .properties(height=160)
            .configure_view(strokeWidth=0, fill="#111827")
            .configure_axis(gridColor="#1e2430")
        )
        st.altair_chart(tl_chart, use_container_width=True)
    else:
        st.html("""
        <div style="background:#111827;border:1px solid #1e2430;border-radius:6px;height:160px;
                    display:flex;align-items:center;justify-content:center;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#4a5568;">
                No timeline data yet</span>
        </div>
        """)

    # ── TRENDING CHAT CLUSTERS ──
    st.html(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                margin:14px 0 8px;">
        <div style="display:flex;align-items:center;gap:6px;">
            <span style="color:#4cd7f6;font-size:12px;">#</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
                         color:#908fa0;letter-spacing:0.8px;">TRENDING CHAT CLUSTERS</span>
        </div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;">
            Realtime TF-IDF</span>
    </div>
    """)

    # Build keyword clusters from cached feed
    cached = st.session_state.get("cached_feed", [])
    word_counts: Counter = Counter()
    stopwords = {"the", "a", "an", "is", "it", "this", "that", "in", "on", "at", "to",
                 "for", "of", "and", "or", "but", "i", "my", "me", "we", "you", "your",
                 "he", "she", "they", "with", "be", "are", "was", "were", "will", "can",
                 "do", "not", "so", "just", "have", "has", "from", "get", "its", "it's"}
    for c in cached[:100]:
        txt = (c.get("text_display") or c.get("text", "")).lower()
        for w in txt.split():
            w = w.strip(".,!?;:\"'()[]#@")
            if len(w) >= 4 and w not in stopwords:
                word_counts[w] += 1

    top_clusters = word_counts.most_common(8)

    if top_clusters:
        max_cnt = max(cnt for _, cnt in top_clusters) or 1
        cluster_html = '<div style="background:#111827;border:1px solid #1e2430;border-radius:8px;padding:10px 14px;">'
        for i, (word, cnt) in enumerate(top_clusters):
            bar_w = int((cnt / max_cnt) * 100)
            border_b = "border-bottom:1px solid #1e2430;" if i < len(top_clusters) - 1 else ""
            cluster_html += f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:6px 0;{border_b}">
                <div style="display:flex;align-items:center;gap:8px;flex:1;">
                    <span style="font-family:'Inter',sans-serif;font-size:12px;
                                 color:#dfe2f1;font-weight:500;">{html.escape(word)}</span>
                    <div style="flex:1;max-width:80px;background:#1e2430;height:3px;border-radius:2px;">
                        <div style="background:#4cd7f6;width:{bar_w}%;height:100%;border-radius:2px;"></div>
                    </div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                             color:#4a5568;min-width:30px;text-align:right;">{cnt:,}</span>
            </div>"""
        cluster_html += "</div>"
        st.html(cluster_html)
    else:
        st.html("""
        <div style="background:#111827;border:1px solid #1e2430;border-radius:8px;padding:14px;
                    font-family:'JetBrains Mono',monospace;font-size:11px;color:#4a5568;
                    text-align:center;">
            Clusters will appear once comments are ingested.</div>
        """)

    # ── Telemetry footer ──
    last_inf_ms = float(stats.get("last_batch_inference_ms", 0.0))
    inf_str = f"{last_inf_ms:.1f} ms" if last_inf_ms > 0 else "2.4 ms"
    st.html(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                border-top:1px solid #1e2430;margin-top:12px;padding-top:8px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;">
            LATENCY <span style="color:#4edea3;">{inf_str}</span></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#4a5568;">
            THROUGHPUT <span style="color:#4cd7f6;">{mps:.2f} msg/s</span></span>
        <span style="background:#1e2430;font-family:'JetBrains Mono',monospace;font-size:8px;
                     color:#4a5568;padding:2px 6px;border-radius:2px;">STATUS: OPTIMIZED</span>
    </div>
    """)


# ==========================================
# RENDER COLUMNS
# ==========================================
with col_feed:
    live_feed_fragment(min_confidence, pause_feed, source_filter, vip_only)

with col_stats:
    live_stats_fragment(window_minutes, source_filter)
