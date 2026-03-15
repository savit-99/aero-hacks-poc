import streamlit as st
import pandas as pd
import json
import altair as alt
import numpy as np
import subprocess
import time
import os
import shutil
import hashlib
import uuid
from datetime import datetime

from db import atlas_enabled, get_db, load_latest_tactical_output

MAP_MIN = 0.0
MAP_MAX = 500.0
DEFAULT_DETECTION_RADIUS = 20.0
TELEMETRY_META_PATH = "telemetry_metadata.json"
ANALYSIS_REGISTRY_PATH = "analysis_registry.json"
ANALYSIS_RUNS_DIR = "analysis_runs"
ANALYSIS_CARDS_PER_PAGE = 9

# --- Shared visual palette for consistent chart styling ---
COLOR_ROUTE_GLOW = "#ff6b00"
COLOR_ROUTE_LINE = "#ffd60a"
COLOR_POI_HALO = "#ffffff"
COLOR_POI_STROKE = "#0b132b"
COLOR_START_MARKER = "#06d6a0"
COLOR_FINISH_MARKER = "#ef476f"
COLOR_MINE_DETECTED = "#e63946"
COLOR_MINE_NOT_MINE = "#2a9d8f"
COLOR_MINE_UNKNOWN = "#adb5bd"

SIZE_POI_HALO = 620
SIZE_POI_MARKER = 340
SIZE_MINE_UNKNOWN = 45
SIZE_MINE_NOT_MINE = 85
SIZE_MINE_DETECTED = 105

# --- Page Configuration ---
st.set_page_config(
    page_title="Recon Mission Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(ttl=10, show_spinner=False)
def _load_tactical_data_cached(mission_id, atlas_on):
    if atlas_on:
        try:
            atlas_data, resolved_mission_id = load_latest_tactical_output(mission_id)
            if atlas_data:
                return atlas_data, "atlas", resolved_mission_id
        except Exception as exc:
            return None, f"atlas_error::{exc}", None

    local_archive_path = ""
    if mission_id:
        local_archive_path = os.path.join(ANALYSIS_RUNS_DIR, str(mission_id), "tactical_map_data.json")
        if os.path.exists(local_archive_path):
            try:
                with open(local_archive_path, "r") as f:
                    return json.load(f), "local_archive", None
            except Exception:
                pass

    try:
        with open("tactical_map_data.json", "r") as f:
            return json.load(f), "local_json", None
    except FileNotFoundError:
        return None, "missing", None


def load_tactical_data(mission_id=None):
    data, source, resolved_mission_id = _load_tactical_data_cached(
        str(mission_id or "").strip() or None,
        atlas_enabled(),
    )
    if source.startswith("atlas_error::"):
        exc = source.replace("atlas_error::", "", 1)
        if not st.session_state.get("_atlas_load_warned", False):
            st.warning(f"Atlas tactical load failed ({exc}). Falling back to local JSON.")
            st.session_state._atlas_load_warned = True
        # Retry directly without atlas cache branch.
        data, source, resolved_mission_id = _load_tactical_data_cached(
            str(mission_id or "").strip() or None,
            False,
        )

    if data and source == "atlas" and resolved_mission_id:
        st.session_state.current_mission_id = resolved_mission_id
    if data:
        st.session_state.data_source = source
        return data

    if source == "missing":
        st.error("No tactical data found. Run phase2_engine.py first.")
    return None


def clear_dashboard_caches():
    _load_tactical_data_cached.clear()
    _load_route_snapshots_for_missions_cached.clear()
    _load_mine_snapshots_for_missions_cached.clear()
    _load_analysis_records_cached.clear()
    _prepare_dashboard_frames_cached.clear()


def apply_dashboard_theme():
    st.markdown(
        """
        <style>
        :root {
            --m3-primary: #d0bcff;
            --m3-on-primary: #381e72;
            --m3-primary-container: #4f378b;
            --m3-on-primary-container: #eaddff;
            --m3-secondary-container: #4a4458;
            --m3-on-secondary-container: #e8def8;
            --m3-tertiary-container: #633b48;
            --m3-on-tertiary-container: #ffd8e4;
            --m3-surface: #141218;
            --m3-surface-container: #1d1b20;
            --m3-surface-container-high: #2b2930;
            --m3-surface-variant: #49454f;
            --m3-outline: #938f99;
            --m3-outline-variant: #49454f;
            --m3-on-surface: #e6e1e5;
            --m3-on-surface-variant: #cac4d0;
        }
        .stApp {
            background:
                radial-gradient(1100px 680px at 8% -8%, rgba(208, 188, 255, 0.16), transparent 62%),
                radial-gradient(900px 520px at 94% 1%, rgba(255, 216, 228, 0.11), transparent 58%),
                linear-gradient(180deg, #111015 0%, #131218 46%, #16151c 100%);
            color: var(--m3-on-surface);
            font-family: Roboto, "Segoe UI", Arial, sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: transparent;
        }
        .main .block-container {
            padding-top: 1.05rem;
            padding-bottom: 1.3rem;
            max-width: 1360px;
        }
        .main h1, .main h2, .main h3 {
            letter-spacing: 0;
            font-weight: 600;
            color: var(--m3-on-surface);
        }
        .stMetric {
            border: 1px solid color-mix(in srgb, var(--m3-outline) 55%, transparent);
            border-radius: 16px;
            padding: 10px 14px;
            background: color-mix(in srgb, var(--m3-surface-container-high) 88%, transparent);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.22);
        }
        .command-hero {
            border: 1px solid color-mix(in srgb, var(--m3-outline) 48%, transparent);
            border-radius: 22px;
            padding: 14px 16px;
            margin-bottom: 0.72rem;
            background:
                linear-gradient(
                    130deg,
                    color-mix(in srgb, var(--m3-primary-container) 78%, transparent),
                    color-mix(in srgb, var(--m3-surface-container) 84%, transparent)
                );
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.24);
        }
        .command-eyebrow {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--m3-on-primary-container);
            font-weight: 650;
            margin-bottom: 0.16rem;
        }
        .command-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--m3-on-surface);
            margin: 0.08rem 0 0.15rem 0;
        }
        .command-subtitle {
            font-size: 0.9rem;
            color: var(--m3-on-surface-variant);
            margin: 0;
        }
        .library-hero {
            border: 1px solid color-mix(in srgb, var(--m3-outline) 48%, transparent);
            border-radius: 24px;
            padding: 16px 18px;
            margin-bottom: 0.9rem;
            background:
                linear-gradient(
                    140deg,
                    color-mix(in srgb, var(--m3-primary-container) 76%, transparent),
                    color-mix(in srgb, var(--m3-surface-container) 84%, transparent)
                ),
                radial-gradient(circle at 87% 16%, rgba(208, 188, 255, 0.24), transparent 55%);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
        }
        .library-hero-kicker {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--m3-on-primary-container);
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .library-hero-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: var(--m3-on-surface);
            margin: 0 0 0.2rem 0;
        }
        .library-hero-subtitle {
            font-size: 0.9rem;
            color: var(--m3-on-surface-variant);
            margin: 0;
        }
        .library-card-title {
            font-size: 1.01rem;
            font-weight: 700;
            color: var(--m3-on-surface);
            margin: 0 0 0.16rem 0;
        }
        .library-card-meta {
            font-size: 0.79rem;
            color: var(--m3-on-surface-variant);
            margin: 0 0 0.4rem 0;
        }
        .library-chip {
            display: inline-block;
            padding: 0.16rem 0.5rem;
            border-radius: 999px;
            margin-right: 0.36rem;
            margin-bottom: 0.24rem;
            border: 1px solid color-mix(in srgb, var(--m3-outline) 55%, transparent);
            background: color-mix(in srgb, var(--m3-secondary-container) 72%, transparent);
            color: var(--m3-on-secondary-container);
            font-size: 0.72rem;
            font-weight: 650;
            letter-spacing: 0.02em;
        }
        .status-chip {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border: 1px solid rgba(148, 163, 184, 0.45);
            color: #cbd5e1;
            background: rgba(30, 41, 59, 0.68);
            margin-bottom: 6px;
        }
        .sidebar-metric-card {
            border-radius: 16px;
            padding: 0.5rem 0.6rem;
            border: 1px solid color-mix(in srgb, var(--m3-outline) 48%, transparent);
            margin: 0.2rem 0;
            background: color-mix(in srgb, var(--m3-surface-container-high) 90%, transparent);
        }
        .sidebar-metric-label {
            font-size: 0.74rem;
            letter-spacing: 0.03em;
            font-weight: 700;
            margin-bottom: 0.12rem;
            text-transform: uppercase;
            opacity: 0.95;
        }
        .sidebar-metric-value {
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0;
        }
        .sidebar-metric-card.metric-amber {
            border-left: 3px solid #fb923c;
            color: #fed7aa;
        }
        .sidebar-metric-card.metric-blue {
            border-left: 3px solid #60a5fa;
            color: #dbeafe;
        }
        .sidebar-metric-card.metric-violet {
            border-left: 3px solid #a78bfa;
            color: #ede9fe;
        }
        .sidebar-metric-card.metric-green {
            border-left: 3px solid #4ade80;
            color: #dcfce7;
        }
        .sidebar-metric-card.metric-teal {
            border-left: 3px solid #2dd4bf;
            color: #ccfbf1;
        }
        .sidebar-metric-card.metric-slate {
            border-left: 3px solid #94a3b8;
            color: #e2e8f0;
        }
        .sidebar-metric-card.metric-rose {
            border-left: 3px solid #fb7185;
            color: #ffe4e6;
        }
        .stButton > button {
            border-radius: 20px;
            font-weight: 600;
            border: 1px solid color-mix(in srgb, var(--m3-outline) 58%, transparent);
            transition: all 0.2s ease;
            background: color-mix(in srgb, var(--m3-surface-container-high) 86%, transparent);
            color: var(--m3-on-surface);
        }
        .stButton > button:hover {
            border-color: color-mix(in srgb, var(--m3-primary) 58%, transparent);
            background: color-mix(in srgb, var(--m3-primary-container) 58%, transparent);
        }
        .stButton > button[kind="primary"] {
            background: color-mix(in srgb, var(--m3-primary) 90%, transparent);
            border-color: transparent;
            color: var(--m3-on-primary);
        }
        .stButton > button[kind="primary"]:hover {
            background: color-mix(in srgb, var(--m3-primary) 78%, transparent);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid color-mix(in srgb, var(--m3-outline) 42%, transparent);
            background: color-mix(in srgb, var(--m3-surface-container) 94%, transparent);
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            border-radius: 14px;
            background-color: color-mix(in srgb, var(--m3-surface-container-high) 84%, transparent);
            border-color: color-mix(in srgb, var(--m3-outline) 50%, transparent);
        }
        [data-testid="stSlider"] [role="slider"] {
            box-shadow: 0 0 0 2px color-mix(in srgb, var(--m3-primary) 46%, transparent);
            background: var(--m3-primary);
        }
        details {
            border: 1px solid color-mix(in srgb, var(--m3-outline) 42%, transparent);
            border-radius: 18px;
            background: color-mix(in srgb, var(--m3-surface-container-high) 88%, transparent);
            padding: 0.1rem 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_metric_card(label, value, tone="slate"):
    st.markdown(
        f"""
        <div class="sidebar-metric-card metric-{tone}">
          <div class="sidebar-metric-label">{label}</div>
          <p class="sidebar-metric-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _normalize_route_points(route):
    if not isinstance(route, list):
        return []
    normalized = []
    for p in route:
        if isinstance(p, dict):
            normalized.append(
                {
                    "x": float(p.get("x", 0.0)),
                    "y": float(p.get("y", 0.0)),
                    "z": float(p.get("z", 0.0)),
                }
            )
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            normalized.append(
                {
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "z": float(p[2]) if len(p) >= 3 else 0.0,
                }
            )
    return normalized


def _normalize_mine_points(points):
    if not isinstance(points, list):
        return []
    normalized = []
    for p in points:
        if isinstance(p, dict):
            normalized.append(
                {
                    "x": float(p.get("x", 0.0)),
                    "y": float(p.get("y", 0.0)),
                    "z": float(p.get("z", 0.0)),
                }
            )
        elif isinstance(p, (list, tuple)):
            if len(p) >= 4:
                # Typical anomaly format: [discovery_index, x, y, z]
                normalized.append(
                    {
                        "x": float(p[1]),
                        "y": float(p[2]),
                        "z": float(p[3]),
                    }
                )
            elif len(p) >= 2:
                normalized.append(
                    {
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "z": float(p[2]) if len(p) >= 3 else 0.0,
                    }
                )
    return normalized


def status_chip_text(status):
    text = str(status or "completed").strip()
    if not text:
        text = "completed"
    return text.replace("_", " ").title()


@st.cache_data(ttl=30, show_spinner=False)
def _load_route_snapshots_for_missions_cached(mission_ids, atlas_on):
    ids = [str(m).strip() for m in mission_ids if str(m).strip()]
    route_map = {m: [] for m in ids}
    if not ids:
        return route_map

    if atlas_on:
        try:
            db = get_db()
            cursor = db.routes.find(
                {"mission_id": {"$in": ids}},
                {"_id": 0, "mission_id": 1, "safe_route": 1, "created_at": 1},
            ).sort([("mission_id", 1), ("created_at", -1)])
            seen = set()
            for doc in cursor:
                mission_id = str(doc.get("mission_id", "")).strip()
                if mission_id in seen or mission_id not in route_map:
                    continue
                route_map[mission_id] = _normalize_route_points(doc.get("safe_route", []))
                seen.add(mission_id)
        except Exception:
            pass

        missing = [m for m in ids if not route_map.get(m)]
        if missing:
            try:
                db = get_db()
                cursor = db.tactical_outputs.find(
                    {"mission_id": {"$in": missing}},
                    {"_id": 0, "mission_id": 1, "payload.safe_route": 1, "created_at": 1},
                ).sort([("mission_id", 1), ("created_at", -1)])
                seen = set()
                for doc in cursor:
                    mission_id = str(doc.get("mission_id", "")).strip()
                    if mission_id in seen or mission_id not in route_map:
                        continue
                    payload = doc.get("payload", {}) if isinstance(doc.get("payload"), dict) else {}
                    route_map[mission_id] = _normalize_route_points(payload.get("safe_route", []))
                    seen.add(mission_id)
            except Exception:
                pass
    missing_after_db = [m for m in ids if not route_map.get(m)]
    for mission_id in missing_after_db:
        local_path = os.path.join(ANALYSIS_RUNS_DIR, mission_id, "tactical_map_data.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, "r") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    route_map[mission_id] = _normalize_route_points(payload.get("safe_route", []))
            except Exception:
                pass

        if route_map.get(mission_id):
            continue

        try:
            with open("tactical_map_data.json", "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                run_context = payload.get("run_context", {})
                context_mission_id = str(run_context.get("mission_id", "")).strip() if isinstance(run_context, dict) else ""
                if context_mission_id and context_mission_id == mission_id:
                    route_map[mission_id] = _normalize_route_points(payload.get("safe_route", []))
        except Exception:
            pass

    return route_map


def load_route_snapshots_for_missions(mission_ids):
    return _load_route_snapshots_for_missions_cached(tuple(mission_ids), atlas_enabled())


@st.cache_data(ttl=30, show_spinner=False)
def _load_mine_snapshots_for_missions_cached(mission_ids, atlas_on):
    ids = [str(m).strip() for m in mission_ids if str(m).strip()]
    mine_map = {m: [] for m in ids}
    if not ids:
        return mine_map

    if atlas_on:
        try:
            db = get_db()
            cursor = db.tactical_outputs.find(
                {"mission_id": {"$in": ids}},
                {"_id": 0, "mission_id": 1, "payload.anomalies": 1, "created_at": 1},
            ).sort([("mission_id", 1), ("created_at", -1)])
            seen = set()
            for doc in cursor:
                mission_id = str(doc.get("mission_id", "")).strip()
                if mission_id in seen or mission_id not in mine_map:
                    continue
                payload = doc.get("payload", {}) if isinstance(doc.get("payload"), dict) else {}
                mine_map[mission_id] = _normalize_mine_points(payload.get("anomalies", []))
                seen.add(mission_id)
        except Exception:
            pass

    missing_after_db = [m for m in ids if not mine_map.get(m)]
    for mission_id in missing_after_db:
        local_path = os.path.join(ANALYSIS_RUNS_DIR, mission_id, "tactical_map_data.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, "r") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    mine_map[mission_id] = _normalize_mine_points(payload.get("anomalies", []))
            except Exception:
                pass

        if mine_map.get(mission_id):
            continue

        try:
            with open("tactical_map_data.json", "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                run_context = payload.get("run_context", {})
                context_mission_id = str(run_context.get("mission_id", "")).strip() if isinstance(run_context, dict) else ""
                if context_mission_id and context_mission_id == mission_id:
                    mine_map[mission_id] = _normalize_mine_points(payload.get("anomalies", []))
        except Exception:
            pass

    return mine_map


def load_mine_snapshots_for_missions(mission_ids):
    return _load_mine_snapshots_for_missions_cached(tuple(mission_ids), atlas_enabled())


def _load_local_analysis_registry():
    if not os.path.exists(ANALYSIS_REGISTRY_PATH):
        return []
    try:
        with open(ANALYSIS_REGISTRY_PATH, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
    except Exception:
        return []
    return []


def _write_local_analysis_registry(records):
    try:
        with open(ANALYSIS_REGISTRY_PATH, "w") as f:
            json.dump(records, f, indent=2)
    except Exception:
        pass


def persist_analysis_snapshot(mission_id, start_x, start_y, target_x, target_y, detection_radius):
    if not mission_id:
        return
    mission_id = str(mission_id)
    run_dir = os.path.join(ANALYSIS_RUNS_DIR, mission_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        with open("tactical_map_data.json", "r") as src:
            tactical = json.load(src)
        with open(os.path.join(run_dir, "tactical_map_data.json"), "w") as dst:
            json.dump(tactical, dst)
    except Exception:
        pass

    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    record = {
        "mission_id": mission_id,
        "created_at": created_at,
        "start_x": float(start_x),
        "start_y": float(start_y),
        "target_x": float(target_x),
        "target_y": float(target_y),
        "detection_radius": float(detection_radius),
    }
    records = _load_local_analysis_registry()
    existing_idx = next((i for i, r in enumerate(records) if str(r.get("mission_id", "")) == mission_id), None)
    if existing_idx is not None:
        records[existing_idx] = {**records[existing_idx], **record}
    else:
        records.append(record)
    records.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
    _write_local_analysis_registry(records)
    clear_dashboard_caches()


def delete_analysis_mission(mission_id):
    mission_id = str(mission_id or "").strip()
    if not mission_id:
        return False, "Missing mission id."

    errors = []
    removed_any = False

    if atlas_enabled():
        try:
            db = get_db()
            db.telemetry_raw.delete_many({"mission_id": mission_id})
            db.hazards.delete_many({"mission_id": mission_id})
            db.routes.delete_many({"mission_id": mission_id})
            db.terrain_cells.delete_many({"mission_id": mission_id})
            db.vision_events.delete_many({"mission_id": mission_id})
            db.tactical_outputs.delete_many({"mission_id": mission_id})
            db.alerts.delete_many({"mission_id": mission_id})
            result = db.missions.delete_one({"_id": mission_id})
            removed_any = removed_any or (result.deleted_count > 0)
        except Exception as exc:
            errors.append(f"atlas: {exc}")

    try:
        records = _load_local_analysis_registry()
        remaining = [r for r in records if str(r.get("mission_id", "")).strip() != mission_id]
        if len(remaining) != len(records):
            _write_local_analysis_registry(remaining)
            removed_any = True
    except Exception as exc:
        errors.append(f"registry: {exc}")

    run_dir = os.path.join(ANALYSIS_RUNS_DIR, mission_id)
    try:
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
            removed_any = True
    except Exception as exc:
        errors.append(f"archive: {exc}")

    if st.session_state.get("current_mission_id") == mission_id:
        st.session_state.current_mission_id = ""
        st.session_state.route_requested = False
        st.session_state.has_run = False
        st.session_state.animate_playback = False

    clear_dashboard_caches()
    if errors:
        return removed_any, "Delete completed with warnings: " + "; ".join(errors)
    if not removed_any:
        return False, "Mission not found in Atlas/local records."
    return True, "Mission deleted."


def load_analysis_records(limit=100):
    return _load_analysis_records_cached(int(limit), atlas_enabled())


@st.cache_data(ttl=15, show_spinner=False)
def _load_analysis_records_cached(limit, atlas_on):
    if atlas_on:
        try:
            docs = list(
                get_db().missions.find(
                    {},
                    {
                        "_id": 1,
                        "created_at": 1,
                        "target": 1,
                        "detection_radius": 1,
                        "status": 1,
                        "summary": 1,
                    },
                ).sort("created_at", -1).limit(int(limit))
            )
            rows = []
            for d in docs:
                start = d.get("start", {}) if isinstance(d.get("start"), dict) else {}
                target = d.get("target", {}) if isinstance(d.get("target"), dict) else {}
                summary = d.get("summary", {}) if isinstance(d.get("summary"), dict) else {}
                created_at = d.get("created_at")
                rows.append(
                    {
                        "mission_id": str(d.get("_id", "")),
                        "created_at": created_at.isoformat(timespec="seconds") + "Z" if created_at else "",
                        "status": str(d.get("status", "")),
                        "start_x": float(start.get("x", 0.0)),
                        "start_y": float(start.get("y", 0.0)),
                        "target_x": float(target.get("x", 0.0)),
                        "target_y": float(target.get("y", 0.0)),
                        "detection_radius": float(d.get("detection_radius", 0.0)),
                        "hazards": int(summary.get("hazards", 0)),
                    }
                )
            return rows
        except Exception:
            return []

    return _load_local_analysis_registry()[: int(limit)]


def load_route_snapshot_for_mission(mission_id):
    mission_id = str(mission_id or "").strip()
    if not mission_id:
        return []
    return load_route_snapshots_for_missions([mission_id]).get(mission_id, [])


@st.cache_data(ttl=30, show_spinner=False)
def _prepare_dashboard_frames_cached(tactical_data, detection_radius):
    traj_data = tactical_data.get("trajectory", [])
    if traj_data and isinstance(traj_data[0], dict):
        df_traj = pd.DataFrame(traj_data)
    else:
        df_traj = pd.DataFrame(traj_data, columns=["x", "y", "z"])
    if "drone_id" not in df_traj.columns:
        df_traj["drone_id"] = "drone_1"
    df_traj = df_traj.reset_index(drop=True)
    df_traj["drone_step"] = df_traj.groupby("drone_id").cumcount()

    df_mines = pd.DataFrame(tactical_data["anomalies"], columns=["discovery_index", "x", "y", "z"])
    df_route = pd.DataFrame(tactical_data["safe_route"], columns=["x", "y", "z"]).reset_index()
    df_terrain = pd.DataFrame(tactical_data["terrain_grid"])
    df_offroad = pd.DataFrame(tactical_data.get("off_road_viability", []))
    df_gt = pd.DataFrame(tactical_data.get("ground_truth", []))
    df_vision = pd.DataFrame(tactical_data.get("llm_vision_events", []))

    df_traj = clamp_df_xy(df_traj)
    df_mines = clamp_df_xy(df_mines)
    df_route = clamp_df_xy(df_route)
    df_terrain = clamp_df_xy(df_terrain)
    df_offroad = clamp_df_xy(df_offroad)
    df_gt = clamp_df_xy(df_gt)
    df_vision = clamp_df_xy(df_vision)

    if {"x", "y", "z"}.issubset(df_terrain.columns):
        df_terrain["x2"] = df_terrain["x"] + 20
        df_terrain["y2"] = df_terrain["y"] + 20

    if not df_gt.empty and not df_traj.empty and {"x", "y"}.issubset(df_gt.columns) and {"x", "y", "drone_step"}.issubset(df_traj.columns):
        traj_xy = df_traj[["x", "y"]].to_numpy(dtype=np.float64)
        traj_steps = df_traj["drone_step"].to_numpy(dtype=np.int32)
        gt_xy = df_gt[["x", "y"]].to_numpy(dtype=np.float64)
        radius_sq = float(detection_radius) ** 2

        dx = traj_xy[:, None, 0] - gt_xy[None, :, 0]
        dy = traj_xy[:, None, 1] - gt_xy[None, :, 1]
        within = (dx * dx + dy * dy) <= radius_sq
        any_hit = within.any(axis=0)

        sentinel = np.iinfo(np.int32).max
        step_matrix = np.where(within, traj_steps[:, None], sentinel)
        reveal_steps = step_matrix.min(axis=0)
        reveal_steps = np.where(any_hit, reveal_steps, 10**9).astype(np.int64)
        df_gt = df_gt.copy()
        df_gt["reveal_step"] = reveal_steps
    else:
        df_gt = df_gt.copy()
        df_gt["reveal_step"] = 10**9

    df_gt = df_gt.reset_index().rename(columns={"index": "mine_id"})
    df_gt["mine_id"] = df_gt["mine_id"].astype(int)
    df_gt["decision_label"] = "unverified"
    df_gt["decision_is_mine"] = True
    df_gt["status"] = "unverified"
    df_gt["source"] = "none"
    df_gt["model"] = ""
    df_gt["confidence"] = 0.0
    df_gt["error"] = ""

    if not df_vision.empty and "mine_id" in df_vision.columns:
        df_vision_mapped = df_vision[df_vision["mine_id"] >= 0].copy()
        if not df_vision_mapped.empty:
            if "discovery_index" in df_vision_mapped.columns:
                df_vision_mapped = df_vision_mapped.sort_values("discovery_index")
            else:
                df_vision_mapped = df_vision_mapped.reset_index(drop=True)
                df_vision_mapped["discovery_index"] = df_vision_mapped.index
            df_vision_mapped = df_vision_mapped.drop_duplicates(subset=["mine_id"], keep="first")
            merge_cols = ["mine_id", "decision_label", "decision_is_mine", "status", "source", "model", "confidence", "error"]
            df_gt = df_gt.drop(columns=merge_cols[1:], errors="ignore").merge(
                df_vision_mapped[merge_cols],
                on="mine_id",
                how="left",
            )
            df_gt["decision_label"] = df_gt["decision_label"].fillna("unverified")
            df_gt["decision_is_mine"] = df_gt["decision_is_mine"].fillna(True)
            df_gt["status"] = df_gt["status"].fillna("unverified")
            df_gt["source"] = df_gt["source"].fillna("none")
            df_gt["model"] = df_gt["model"].fillna("")
            df_gt["confidence"] = df_gt["confidence"].fillna(0.0)
            df_gt["error"] = df_gt["error"].fillna("")

    return {
        "df_traj": df_traj,
        "df_mines": df_mines,
        "df_route": df_route,
        "df_terrain": df_terrain,
        "df_offroad": df_offroad,
        "df_gt": df_gt,
        "df_vision": df_vision,
    }


def build_route_snapshot_chart(route_points, mine_points=None):
    if not isinstance(route_points, list) or len(route_points) < 2:
        return None

    rows = []
    for idx, p in enumerate(route_points):
        if isinstance(p, dict):
            x, y = p.get("x", 0.0), p.get("y", 0.0)
            z = p.get("z", 0.0)
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = p[0], p[1]
            z = p[2] if len(p) >= 3 else 0.0
        else:
            continue
        rows.append({"x": float(x), "y": float(y), "z": float(z), "index": int(idx)})

    if len(rows) < 2:
        return None

    df_route = clamp_df_xy(pd.DataFrame(rows))
    start_row = df_route.iloc[0]
    end_row = df_route.iloc[-1]
    poi = build_poi_style_df(start_row["x"], start_row["y"], end_row["x"], end_row["y"], end_label="Finish")

    base = alt.Chart(pd.DataFrame({"x": [MAP_MIN, MAP_MAX], "y": [MAP_MIN, MAP_MAX]})).mark_rect(opacity=0).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
    )
    route_glow = alt.Chart(df_route).mark_line(strokeWidth=12, opacity=0.3).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        order="index:Q",
        color=alt.value(COLOR_ROUTE_GLOW),
    )
    route = alt.Chart(df_route).mark_line(strokeWidth=4, opacity=1.0).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        order="index:Q",
        color=alt.value(COLOR_ROUTE_LINE),
        tooltip=[
            alt.Tooltip("x:Q", title="X", format=".1f"),
            alt.Tooltip("y:Q", title="Y", format=".1f"),
            alt.Tooltip("z:Q", title="Altitude", format=".1f"),
        ],
    )
    poi_halo = alt.Chart(poi).mark_point(filled=True, size=SIZE_POI_HALO, color=COLOR_POI_HALO, opacity=0.88).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        shape=alt.Shape("marker_shape:N", scale=None, legend=None),
        order=alt.Order("sort_order:Q", sort="ascending"),
    )
    poi_markers = alt.Chart(poi).mark_point(filled=True, size=SIZE_POI_MARKER, opacity=1.0, stroke=COLOR_POI_STROKE, strokeWidth=1.5).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
        shape=alt.Shape("marker_shape:N", scale=None, legend=None),
        color=alt.Color("marker_color:N", scale=None, legend=None),
        tooltip="label:N",
        order=alt.Order("sort_order:Q", sort="ascending"),
    )
    layers = [base]
    if isinstance(mine_points, list) and mine_points:
        df_mines = clamp_df_xy(pd.DataFrame(mine_points))
        if {"x", "y"}.issubset(df_mines.columns):
            mines_layer = alt.Chart(df_mines).mark_circle(
                size=55, color=COLOR_MINE_DETECTED, opacity=0.55, stroke=COLOR_POI_HALO, strokeWidth=0.8
            ).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), axis=None),
                tooltip=[
                    alt.Tooltip("x:Q", title="Mine X", format=".1f"),
                    alt.Tooltip("y:Q", title="Mine Y", format=".1f"),
                    alt.Tooltip("z:Q", title="Mine Altitude", format=".1f"),
                ],
            )
            layers.append(mines_layer)

    layers.extend([route_glow, route, poi_halo, poi_markers])
    return alt.layer(*layers).properties(height=180).configure_view(strokeOpacity=0)


def render_analysis_overview():
    st.markdown(
        """
        <div class="library-hero">
            <div class="library-hero-kicker">Mission Control</div>
            <div class="library-hero-title">Mission Analysis Library</div>
            <p class="library-hero-subtitle">
                Launch a new mission study or reopen prior analyses with route snapshots and hazard context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_new, col_refresh = st.columns([1, 1])
    if col_new.button("New Mission Analysis", width="stretch", type="primary"):
        st.session_state.current_mission_id = ""
        st.session_state.route_requested = False
        st.session_state.has_run = True
        st.session_state.animate_playback = False
        st.session_state.dashboard_view = "analysis"
        st.rerun()
    if col_refresh.button("Refresh List", width="stretch"):
        clear_dashboard_caches()
        st.rerun()

    analyses = load_analysis_records(limit=200)
    if not analyses:
        st.info("No saved analyses yet. Start a new mission analysis to populate this library.")
        return

    total_analyses = len(analyses)
    status_counts = {}
    for item in analyses:
        status = status_chip_text(item.get("status", "completed")).lower()
        status_counts[status] = status_counts.get(status, 0) + 1

    newest_created_at = str(analyses[0].get("created_at", "")) if analyses else "n/a"
    m1, m2, m3 = st.columns(3)
    m1.metric("Analyses Stored", total_analyses)
    m2.metric("Completed", status_counts.get("completed", 0))
    m3.metric("Latest Snapshot", newest_created_at if newest_created_at else "n/a")

    st.markdown("### Saved Mission Analyses")
    visible_analyses = analyses
    visible_ids = [str(a.get("mission_id", "")).strip() for a in visible_analyses if str(a.get("mission_id", "")).strip()]
    route_map = load_route_snapshots_for_missions(visible_ids)
    mine_map = load_mine_snapshots_for_missions(visible_ids)

    per_row = 3
    for i in range(0, len(visible_analyses), per_row):
        row = visible_analyses[i : i + per_row]
        cols = st.columns(per_row)
        for j, analysis in enumerate(row):
            mission_id = str(analysis.get("mission_id", ""))
            with cols[j]:
                with st.container(border=True):
                    title = f"Mission {mission_id[:8]}" if mission_id else "Mission"
                    st.markdown(f'<div class="library-card-title">{title}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<p class="library-card-meta">{str(analysis.get("created_at", ""))}</p>',
                        unsafe_allow_html=True,
                    )

                    route_points = route_map.get(mission_id, [])
                    mine_points = mine_map.get(mission_id, [])
                    snapshot_chart = build_route_snapshot_chart(route_points, mine_points)
                    if snapshot_chart is not None:
                        st.altair_chart(snapshot_chart, width="stretch")
                    else:
                        st.info("Route preview unavailable for this mission.")

                    hazards = int(analysis.get("hazards", 0)) if str(analysis.get("hazards", "")).strip() else 0
                    target_x = float(analysis.get("target_x", 0.0))
                    target_y = float(analysis.get("target_y", 0.0))
                    detection_radius = float(analysis.get("detection_radius", DEFAULT_DETECTION_RADIUS))
                    status_label = status_chip_text(analysis.get("status", "completed"))
                    st.markdown(
                        (
                            f'<span class="library-chip">Status: {status_label}</span>'
                            f'<span class="library-chip">Hazards: {hazards}</span>'
                            f'<span class="library-chip">Detection: {detection_radius:.1f} m</span>'
                        ),
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Destination: ({target_x:.1f}, {target_y:.1f})")
                    action_open, action_delete = st.columns(2)
                    if action_open.button("Open Mission", key=f"open_analysis_{mission_id}", width="stretch"):
                        st.session_state.current_mission_id = mission_id
                        st.session_state.start_x = float(analysis.get("start_x", MAP_MIN))
                        st.session_state.start_y = float(analysis.get("start_y", MAP_MIN))
                        st.session_state.target_x = float(analysis.get("target_x", MAP_MAX))
                        st.session_state.target_y = float(analysis.get("target_y", MAP_MAX))
                        st.session_state.detection_radius = float(analysis.get("detection_radius", DEFAULT_DETECTION_RADIUS))
                        st.session_state.route_requested = True
                        st.session_state.has_run = True
                        st.session_state.animate_playback = False
                        st.session_state.dashboard_view = "analysis"
                        st.rerun()
                    if action_delete.button("Delete Mission", key=f"delete_analysis_{mission_id}", width="stretch"):
                        success, msg = delete_analysis_mission(mission_id)
                        if success:
                            st.toast(msg, icon="🗑️")
                        else:
                            st.warning(msg)
                        st.rerun()

def load_ground_truth_points():
    """Load generated mine spots for preview before mission execution."""
    try:
        df = pd.read_csv("ground_truth_mines.csv")
        if not {"X", "Y"}.issubset(df.columns):
            return pd.DataFrame(columns=["x", "y"])
        preview = df.rename(columns={"X": "x", "Y": "y"})[["x", "y"]]
        return clamp_df_xy(preview)
    except FileNotFoundError:
        return pd.DataFrame(columns=["x", "y"])

def file_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_telemetry_metadata():
    if not os.path.exists(TELEMETRY_META_PATH):
        return None
    try:
        with open(TELEMETRY_META_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_local_tactical_data():
    try:
        with open("tactical_map_data.json", "r") as f:
            return json.load(f)
    except Exception:
        return None

def tactical_output_matches_context(tactical_data, start_x, start_y, target_x, target_y, detection_radius, minefield_hash, mission_id):
    """Check whether cached tactical output already matches the requested mission context."""
    if not isinstance(tactical_data, dict):
        return False
    ctx = tactical_data.get("run_context", {})
    if not isinstance(ctx, dict):
        return False
    try:
        return (
            abs(float(ctx.get("start_x", 0.0)) - float(start_x)) < 1e-6
            and abs(float(ctx.get("start_y", 0.0)) - float(start_y)) < 1e-6
            and abs(float(ctx.get("target_x", -1.0)) - float(target_x)) < 1e-6
            and abs(float(ctx.get("target_y", -1.0)) - float(target_y)) < 1e-6
            and abs(float(ctx.get("detection_radius", -1.0)) - float(detection_radius)) < 1e-6
            and str(ctx.get("minefield_hash", "")) == str(minefield_hash or "")
            and str(ctx.get("mission_id", "")) == str(mission_id or "")
        )
    except Exception:
        return False

def run_pipeline_command(cmd, step_name, log_box):
    """Run subprocess and stream recent output lines into the dashboard."""
    log_box.markdown(f"- Running `{step_name}`...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured = []
    if proc.stdout is not None:
        for line in proc.stdout:
            captured.append(line.rstrip("\n"))
            log_box.code("\n".join(captured[-14:]), language="text")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{step_name} failed (exit {rc})")
    log_box.markdown(f"- `{step_name}` complete.")

def calculate_distance(points):
    """Calculates total 3D path distance"""
    dist = 0
    if len(points) < 2: return 0
    for i in range(1, len(points)):
        p1, p2 = points[i-1], points[i]
        
        # Handle dicts (Phase 4 A* route output) vs Arrays (Phase 4 engine trajectory output)
        x1 = p1['x'] if isinstance(p1, dict) else p1[0]
        y1 = p1['y'] if isinstance(p1, dict) else p1[1]
        z1 = p1['z'] if isinstance(p1, dict) else p1[2]
        
        x2 = p2['x'] if isinstance(p2, dict) else p2[0]
        y2 = p2['y'] if isinstance(p2, dict) else p2[1]
        z2 = p2['z'] if isinstance(p2, dict) else p2[2]
        
        dist += ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
    return dist

def clamp_df_xy(df, x_col='x', y_col='y'):
    """Clamp x/y columns to fixed map bounds for rendering safety."""
    if x_col in df.columns:
        df[x_col] = df[x_col].clip(MAP_MIN, MAP_MAX)
    if y_col in df.columns:
        df[y_col] = df[y_col].clip(MAP_MIN, MAP_MAX)
    return df


def clamp_xy(x, y):
    """Clamp a single x/y coordinate pair to map bounds."""
    return (
        max(MAP_MIN, min(MAP_MAX, float(x))),
        max(MAP_MIN, min(MAP_MAX, float(y))),
    )


def build_poi_style_df(start_x, start_y, end_x, end_y, end_label="Finish"):
    """Return styled POI markers so start/end stand out on dense maps."""
    points = [
        {
            "x": float(start_x),
            "y": float(start_y),
            "label": "Start",
            "marker_shape": "circle",
            "marker_color": COLOR_START_MARKER,
            "sort_order": 0,
        },
        {
            "x": float(end_x),
            "y": float(end_y),
            "label": str(end_label),
            "marker_shape": "diamond",
            "marker_color": COLOR_FINISH_MARKER,
            "sort_order": 1,
        },
    ]
    return clamp_df_xy(pd.DataFrame(points))


def extract_xy_from_chart_event(event):
    """Extract x/y from Streamlit chart selection payloads (best-effort)."""
    if event is None:
        return None

    payload = event
    if not isinstance(payload, (dict, list)):
        try:
            payload = dict(event)
        except Exception:
            selection = getattr(event, "selection", None)
            if selection is not None:
                payload = selection

    def _walk(obj):
        if isinstance(obj, dict):
            if "x" in obj and "y" in obj:
                try:
                    return float(obj["x"]), float(obj["y"])
                except (TypeError, ValueError):
                    pass
            for value in obj.values():
                found = _walk(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    return _walk(payload)

@st.dialog("Hazard Verification Details", width="large")
def show_mine_details_dialog(detail_rows):
    """Modal popup for selected mine verification details."""
    st.table(pd.DataFrame(detail_rows))
    if st.button("Close", key="close_mine_details_dialog_btn"):
        st.session_state.show_mine_dialog = False
        st.rerun()

def render_dashboard(show_overview=True):
    apply_dashboard_theme()

    # --- State Initialization & Sidebar Inputs ---
    if 'has_run' not in st.session_state:
        # Do not auto-run pipeline on initial page load/reload.
        st.session_state.has_run = True
        st.session_state.route_requested = False
        st.session_state.start_x = 0.0
        st.session_state.start_y = 0.0
        st.session_state.target_x = 500.0
        st.session_state.target_y = 500.0
        st.session_state.num_mines = 40
        st.session_state.detection_radius = DEFAULT_DETECTION_RADIUS
        st.session_state.mine_seed = int(time.time())
        st.session_state.current_mission_id = ""
        st.session_state.data_source = "local_json"
        st.session_state.animate_playback = False
        st.session_state.force_route_recompute = False

    if "dashboard_view" not in st.session_state:
        st.session_state.dashboard_view = "overview" if show_overview else "analysis"
    if "start_x" not in st.session_state:
        st.session_state.start_x = MAP_MIN
    if "start_y" not in st.session_state:
        st.session_state.start_y = MAP_MIN
    if "force_route_recompute" not in st.session_state:
        st.session_state.force_route_recompute = False

    if st.session_state.dashboard_view == "overview":
        render_analysis_overview()
        return

    with st.sidebar:
        if st.button("Back to Mission Library", width="stretch"):
            st.session_state.dashboard_view = "overview"
            st.rerun()
        st.markdown("### Mission Setup")
        
        st.markdown("#### Destination")
        route_profile = st.selectbox(
            "Choose Destination Profile:",
            ["Default (0,0 -> 500,500)", "Custom Start + End"],
            index=0,
            key="route_profile_mode",
        )

        if route_profile == "Custom Start + End":
            st.caption("Set both start (Point A) and destination (Point B).")
            tap_target = st.radio(
                "Map tap sets:",
                ["Start (Point A)", "Finish (Point B)"],
                horizontal=True,
                key="tap_target_mode",
            )
            col_a_x, col_a_y = st.columns(2)
            pt_a_x = col_a_x.number_input(
                "Start X (Point A)",
                min_value=MAP_MIN,
                max_value=MAP_MAX,
                value=float(st.session_state.get("start_x", MAP_MIN)),
                step=10.0,
            )
            pt_a_y = col_a_y.number_input(
                "Start Y (Point A)",
                min_value=MAP_MIN,
                max_value=MAP_MAX,
                value=float(st.session_state.get("start_y", MAP_MIN)),
                step=10.0,
            )
            col_b_x, col_b_y = st.columns(2)
            pt_b_x = col_b_x.number_input(
                "Destination X (Point B)",
                min_value=MAP_MIN,
                max_value=MAP_MAX,
                value=float(st.session_state.get("target_x", MAP_MAX)),
                step=10.0,
            )
            pt_b_y = col_b_y.number_input(
                "Destination Y (Point B)",
                min_value=MAP_MIN,
                max_value=MAP_MAX,
                value=float(st.session_state.get("target_y", MAP_MAX)),
                step=10.0,
            )
        else:
            pt_a_x, pt_a_y = MAP_MIN, MAP_MIN
            pt_b_x, pt_b_y = MAP_MAX, MAP_MAX
            tap_target = "Finish (Point B)"

        st.caption(
            f"A ({pt_a_x:.1f}, {pt_a_y:.1f}) -> B ({pt_b_x:.1f}, {pt_b_y:.1f})"
        )

        st.markdown("#### Hazard Field")
        st.caption("Generate or refresh hazard placement for the operating area.")
        num_mines = st.slider(
            "Simulated Hazard Count",
            min_value=5,
            max_value=150,
            value=int(st.session_state.num_mines),
            step=5,
            help="Lower values reduce hazard density and usually produce simpler routes.",
        )
        detection_radius = st.slider(
            "Detection Radius (m)",
            min_value=15.0,
            max_value=30.0,
            value=max(15.0, min(30.0, float(st.session_state.detection_radius))),
            step=0.5,
            help="Hazards within this radius of the reconnaissance path are detected.",
        )

        col_gen, col_regen = st.columns(2)
        generate_mines_clicked = col_gen.button("Generate Field", width="stretch")
        regenerate_mines_clicked = col_regen.button("Refresh Field", width="stretch")

        if regenerate_mines_clicked:
            st.session_state.mine_seed = int(time.time() * 1000) % 100000000

        if generate_mines_clicked or regenerate_mines_clicked:
            # Hazard field operations should only update placement and never auto-run mission scan.
            st.session_state.route_requested = False
            st.session_state.has_run = True
            st.session_state.num_mines = num_mines
            st.session_state.detection_radius = detection_radius
            subprocess.run([
                "python3",
                "phase1_sim.py",
                "--generate_mines_only",
                "--num_mines",
                str(st.session_state.num_mines),
                "--mine_seed",
                str(st.session_state.mine_seed),
            ])
            st.success(
                f"Hazard field ready with {st.session_state.num_mines} spots "
                f"(seed {st.session_state.mine_seed})."
            )

        st.markdown("#### Analysis")
        if st.button("Run Route Analysis", width="stretch", type="primary"):
            st.session_state.route_requested = True
            st.session_state.has_run = False
            st.session_state.animate_playback = True
            # Manual runs should always execute Phase 2/A* to avoid stale route reuse.
            st.session_state.force_route_recompute = True
            st.session_state.start_x = pt_a_x
            st.session_state.start_y = pt_a_y
            st.session_state.target_x = pt_b_x
            st.session_state.target_y = pt_b_y
            st.session_state.num_mines = num_mines
            st.session_state.detection_radius = detection_radius

    # --- Processing Pipeline ---
    if st.session_state.route_requested and (not st.session_state.has_run):
        telemetry_exists = os.path.exists("telemetry_log.csv")
        minefield_hash = file_sha256("ground_truth_mines.csv")
        telemetry_meta = load_telemetry_metadata()

        telemetry_matches_current_context = (
            telemetry_exists
            and telemetry_meta is not None
            and minefield_hash is not None
            and abs(float(telemetry_meta.get("start_x", 0.0)) - float(st.session_state.start_x)) < 1e-6
            and abs(float(telemetry_meta.get("start_y", 0.0)) - float(st.session_state.start_y)) < 1e-6
            and abs(float(telemetry_meta.get("target_x", -1.0)) - float(st.session_state.target_x)) < 1e-6
            and abs(float(telemetry_meta.get("target_y", -1.0)) - float(st.session_state.target_y)) < 1e-6
            and abs(float(telemetry_meta.get("detection_radius", -1.0)) - float(st.session_state.detection_radius)) < 1e-6
            and str(telemetry_meta.get("minefield_hash", "")) == str(minefield_hash)
        )
        telemetry_meta_mission_id = str((telemetry_meta or {}).get("mission_id", "")).strip()
        if telemetry_matches_current_context and not telemetry_meta_mission_id:
            telemetry_matches_current_context = False

        mission_id = telemetry_meta_mission_id if telemetry_matches_current_context else str(uuid.uuid4())
        st.session_state.current_mission_id = mission_id

        local_tactical_data = load_local_tactical_data()
        tactical_matches_context = tactical_output_matches_context(
            local_tactical_data,
            st.session_state.start_x,
            st.session_state.start_y,
            st.session_state.target_x,
            st.session_state.target_y,
            st.session_state.detection_radius,
            minefield_hash,
            mission_id,
        )
        force_route_recompute = bool(st.session_state.get("force_route_recompute", False))
        if force_route_recompute:
            tactical_matches_context = False

        pipeline_status = st.status("Initializing mission processing pipeline...", expanded=True)
        pipeline_log = pipeline_status.empty()
        pipeline_log.markdown("- Preparing mission execution...")
        try:
            if not telemetry_matches_current_context:
                run_pipeline_command(
                    [
                        "python3",
                        "phase1_sim.py",
                        "--mission_id",
                        str(mission_id),
                        "--start_x",
                        str(st.session_state.start_x),
                        "--start_y",
                        str(st.session_state.start_y),
                        "--target_x",
                        str(st.session_state.target_x),
                        "--target_y",
                        str(st.session_state.target_y),
                        "--num_mines",
                        str(st.session_state.num_mines),
                        "--detection_radius",
                        str(st.session_state.detection_radius),
                        "--use_existing_mines",
                    ],
                    "Phase 1 simulation",
                    pipeline_log,
                )
            else:
                pipeline_log.markdown("- Reusing telemetry from previous matching run.")

            if not tactical_matches_context:
                run_pipeline_command(
                    [
                        "python3",
                        "phase2_engine.py",
                        "--mission_id",
                        str(mission_id),
                        "--start_x",
                        str(st.session_state.start_x),
                        "--start_y",
                        str(st.session_state.start_y),
                        "--target_x",
                        str(st.session_state.target_x),
                        "--target_y",
                        str(st.session_state.target_y),
                        "--detection_radius",
                        str(st.session_state.detection_radius),
                    ],
                    "Phase 2 routing engine",
                    pipeline_log,
                )
            else:
                pipeline_log.markdown("- Reusing tactical output from previous matching run.")
            pipeline_status.update(label="Mission processing complete. Rendering results...", state="complete")
        except Exception as exc:
            pipeline_status.update(label=f"Mission pipeline failed: {exc}", state="error")
            st.stop()
        persist_analysis_snapshot(
            mission_id=st.session_state.current_mission_id,
            start_x=st.session_state.start_x,
            start_y=st.session_state.start_y,
            target_x=st.session_state.target_x,
            target_y=st.session_state.target_y,
            detection_radius=st.session_state.detection_radius,
        )
        st.session_state.has_run = True
        st.session_state.force_route_recompute = False

    if not st.session_state.route_requested:
        st.markdown(
            """
            <div class="command-hero">
                <div class="command-eyebrow">Mission Workspace</div>
                <div class="command-title">Pre-Run Configuration</div>
                <p class="command-subtitle">Tune destination and hazard controls before executing route analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.title("Recon Mission Analysis Workspace")
        st.markdown("### Hazard Field Preview")
        st.caption("Fixed 500 x 500 operating area with generated hazard positions.")

        df_gt_preview = load_ground_truth_points()
        preview_base = alt.Chart(pd.DataFrame({"x": [MAP_MIN, MAP_MAX], "y": [MAP_MIN, MAP_MAX]})).mark_rect(opacity=0).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), title="Search Area X (meters)"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), title="Search Area Y (meters)")
        ).properties(height=600, width="container")

        preview_layers = [preview_base]
        if not df_gt_preview.empty:
            preview_mines = alt.Chart(df_gt_preview).mark_circle(size=60, color=COLOR_MINE_UNKNOWN, opacity=0.7, stroke=COLOR_POI_HALO, strokeWidth=1).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                tooltip=[alt.Tooltip("x:Q", format=".1f"), alt.Tooltip("y:Q", format=".1f")]
            )
            preview_layers.append(preview_mines)

        preview_poi = build_poi_style_df(pt_a_x, pt_a_y, pt_b_x, pt_b_y, end_label="Finish")
        preview_poi_halo = alt.Chart(preview_poi).mark_point(filled=True, size=SIZE_POI_HALO, color=COLOR_POI_HALO, opacity=0.88).encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
            shape=alt.Shape("marker_shape:N", scale=None, legend=None),
            order=alt.Order("sort_order:Q", sort="ascending"),
        )
        preview_poi_markers = alt.Chart(preview_poi).mark_point(filled=True, size=SIZE_POI_MARKER, opacity=1.0, stroke=COLOR_POI_STROKE, strokeWidth=1.6).encode(
            x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
            y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
            shape=alt.Shape("marker_shape:N", scale=None, legend=None),
            color=alt.Color("marker_color:N", scale=None, legend=None),
            tooltip='label',
            order=alt.Order("sort_order:Q", sort="ascending"),
        )
        preview_layers.extend([preview_poi_halo, preview_poi_markers])

        click_grid_values = np.arange(MAP_MIN, MAP_MAX + 0.1, 10.0)
        click_grid = pd.DataFrame(
            [(float(x), float(y)) for x in click_grid_values for y in click_grid_values],
            columns=["x", "y"],
        )
        route_pick = alt.selection_point(
            name="route_pick",
            fields=["x", "y"],
            on="click",
            nearest=True,
            empty=False,
        )
        click_layer = alt.Chart(click_grid).mark_circle(size=170, opacity=0.001).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
            tooltip=[
                alt.Tooltip("x:Q", title="X", format=".1f"),
                alt.Tooltip("y:Q", title="Y", format=".1f"),
            ],
        ).add_params(route_pick)
        preview_chart = alt.layer(*preview_layers, click_layer).configure_view(strokeOpacity=0)
        preview_event = st.altair_chart(
            preview_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="route_pick",
            key="route_point_picker_chart",
        )

        selected_xy = extract_xy_from_chart_event(preview_event)
        if selected_xy is not None and route_profile == "Custom Start + End":
            selected_x, selected_y = clamp_xy(*selected_xy)
            if tap_target.startswith("Start"):
                changed = (
                    abs(float(st.session_state.get("start_x", MAP_MIN)) - selected_x) > 1e-6
                    or abs(float(st.session_state.get("start_y", MAP_MIN)) - selected_y) > 1e-6
                )
                if changed:
                    st.session_state.start_x = selected_x
                    st.session_state.start_y = selected_y
                    st.rerun()
            else:
                changed = (
                    abs(float(st.session_state.get("target_x", MAP_MAX)) - selected_x) > 1e-6
                    or abs(float(st.session_state.get("target_y", MAP_MAX)) - selected_y) > 1e-6
                )
                if changed:
                    st.session_state.target_x = selected_x
                    st.session_state.target_y = selected_y
                    st.rerun()

        st.info("Configure destination and hazard field, then click `Run Route Analysis` to start.")
        return

    # --- Data Loading (Post-Calculation) ---
    data = load_tactical_data(st.session_state.get("current_mission_id", "").strip() or None)
    if not data: return
    
    mission_label = str(st.session_state.get("current_mission_id", "")).strip()[:12] or "n/a"
    st.markdown(
        f"""
        <div class="command-hero">
            <div class="command-eyebrow">Mission Workspace</div>
            <div class="command-title">Recon Mission Analysis Workspace</div>
            <p class="command-subtitle">Mission {mission_label} loaded. Review terrain scan, trajectory, and route recommendation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prepared = _prepare_dashboard_frames_cached(data, float(st.session_state.detection_radius))
    df_traj = prepared["df_traj"]
    df_mines = prepared["df_mines"]
    df_route = prepared["df_route"]
    df_terrain = prepared["df_terrain"]
    df_offroad = prepared["df_offroad"]
    df_gt = prepared["df_gt"]
    df_vision = prepared["df_vision"]

    # --- Sidebar Metrics ---
    with st.sidebar:
        st.markdown("---")
        st.caption(f"Data Source: {st.session_state.get('data_source', 'unknown')}")
        if st.session_state.get("current_mission_id"):
            st.caption(f"Mission ID: {st.session_state.get('current_mission_id')}")
        ins_dist = calculate_distance(data['trajectory'])
        convoy_dist = calculate_distance(data['safe_route'])
        llm_trace = data.get("llm_vision_trace", {})
        total_vision = len(df_vision)
        vision_mines = int(df_vision["decision_is_mine"].sum()) if (not df_vision.empty and "decision_is_mine" in df_vision.columns) else 0
        vision_not_mines = max(0, total_vision - vision_mines)
        
        render_sidebar_metric_card("Detected Hazards", len(data['anomalies']), "amber")
        render_sidebar_metric_card("Recommended Route Distance", f"{convoy_dist:.1f} m", "blue")
        render_sidebar_metric_card("Peak Terrain Altitude", f"{df_traj['z'].max():.1f} m", "violet")
        render_sidebar_metric_card("Gemini Verified Hazards", vision_mines, "green")
        render_sidebar_metric_card("Gemini Rejected Hazards", vision_not_mines, "teal")
        if isinstance(llm_trace, dict) and llm_trace:
            render_sidebar_metric_card("Gemini Calls", int(llm_trace.get("gemini_calls", 0)), "slate")
            render_sidebar_metric_card("Gemini Failures", int(llm_trace.get("gemini_failures", 0)), "rose")
            with st.expander("Gemini Trace", expanded=False):
                st.caption(f"Model: {llm_trace.get('model_requested', 'n/a')}")
                events = llm_trace.get("trace_events", [])
                if events:
                    st.json(events[:10], expanded=False)
                else:
                    st.caption("No trace events captured.")

        st.markdown("---")
        st.caption("A* routing minimizes hazard exposure and steep terrain transitions.")

    # --- Main Visualization Area (Altair) ---
    st.markdown("### Recon Flight Path and Terrain Scan")
    st.caption("Initial aerial pass used to map terrain and identify potential hazards.")
    
    base = alt.Chart(pd.DataFrame({'x': [MAP_MIN, MAP_MAX], 'y': [MAP_MIN, MAP_MAX]})).mark_rect(opacity=0).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), title='Search Area X (meters)'),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX]), title='Search Area Y (meters)')
    ).properties(height=600, width='container')

    heatmap = alt.Chart(df_terrain).mark_rect(opacity=0.6).encode(
        x='x:Q', x2='x2:Q', y='y:Q', y2='y2:Q',
        color=alt.Color(
            'z:Q',
            scale=alt.Scale(scheme='blues', domain=[0, 45]),
            title='Altitude (m)',
            legend=alt.Legend(orient='bottom')
        ),
        tooltip=[alt.Tooltip('x:Q', title='X', format='.1f'), alt.Tooltip('y:Q', title='Y', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    line_traj = alt.Chart(df_traj).mark_line(strokeWidth=2, opacity=0.85, strokeDash=[4,4]).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        order='drone_step:Q',
        detail='drone_id:N',
        color=alt.Color('drone_id:N', legend=alt.Legend(title='Drone')),
        tooltip=[
            alt.Tooltip('drone_id:N', title='Drone'),
            alt.Tooltip('x:Q', format='.1f'),
            alt.Tooltip('y:Q', format='.1f'),
            alt.Tooltip('z:Q', title='Altitude', format='.1f')
        ]
    )
    
    max_traj_step = int(df_traj['drone_step'].max()) if not df_traj.empty else 0
    df_gt_static = df_gt.copy()
    df_gt_static["detected"] = df_gt_static["reveal_step"] <= max_traj_step
    df_gt_static["gemini_color"] = np.where(
        df_gt_static["detected"],
        np.where(df_gt_static["decision_label"] == "not_mine", COLOR_MINE_NOT_MINE, COLOR_MINE_DETECTED),
        COLOR_MINE_UNKNOWN,
    )
    df_gt_static["gemini_size"] = np.where(
        df_gt_static["detected"],
        np.where(df_gt_static["decision_label"] == "not_mine", SIZE_MINE_NOT_MINE, SIZE_MINE_DETECTED),
        SIZE_MINE_UNKNOWN,
    )
    df_gt_static["gemini_opacity"] = np.where(df_gt_static["detected"], 0.9, 0.35)
    scatter_mines_state = alt.Chart(df_gt_static).mark_circle(stroke=COLOR_POI_HALO, strokeWidth=1).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        size=alt.Size("gemini_size:Q", scale=None, legend=None),
        opacity=alt.Opacity("gemini_opacity:Q", scale=None, legend=None),
        color=alt.Color("gemini_color:N", scale=None, legend=None),
        tooltip=[
            alt.Tooltip('mine_id:Q', title='Mine ID'),
            alt.Tooltip('x:Q', title='X', format='.1f'),
            alt.Tooltip('y:Q', title='Y', format='.1f'),
            alt.Tooltip('detected:N', title='Detected by Drone'),
            alt.Tooltip('decision_label:N', title='Gemini Decision'),
            alt.Tooltip('status:N', title='Gemini Status'),
            alt.Tooltip('source:N', title='Decision Source'),
            alt.Tooltip('model:N', title='Model'),
            alt.Tooltip('confidence:Q', title='Confidence', format='.2f'),
            alt.Tooltip('error:N', title='Error')
        ]
    )

    # Dynamic Safe Convoy Route
    route_glow = alt.Chart(df_route).mark_line(strokeWidth=12, opacity=0.3).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        order='index:Q', color=alt.value(COLOR_ROUTE_GLOW)
    )
    line_route = alt.Chart(df_route).mark_line(strokeWidth=4, opacity=1.0).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        order='index:Q',
        color=alt.value(COLOR_ROUTE_LINE),
        tooltip=[alt.Tooltip('x:Q', title='X', format='.1f'), alt.Tooltip('y:Q', title='Y', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    if not df_route.empty:
        route_start_x = float(df_route.iloc[0]["x"])
        route_start_y = float(df_route.iloc[0]["y"])
        route_end_x = float(df_route.iloc[-1]["x"])
        route_end_y = float(df_route.iloc[-1]["y"])
    else:
        route_start_x = float(st.session_state.get("start_x", MAP_MIN))
        route_start_y = float(st.session_state.get("start_y", MAP_MIN))
        route_end_x = float(st.session_state.get("target_x", MAP_MAX))
        route_end_y = float(st.session_state.get("target_y", MAP_MAX))
    poi_data = build_poi_style_df(
        route_start_x,
        route_start_y,
        route_end_x,
        route_end_y,
        end_label="Finish",
    )
    poi_halo = alt.Chart(poi_data).mark_point(filled=True, size=SIZE_POI_HALO, color=COLOR_POI_HALO, opacity=0.88).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        shape=alt.Shape("marker_shape:N", scale=None, legend=None),
        order=alt.Order("sort_order:Q", sort="ascending"),
    )
    scatter_poi = alt.Chart(poi_data).mark_point(filled=True, size=SIZE_POI_MARKER, opacity=1.0, stroke=COLOR_POI_STROKE, strokeWidth=1.6).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
        shape=alt.Shape("marker_shape:N", scale=None, legend=None),
        color=alt.Color("marker_color:N", scale=None, legend=None),
        tooltip='label',
        order=alt.Order("sort_order:Q", sort="ascending"),
    )
    # Compile layer sets
    layers_convoy = [base, heatmap, line_traj, scatter_mines_state, route_glow, line_route, poi_halo, scatter_poi]
    layers_search = [base, line_traj, scatter_mines_state, poi_halo, scatter_poi]
    
    chart_convoy = alt.layer(*layers_convoy).configure_view(strokeOpacity=0)
    chart_search_static = alt.layer(*layers_search).configure_view(strokeOpacity=0)

    should_animate = bool(st.session_state.get("animate_playback", False))

    # Render animated search chart only for fresh mission runs.
    search_placeholder = st.empty()
    if should_animate:
        search_frames = max(90, min(220, max_traj_step + 1))

        for frame in range(1, search_frames + 1):
            progress = frame / search_frames
            step_threshold = int(progress * max_traj_step) if max_traj_step > 0 else 0

            df_traj_chunk = df_traj[df_traj['drone_step'] <= step_threshold]
            line_traj_chunk = alt.Chart(df_traj_chunk).mark_line(strokeWidth=2, opacity=0.85, strokeDash=[4,4]).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                order='drone_step:Q',
                detail='drone_id:N',
                color=alt.Color('drone_id:N', legend=alt.Legend(title='Drone')),
                tooltip=[
                    alt.Tooltip('drone_id:N', title='Drone'),
                    alt.Tooltip('x:Q', format='.1f'),
                    alt.Tooltip('y:Q', format='.1f'),
                    alt.Tooltip('z:Q', title='Altitude', format='.1f')
                ]
            )
            
            # Use exact mine coordinates and change color in-place as drones come within radius.
            df_gt_chunk = df_gt.copy()
            df_gt_chunk["detected"] = df_gt_chunk["reveal_step"] <= step_threshold
            df_gt_chunk["gemini_color"] = np.where(
                df_gt_chunk["detected"],
                np.where(df_gt_chunk["decision_label"] == "not_mine", COLOR_MINE_NOT_MINE, COLOR_MINE_DETECTED),
                COLOR_MINE_UNKNOWN,
            )
            df_gt_chunk["gemini_size"] = np.where(
                df_gt_chunk["detected"],
                np.where(df_gt_chunk["decision_label"] == "not_mine", SIZE_MINE_NOT_MINE, SIZE_MINE_DETECTED),
                SIZE_MINE_UNKNOWN,
            )
            df_gt_chunk["gemini_opacity"] = np.where(df_gt_chunk["detected"], 0.9, 0.35)
            scatter_mines_chunk = alt.Chart(df_gt_chunk).mark_circle(stroke=COLOR_POI_HALO, strokeWidth=1).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                size=alt.Size("gemini_size:Q", scale=None, legend=None),
                opacity=alt.Opacity("gemini_opacity:Q", scale=None, legend=None),
                color=alt.Color("gemini_color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip('mine_id:Q', title='Mine ID'),
                    alt.Tooltip('x:Q', title='X', format='.1f'),
                    alt.Tooltip('y:Q', title='Y', format='.1f'),
                    alt.Tooltip('decision_label:N', title='Gemini Decision'),
                    alt.Tooltip('status:N', title='Gemini Status'),
                    alt.Tooltip('source:N', title='Decision Source'),
                    alt.Tooltip('confidence:Q', title='Confidence', format='.2f'),
                ]
            )
            
            layers_search_chunk = [base, line_traj_chunk, scatter_mines_chunk, poi_halo, scatter_poi]
            chart_search_chunk = alt.layer(*layers_search_chunk).configure_view(strokeOpacity=0)
            search_placeholder.altair_chart(chart_search_chunk, width="stretch")
            time.sleep(0.018)

    search_placeholder.altair_chart(chart_search_static, width="stretch")
    
    # --- PHASE 2 GROUND CONVOY CHART ---
    st.markdown("---")
    st.markdown("### Final Ground Navigation Recommendation")
    st.caption("Computed route balancing safety constraints and terrain slope.")
    convoy_placeholder = st.empty()
    if should_animate and not df_route.empty:
        route_frames = max(70, min(180, len(df_route)))
        max_route_idx = len(df_route) - 1
        for frame in range(1, route_frames + 1):
            progress = frame / route_frames
            route_idx_threshold = int(progress * max_route_idx)
            df_route_chunk = df_route.iloc[:route_idx_threshold + 1]

            route_glow_chunk = alt.Chart(df_route_chunk).mark_line(strokeWidth=12, opacity=0.3).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                order='index:Q', color=alt.value(COLOR_ROUTE_GLOW)
            )
            line_route_chunk = alt.Chart(df_route_chunk).mark_line(strokeWidth=4, opacity=1.0).encode(
                x=alt.X('x:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                y=alt.Y('y:Q', scale=alt.Scale(domain=[MAP_MIN, MAP_MAX])),
                order='index:Q',
                color=alt.value(COLOR_ROUTE_LINE),
                tooltip=[alt.Tooltip('x:Q', title='X', format='.1f'), alt.Tooltip('y:Q', title='Y', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
            )

            convoy_layers_chunk = [base, heatmap, line_traj, scatter_mines_state, route_glow_chunk, line_route_chunk, poi_halo, scatter_poi]
            chart_convoy_chunk = alt.layer(*convoy_layers_chunk).configure_view(strokeOpacity=0)
            convoy_placeholder.altair_chart(chart_convoy_chunk, width="stretch")
            time.sleep(0.018)

    convoy_placeholder.altair_chart(chart_convoy, width="stretch")
    st.session_state.animate_playback = False

    st.markdown("---")
    # --- Technical Breakdown ---
    st.markdown("### Routing and Verification Methodology")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.expander("Recon Flight Mapping", expanded=True):
            st.markdown("""
            **Dynamic Hazard Avoidance:**
            The reconnaissance path continuously adapts as hazards are detected, preserving safe separation while
            maintaining progress toward the destination.
            """)
            
        with st.expander("Ground Route Feasibility"):
            st.markdown("""
            **Terrain Assessment:**
            Elevation variance and route deviations are evaluated to determine whether ground vehicles can move through
            the same corridor with acceptable risk.
            """)
            
    with col_b:
        with st.expander("A* Route Computation", expanded=True):
            st.markdown("""
            **Multi-constraint Navigation:**
            A* pathfinding combines distance, terrain steepness, and hard exclusion zones to produce a route that
            favors safer and more stable ground traversal.
            """)
            
if __name__ == "__main__":
    render_dashboard(show_overview=True)
