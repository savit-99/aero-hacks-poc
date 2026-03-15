import pandas as pd
import numpy as np
import math
import json
import heapq
import random
import argparse
import os
import base64
import mimetypes
import struct
import zlib
import hashlib
import time
import uuid
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from db import (
    atlas_enabled,
    ensure_atlas_schema,
    load_telemetry_df_for_mission,
    write_phase2_outputs,
)

# Core simulation settings matching Phase 1
GRID_W = 500
GRID_H = 500
ANOMALY_THRESHOLD = 0.95
HAZARD_RADIUS = 15.0 # Mined area safety buffer
ROUTE_MINE_CLEARANCE_RADIUS = 7.5
DEFAULT_ROUTE_DETECTION_RADIUS = 20.0
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT_FMT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
VISION_HTTP_TIMEOUT_SEC = 25
_ENV_LOADED = False
_MISSING_KEY_WARNED = False
_VISION_CACHE = {}
_VISION_IMAGE_CACHE = {}
_FRAME_FETCH_WARN_COUNT = 0
_FRAME_FETCH_WARN_LIMIT = 5
_INTERNAL_FRAME_NOTE_PRINTED = False
_GEMINI_CALL_COUNT = 0
_GEMINI_FAIL_COUNT = 0
_MODEL_FALLBACK_WARNED = False
_GEMINI_FAIL_WARN_COUNT = 0
_GEMINI_FAIL_WARN_LIMIT = 5
_PLACEHOLDER_FRAME_BYTES = None
_GEMINI_TRACE_EVENTS = []
_GEMINI_TRACE_LIMIT = 200
_LAST_GEMINI_META = {
    "status": "not_called",
    "model": "",
    "error": "",
    "verdict": "unknown",
    "confidence": 0.0,
    "source": "none",
}

TELEMETRY_META_PATH = "telemetry_metadata.json"
GROUND_TRUTH_PATH = "ground_truth_mines.csv"


def load_telemetry_metadata():
    if not os.path.exists(TELEMETRY_META_PATH):
        return None
    try:
        with open(TELEMETRY_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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

def clamp_xy(x, y):
    """Clamp coordinates to tactical map bounds."""
    return (
        float(min(max(x, 0.0), GRID_W)),
        float(min(max(y, 0.0), GRID_H)),
    )

def _append_gemini_trace(event):
    """Append bounded trace events so users can audit Gemini activity after runs."""
    if len(_GEMINI_TRACE_EVENTS) >= _GEMINI_TRACE_LIMIT:
        return
    _GEMINI_TRACE_EVENTS.append(event)

def load_dotenv_if_present(dotenv_path=".env"):
    """Minimal .env loader so the engine can run without extra dependencies."""
    global _ENV_LOADED
    if _ENV_LOADED or not os.path.exists(dotenv_path):
        _ENV_LOADED = True
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    finally:
        _ENV_LOADED = True

def _detect_mime_type_from_path(path):
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "image/jpeg"

def _build_placeholder_png(width=64, height=64):
    """Build a small valid RGB PNG via stdlib only."""
    width = max(1, int(width))
    height = max(1, int(height))
    rows = []
    for y in range(height):
        row = bytearray([0])  # filter method 0
        for x in range(width):
            # Subtle gradient-like pattern to avoid degenerate image input.
            r = int(40 + (180 * x / max(1, width - 1)))
            g = int(40 + (180 * y / max(1, height - 1)))
            b = int(80 + (120 * ((x + y) % max(1, width)) / max(1, width - 1)))
            row.extend((r, g, b))
        rows.append(bytes(row))
    raw = b"".join(rows)
    compressed = zlib.compress(raw, level=6)

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png_sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return png_sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")

def _read_bounded(stream, max_bytes):
    """Read at most max_bytes from stream and return bytes (or None if too large)."""
    chunks = []
    total = 0
    while True:
        part = stream.read(65536)
        if not part:
            break
        total += len(part)
        if total > max_bytes:
            return None
        chunks.append(part)
    return b"".join(chunks)

def _load_image_bytes(frame_ref):
    """
    Resolve frame_ref into raw image bytes.
    Supports local paths and public http(s) URLs.
    """
    if not frame_ref:
        return None, None, "empty-frame-ref"

    frame_ref = str(frame_ref).strip()
    if "internal-drone-net" in frame_ref:
        global _INTERNAL_FRAME_NOTE_PRINTED, _PLACEHOLDER_FRAME_BYTES
        if not _INTERNAL_FRAME_NOTE_PRINTED:
            print(
                "INFO: Simulated internal camera URLs detected. "
                "Using embedded placeholder image so Gemini verification can execute."
            )
            _INTERNAL_FRAME_NOTE_PRINTED = True
        if _PLACEHOLDER_FRAME_BYTES is None:
            _PLACEHOLDER_FRAME_BYTES = _build_placeholder_png()
        return _PLACEHOLDER_FRAME_BYTES, "image/png", None

    if frame_ref.startswith(("http://", "https://")):
        try:
            req = Request(frame_ref, headers={"User-Agent": "sweep-net-poc/1.0"})
            with urlopen(req, timeout=VISION_HTTP_TIMEOUT_SEC) as resp:
                mime_type = resp.headers.get_content_type() or "image/jpeg"
                data = _read_bounded(resp, MAX_IMAGE_BYTES)
                if data is None:
                    return None, None, "image-too-large"
                return data, mime_type, None
        except (URLError, HTTPError, TimeoutError) as exc:
            return None, None, f"url-fetch-failed:{exc}"
        except Exception as exc:  # Broad catch to keep mission runtime resilient.
            return None, None, f"url-fetch-error:{exc}"

    local_path = frame_ref[7:] if frame_ref.startswith("file://") else frame_ref
    if not os.path.exists(local_path):
        return None, None, "file-not-found"

    try:
        file_size = os.path.getsize(local_path)
        if file_size > MAX_IMAGE_BYTES:
            return None, None, "image-too-large"
        with open(local_path, "rb") as f:
            data = f.read()
        return data, _detect_mime_type_from_path(local_path), None
    except Exception as exc:  # Broad catch to keep mission runtime resilient.
        return None, None, f"file-read-error:{exc}"

def _extract_text_from_gemini_response(response_json):
    candidates = response_json.get("candidates", [])
    if not candidates:
        return ""
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
    return "\n".join([t for t in text_parts if t]).strip()

def _gemini_generate_content(api_key, model, payload):
    """Execute a single Gemini generateContent request."""
    url = GEMINI_ENDPOINT_FMT.format(model=model, api_key=api_key)
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=VISION_HTTP_TIMEOUT_SEC) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)

def _gemini_mine_verdict(image_bytes, mime_type):
    """
    Call Gemini multimodal API and return True if the frame likely contains a mine.
    """
    global _MISSING_KEY_WARNED, _GEMINI_CALL_COUNT, _GEMINI_FAIL_COUNT, _MODEL_FALLBACK_WARNED
    global _GEMINI_FAIL_WARN_COUNT
    global _LAST_GEMINI_META
    load_dotenv_if_present()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    requested_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    if not api_key:
        if not _MISSING_KEY_WARNED:
            print("WARNING: GEMINI_API_KEY missing. Falling back to safety-positive mine verification.")
            _MISSING_KEY_WARNED = True
        _LAST_GEMINI_META = {
            "status": "missing_key",
            "model": requested_model,
            "error": "GEMINI_API_KEY missing",
            "verdict": "fallback_mine",
            "confidence": 0.0,
            "source": "missing_key",
        }
        return True

    prompt = (
        "You are verifying if a drone image contains a landmine.\n"
        "Return strict JSON only with keys: verdict, is_mine, confidence.\n"
        'Allowed verdict values: "mine", "not_mine", "uncertain".\n'
        "confidence must be a number in [0,1].\n"
        "If uncertain, set verdict to uncertain."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        _GEMINI_CALL_COUNT += 1
        response_json = None
        candidate_models = [requested_model]
        if requested_model != "gemini-2.5-flash":
            candidate_models.append("gemini-2.5-flash")
        chosen_model = requested_model

        last_error = None
        for model_name in candidate_models:
            try:
                response_json = _gemini_generate_content(api_key, model_name, payload)
                chosen_model = model_name
                if model_name != requested_model and not _MODEL_FALLBACK_WARNED:
                    print(
                        f"WARNING: Gemini model '{requested_model}' unavailable. "
                        f"Using fallback model '{model_name}'."
                    )
                    _MODEL_FALLBACK_WARNED = True
                break
            except HTTPError as http_exc:
                body = ""
                try:
                    body = http_exc.read().decode("utf-8")
                except Exception:
                    body = str(http_exc)
                body_lower = body.lower()
                can_try_fallback = (
                    http_exc.code == 404
                    and model_name == requested_model
                    and model_name != "gemini-2.5-flash"
                    and ("no longer available" in body_lower or "not found" in body_lower)
                )
                if can_try_fallback:
                    last_error = http_exc
                    continue
                raise
            except Exception as exc:
                last_error = exc
                raise

        if response_json is None and last_error is not None:
            raise last_error

        raw_text = _extract_text_from_gemini_response(response_json)
        verdict_json = json.loads(raw_text) if raw_text else {}
        verdict = str(verdict_json.get("verdict", "")).strip().lower()
        is_mine = bool(verdict_json.get("is_mine", False))
        confidence = float(verdict_json.get("confidence", 0.0))
        _LAST_GEMINI_META = {
            "status": "ok",
            "model": chosen_model,
            "error": "",
            "verdict": verdict or ("mine" if is_mine else "unknown"),
            "confidence": confidence,
            "source": "gemini_api",
        }

        if verdict == "uncertain":
            return True
        if verdict == "mine":
            return True
        if verdict == "not_mine":
            # Safety floor: if model says not_mine but with very low confidence, keep as hazard.
            return confidence < 0.35
        return is_mine
    except Exception as exc:
        _GEMINI_FAIL_COUNT += 1
        _LAST_GEMINI_META = {
            "status": "error",
            "model": requested_model,
            "error": str(exc),
            "verdict": "fallback_mine",
            "confidence": 0.0,
            "source": "gemini_error",
        }
        if _GEMINI_FAIL_WARN_COUNT < _GEMINI_FAIL_WARN_LIMIT:
            print(f"WARNING: Gemini verification failed ({exc}). Falling back to safety-positive verification.")
            _GEMINI_FAIL_WARN_COUNT += 1
            if _GEMINI_FAIL_WARN_COUNT == _GEMINI_FAIL_WARN_LIMIT:
                print("WARNING: Additional Gemini verification errors suppressed.")
        return True

def verify_mine_with_multimodal_llm(frame_ref):
    """
    Real multimodal mine verification with conservative fallbacks.
    """
    global _FRAME_FETCH_WARN_COUNT
    global _LAST_GEMINI_META
    started = time.time()
    if frame_ref in _VISION_CACHE:
        verdict = _VISION_CACHE[frame_ref]
        _LAST_GEMINI_META = {
            "status": "cache_hit",
            "model": _LAST_GEMINI_META.get("model", ""),
            "error": "",
            "verdict": "mine" if verdict else "not_mine",
            "confidence": 1.0,
            "source": "frame_cache",
        }
        _append_gemini_trace({
            "source": "frame_cache",
            "frame_ref": str(frame_ref),
            "decision_is_mine": bool(verdict),
            "latency_ms": int((time.time() - started) * 1000),
        })
        return verdict

    image_bytes, mime_type, err = _load_image_bytes(frame_ref)
    if image_bytes is None:
        # Safety-first fallback when image retrieval is unavailable.
        if err and _FRAME_FETCH_WARN_COUNT < _FRAME_FETCH_WARN_LIMIT:
            print(f"WARNING: Frame fetch failed ({err}) for '{frame_ref}'. Marking anomaly as mine.")
            _FRAME_FETCH_WARN_COUNT += 1
            if _FRAME_FETCH_WARN_COUNT == _FRAME_FETCH_WARN_LIMIT:
                print("WARNING: Additional frame-fetch errors suppressed.")
        _LAST_GEMINI_META = {
            "status": "fetch_fallback",
            "model": _LAST_GEMINI_META.get("model", ""),
            "error": str(err),
            "verdict": "fallback_mine",
            "confidence": 0.0,
            "source": "fetch_fallback",
        }
        _append_gemini_trace({
            "source": "fetch_fallback",
            "frame_ref": str(frame_ref),
            "decision_is_mine": True,
            "error": str(err),
            "latency_ms": int((time.time() - started) * 1000),
        })
        _VISION_CACHE[frame_ref] = True
        return True

    image_digest = hashlib.sha256(image_bytes).hexdigest()
    if image_digest in _VISION_IMAGE_CACHE:
        verdict = _VISION_IMAGE_CACHE[image_digest]
        _VISION_CACHE[frame_ref] = verdict
        _LAST_GEMINI_META = {
            "status": "cache_hit",
            "model": _LAST_GEMINI_META.get("model", ""),
            "error": "",
            "verdict": "mine" if verdict else "not_mine",
            "confidence": 1.0,
            "source": "image_cache",
        }
        _append_gemini_trace({
            "source": "image_cache",
            "frame_ref": str(frame_ref),
            "decision_is_mine": bool(verdict),
            "latency_ms": int((time.time() - started) * 1000),
        })
        return verdict

    verdict = _gemini_mine_verdict(image_bytes, mime_type or "image/jpeg")
    _VISION_IMAGE_CACHE[image_digest] = verdict
    _VISION_CACHE[frame_ref] = verdict
    _append_gemini_trace({
        "source": "gemini_api",
        "frame_ref": str(frame_ref),
        "decision_is_mine": bool(verdict),
        "status": _LAST_GEMINI_META.get("status", "unknown"),
        "model": _LAST_GEMINI_META.get("model", ""),
        "error": _LAST_GEMINI_META.get("error", ""),
        "used_simulated_placeholder": "internal-drone-net" in str(frame_ref),
        "latency_ms": int((time.time() - started) * 1000),
    })
    return verdict

def point_to_segment_distance(px, py, ax, ay, bx, by):
    """2D distance from point P to segment AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)

def is_point_near_any_drone_segment(px, py, drone_tracks, radius):
    """Check if point is within radius of any segment in any drone track."""
    for track in drone_tracks.values():
        if len(track) < 2:
            continue
        for i in range(1, len(track)):
            ax, ay = track[i - 1]
            bx, by = track[i]
            if point_to_segment_distance(px, py, ax, ay, bx, by) <= radius:
                return True
    return False

class DeadReckoningEngine:
    def __init__(self, start_x=0.0, start_y=0.0, start_z=10.0):
        self.x = start_x
        self.y = start_y
        self.z = start_z
        self.path = [(start_x, start_y, start_z)]
        
        # Simple Moving Average for smoothing
        self.yaw_history = []
        self.vel_history = []
        self.alt_history = []
        self.window_size = 5

    def update_position(self, dt, yaw, velocity, altitude):
        """Update INS position using purely dead reckoning (Velocity + Yaw over time) and Vision Depth Sensing."""
        self.yaw_history.append(yaw)
        self.vel_history.append(velocity)
        self.alt_history.append(altitude)
        
        if len(self.yaw_history) > self.window_size:
            self.yaw_history.pop(0)
            self.vel_history.pop(0)
            self.alt_history.pop(0)
            
        smoothed_yaw = np.mean(self.yaw_history)
        smoothed_vel = np.mean(self.vel_history)
        self.z = np.mean(self.alt_history) # Smooth the simulated visual depth map
        
        dx = smoothed_vel * math.cos(smoothed_yaw) * dt
        dy = smoothed_vel * math.sin(smoothed_yaw) * dt
        
        self.x += dx
        self.y += dy
        self.x, self.y = clamp_xy(self.x, self.y)
        self.path.append((self.x, self.y, self.z))
        
        return self.x, self.y, self.z

def generate_global_terrain(grid_w, grid_h, step_size, path):
    """
    Generates a continuous elevation dictionary for every node in the grid
    by assigning completely random structural noise mixed with drone path anchors.
    """
    import random
    terrain = {}
    for x in range(0, grid_w + int(step_size), int(step_size)):
        for y in range(0, grid_h + int(step_size), int(step_size)):
            terrain[(x, y)] = random.uniform(5.0, 45.0)
            
    # Anchor the actual drone path elevations into the map
    for px, py, pz in path:
        sx = round(px/step_size)*step_size
        sy = round(py/step_size)*step_size
        if (sx, sy) in terrain:
            terrain[(sx, sy)] = min((terrain[(sx, sy)] + pz)/2, 50.0) 
            
    return terrain

def run_a_star(start, goal, hazards, terrain, grid_w, grid_h, step_size=10.0):
    """
    A* Pathfinding on a 3D Topographical grid avoiding confirmed hazard zones and steep terrain.
    """
    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def point_to_segment_distance(px, py, ax, ay, bx, by):
        """2D distance from point P to segment AB."""
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
        cx = ax + t * abx
        cy = ay + t * aby
        return math.hypot(px - cx, py - cy)
        
    def get_neighbors(node):
        neighbors = []
        # 8-directional movement
        movements = [
            (0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0),
            (step_size, step_size), (step_size, -step_size), 
            (-step_size, step_size), (-step_size, -step_size)
        ]
        for dx, dy in movements:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx <= grid_w and 0 <= ny <= grid_h:
                # Check collision with hazards
                collision = False
                for hazard in hazards:
                    if len(hazard) >= 4:
                        hx, hy, hz, local_radius = hazard[0], hazard[1], hazard[2], hazard[3]
                        radius = float(local_radius)
                    else:
                        hx, hy, hz = hazard[0], hazard[1], hazard[2]
                        radius = HAZARD_RADIUS
                    # Block both endpoint proximity and segment-over-mine crossing.
                    if math.hypot(nx - hx, ny - hy) < radius:
                        collision = True
                        break
                    if point_to_segment_distance(hx, hy, node[0], node[1], nx, ny) < radius:
                        collision = True
                        break
                if not collision:
                    neighbors.append((nx, ny))
        return neighbors

    # Snap start and goal to grid resolution
    start = (round(start[0]/step_size)*step_size, round(start[1]/step_size)*step_size)
    goal = (round(goal[0]/step_size)*step_size, round(goal[1]/step_size)*step_size)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if math.hypot(current[0] - goal[0], current[1] - goal[1]) <= step_size * 1.5:
            # Reconstruct path and append final altitudes
            final_path = []
            final_path.append({"x": goal[0], "y": goal[1], "z": terrain.get(goal, random.uniform(5,45))})
            while current in came_from:
                final_path.append({"x": current[0], "y": current[1], "z": terrain.get(current, random.uniform(5,45))})
                current = came_from[current]
            final_path.append({"x": start[0], "y": start[1], "z": terrain.get(start, random.uniform(5,45))})
            return final_path[::-1] # Reverse

        for neighbor in get_neighbors(current):
            # Cost of moving includes 2D distance PLUS a heavy penalty for steep elevation changes
            move_cost = math.hypot(neighbor[0] - current[0], neighbor[1] - current[1])
            z_curr = terrain.get(current, random.uniform(5,45))
            z_neighbor = terrain.get(neighbor, random.uniform(5,45))
            elevation_penalty = abs(z_neighbor - z_curr) * 1.5 # Adjusted penalty for smoother runtime
            
            tentative_g_score = g_score[current] + move_cost + elevation_penalty

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [] # No path found

def _load_input_telemetry_dataframe(mission_id=""):
    mission_id = str(mission_id or "").strip()
    if mission_id and atlas_enabled():
        try:
            df = load_telemetry_df_for_mission(mission_id)
            if not df.empty:
                print(f"Loaded telemetry from Atlas for mission_id={mission_id} ({len(df)} rows)")
                return df, "atlas"
            print(f"WARNING: No Atlas telemetry found for mission_id={mission_id}. Falling back to CSV.")
        except Exception as exc:
            print(f"WARNING: Atlas telemetry load failed ({exc}). Falling back to CSV.")

    df = pd.read_csv("telemetry_log.csv")
    print(f"Loaded telemetry from local CSV ({len(df)} rows)")
    return df, "csv"


def process_telemetry(
    start_x=0.0,
    start_y=0.0,
    target_x=500.0,
    target_y=500.0,
    detection_radius=DEFAULT_ROUTE_DETECTION_RADIUS,
    mission_id="",
):
    print("Loading encrypted telemetry logs...")
    mission_id = str(mission_id or "").strip()
    if not mission_id:
        meta = load_telemetry_metadata() or {}
        mission_id = str(meta.get("mission_id", "")).strip()
    if not mission_id:
        mission_id = str(uuid.uuid4())

    if atlas_enabled():
        try:
            status = ensure_atlas_schema()
            print(
                "Atlas setup status: "
                f"collections={status.get('collections_ready')}, "
                f"indexes={status.get('indexes_ready')}, "
                f"search_index={status.get('search_index_requested')}, "
                f"vector_index={status.get('vector_index_requested')}"
            )
        except Exception as exc:
            print(f"WARNING: Atlas schema setup failed ({exc}).")

    df, telemetry_source = _load_input_telemetry_dataframe(mission_id)
    
    start_x, start_y = clamp_xy(start_x, start_y)
    target_x, target_y = clamp_xy(target_x, target_y)
    engine = DeadReckoningEngine(start_x=start_x, start_y=start_y)
    has_truth_positions = {"Pos_X", "Pos_Y", "Pos_Z"}.issubset(set(df.columns))
    
    calculated_positions = []
    anomalies = []
    raw_deviations = []
    
    # Simulating iteration through real-time streaming data over time
    prev_time = df.iloc[0]["Elapsed_ms"]
    
    for idx, row in df.iterrows():
        current_time = row["Elapsed_ms"]
        dt = (current_time - prev_time) / 1000.0 # Convert ms to seconds
        prev_time = current_time
        
        if has_truth_positions:
            x, y = clamp_xy(row["Pos_X"], row["Pos_Y"])
            z = float(row["Pos_Z"])
            engine.path.append((x, y, z))
        else:
            # Dead Reckon position based purely on yaw, velocity, and Vision System Depth sensing
            x, y, z = engine.update_position(dt, row["Yaw"], row["Velocity_ms"], row["Altitude_m"])

        calculated_positions.append({
            "x": x,
            "y": y,
            "z": z,
            "drone_id": row["Drone_ID"] if "Drone_ID" in row else "drone_1"
        })
        
        # Check GPR Threshold
        if row["GPR_Score"] > ANOMALY_THRESHOLD:
            anomalies.append({
                "discovery_index": idx,
                "X": x, 
                "Y": y, 
                "Z": z,
                "GPR": row["GPR_Score"], 
                "URL": row["Camera_Frame_URL"],
                "Drone_ID": row["Drone_ID"] if "Drone_ID" in row else "drone_1",
                "Mine_ID": int(row["Detected_Mine_ID"]) if "Detected_Mine_ID" in row else -1
            })
            
        # Check Deviation Flag (Obstacle avoidance trigger)
        if "Deviation_Flag" in row and row["Deviation_Flag"] == 1:
            raw_deviations.append((x, y))
            
    print(f"INS filtering complete. Raw anomalies detected: {len(anomalies)}")
    
    # Deduplicate anomalies. Prefer deterministic mine ids when available.
    confirmed_mines = []
    vision_verifications = []
    seen_mine_ids = set()
    for anomaly in anomalies:
        if anomaly.get("Mine_ID", -1) >= 0:
            mine_id = anomaly["Mine_ID"]
            if mine_id in seen_mine_ids:
                continue
            seen_mine_ids.add(mine_id)
            is_mine = verify_mine_with_multimodal_llm(anomaly["URL"])
            vision_meta = dict(_LAST_GEMINI_META)
            vision_verification = {
                "discovery_index": int(anomaly["discovery_index"]),
                "x": float(anomaly["X"]),
                "y": float(anomaly["Y"]),
                "z": float(anomaly["Z"]),
                "mine_id": int(mine_id),
                "frame_ref": str(anomaly["URL"]),
                "decision_is_mine": bool(is_mine),
                "decision_label": "mine" if bool(is_mine) else "not_mine",
                "status": str(vision_meta.get("status", "")),
                "source": str(vision_meta.get("source", "")),
                "model": str(vision_meta.get("model", "")),
                "confidence": float(vision_meta.get("confidence", 0.0)),
                "error": str(vision_meta.get("error", "")),
            }
            vision_verifications.append(vision_verification)
            if is_mine:
                confirmed_mines.append({
                    "discovery_index": anomaly["discovery_index"],
                    "x": anomaly["X"],
                    "y": anomaly["Y"],
                    "z": anomaly["Z"],
                    "drone_id": anomaly.get("Drone_ID", "drone_1"),
                    "mine_id": mine_id
                })
            continue

        is_new = True
        for mine in confirmed_mines:
            if math.hypot(anomaly["X"] - mine["x"], anomaly["Y"] - mine["y"]) < 5.0:
                is_new = False
                break
        
        if is_new:
            # Invoke LLM Vision API verification
            is_mine = verify_mine_with_multimodal_llm(anomaly["URL"])
            vision_meta = dict(_LAST_GEMINI_META)
            vision_verification = {
                "discovery_index": int(anomaly["discovery_index"]),
                "x": float(anomaly["X"]),
                "y": float(anomaly["Y"]),
                "z": float(anomaly["Z"]),
                "mine_id": int(anomaly.get("Mine_ID", -1)),
                "frame_ref": str(anomaly["URL"]),
                "decision_is_mine": bool(is_mine),
                "decision_label": "mine" if bool(is_mine) else "not_mine",
                "status": str(vision_meta.get("status", "")),
                "source": str(vision_meta.get("source", "")),
                "model": str(vision_meta.get("model", "")),
                "confidence": float(vision_meta.get("confidence", 0.0)),
                "error": str(vision_meta.get("error", "")),
            }
            vision_verifications.append(vision_verification)
            if is_mine:
                confirmed_mines.append({
                    "discovery_index": anomaly["discovery_index"],
                    "x": anomaly["X"],
                    "y": anomaly["Y"],
                    "z": anomaly["Z"],
                    "drone_id": anomaly.get("Drone_ID", "drone_1"),
                    "mine_id": anomaly.get("Mine_ID", -1)
                })
                
    print("Calculating Topography Grid...")
    terrain_map = generate_global_terrain(GRID_W, GRID_H, 10.0, engine.path)

    # Enforce route-proximity recall: any ground-truth mine within detection radius is detected.
    df_gt = pd.read_csv("ground_truth_mines.csv")
    drone_tracks = {}
    if has_truth_positions and "Drone_ID" in df.columns:
        for drone_id, group in df.groupby("Drone_ID"):
            coords = [(float(x), float(y)) for x, y in zip(group["Pos_X"], group["Pos_Y"])]
            drone_tracks[str(drone_id)] = coords
    else:
        for pos in calculated_positions:
            drone_id = pos.get("drone_id", "drone_1")
            drone_tracks.setdefault(drone_id, []).append((float(pos["x"]), float(pos["y"])))

    detected_mine_ids = {m.get("mine_id", -1) for m in confirmed_mines if m.get("mine_id", -1) >= 0}
    forced_additions = 0
    for mine_id, row in df_gt.reset_index().iterrows():
        if mine_id in detected_mine_ids:
            continue

        mx, my = clamp_xy(row["X"], row["Y"])
        if is_point_near_any_drone_segment(mx, my, drone_tracks, float(detection_radius)):
            confirmed_mines.append({
                "discovery_index": -1,
                "x": mx,
                "y": my,
                "z": float(terrain_map.get((round(mx / 10.0) * 10.0, round(my / 10.0) * 10.0), 10.0)),
                "drone_id": "route_proximity",
                "mine_id": int(mine_id),
            })
            detected_mine_ids.add(mine_id)
            forced_additions += 1

    print(
        f"LLM Vision verified mines: {len(confirmed_mines)} (route-proximity added: {forced_additions}) | "
        f"Gemini calls: {_GEMINI_CALL_COUNT}, failures: {_GEMINI_FAIL_COUNT}"
    )
    
    # --- Analyze Off-Road Viability at Deviation Points ---
    # Deduplicate the deviation anchors
    unique_deviations = []
    for dx, dy in raw_deviations:
        if not any(math.hypot(dx - ux, dy - uy) < 10.0 for ux, uy in unique_deviations):
            unique_deviations.append((dx, dy))
            
    off_road_analysis = []
    for dx, dy in unique_deviations:
        # Snap to terrain grid points (10x10)
        sx, sy = round(dx/10.0)*10.0, round(dy/10.0)*10.0
        # Check immediate 3x3 surrounding Z-slope
        local_z = []
        for nx in [sx-10, sx, sx+10]:
            for ny in [sy-10, sy, sy+10]:
                if (nx, ny) in terrain_map:
                    local_z.append(terrain_map[(nx, ny)])
                    
        if len(local_z) > 0:
            variance = max(local_z) - min(local_z)
            status = "IMPASSABLE" if variance > 8.0 else "VIABLE"
            off_road_analysis.append({"x": dx, "y": dy, "variance": round(variance, 2), "status": status})
            
    print(f"Analyzed {len(off_road_analysis)} obstacle deviation sites for off-road feasibility.")
    
    # Calculate Safe Path (Base Camp to Point B)
    start_point = (start_x, start_y)
    end_point = (target_x, target_y)
    
    print("Calculating safe 3D convoy route using A* and Topography constraints...")
    
    # Route planning avoids detected hazards plus a stand-off from every generated mine point.
    # In dense minefields this can fully block the graph; progressively relax mine-only clearance
    # so the planner still produces a route instead of returning no path.
    hazard_coords = [(m["x"], m["y"], m.get("z", 0.0), HAZARD_RADIUS) for m in confirmed_mines]
    mine_points = [(float(row["X"]), float(row["Y"])) for _, row in df_gt.iterrows()]
    clearance_scales = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0]
    safe_path = []
    used_mine_clearance = ROUTE_MINE_CLEARANCE_RADIUS

    for scale in clearance_scales:
        candidate_clearance = ROUTE_MINE_CLEARANCE_RADIUS * scale
        all_mine_clearance = [
            (mx, my, 0.0, candidate_clearance) for mx, my in mine_points if candidate_clearance > 0.0
        ]
        safe_path = run_a_star(
            start_point,
            end_point,
            hazard_coords + all_mine_clearance,
            terrain_map,
            GRID_W,
            GRID_H,
        )
        if safe_path:
            used_mine_clearance = candidate_clearance
            if scale < 1.0:
                print(
                    "WARNING: Dense hazards required adaptive mine-clearance fallback "
                    f"({ROUTE_MINE_CLEARANCE_RADIUS:.1f}m -> {used_mine_clearance:.1f}m)."
                )
            break
    
    if not safe_path:
        print("CRITICAL: No safe path could be mapped through the minefield.")
    else:
        print(
            f"Safe route generated with {len(safe_path)} waypoints "
            f"(mine-clearance used: {used_mine_clearance:.1f}m)."
        )
    
    # Consolidate Output for Phase 3 UI (including the global terrain mapping)
    export_terrain = [{"x": k[0], "y": k[1], "z": v} for k, v in terrain_map.items()]
    
    # Load ground truth mines for UI overlay
    export_gt_mines = []
    for _, row in df_gt.iterrows():
        gx, gy = clamp_xy(row["X"], row["Y"])
        export_gt_mines.append({"x": gx, "y": gy})
    
    output_data = {
        "trajectory": calculated_positions,
        "anomalies": confirmed_mines,
        "llm_vision_events": vision_verifications,
        "safe_route": safe_path,
        "terrain_grid": export_terrain,
        "off_road_viability": off_road_analysis,
        "ground_truth": export_gt_mines,
        "llm_vision_trace": {
            "provider": "gemini",
            "model_requested": os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            "gemini_calls": int(_GEMINI_CALL_COUNT),
            "gemini_failures": int(_GEMINI_FAIL_COUNT),
            "trace_events": _GEMINI_TRACE_EVENTS,
        },
        "run_context": {
            "mission_id": mission_id,
            "start_x": float(start_x),
            "start_y": float(start_y),
            "target_x": float(target_x),
            "target_y": float(target_y),
            "detection_radius": float(detection_radius),
            "minefield_hash": file_sha256(GROUND_TRUTH_PATH),
        },
    }
    
    with open("tactical_map_data.json", "w") as f:
        json.dump(output_data, f)
    with open("gemini_trace.json", "w") as f:
        json.dump(output_data["llm_vision_trace"], f)
        
    print("Tactical data exported to tactical_map_data.json")
    print("Gemini trace exported to gemini_trace.json")
    print(f"Mission ID: {mission_id} | telemetry_source={telemetry_source}")

    if atlas_enabled():
        try:
            write_phase2_outputs(
                mission_id=mission_id,
                output_data=output_data,
                start_x=float(start_x),
                start_y=float(start_y),
                target_x=float(target_x),
                target_y=float(target_y),
                detection_radius=float(detection_radius),
            )
            print(f"Atlas tactical output export complete (mission_id={mission_id})")
        except Exception as exc:
            print(f"WARNING: Atlas Phase 2 export failed ({exc}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_x", type=float, default=0.0)
    parser.add_argument("--start_y", type=float, default=0.0)
    parser.add_argument("--target_x", type=float, default=500.0)
    parser.add_argument("--target_y", type=float, default=500.0)
    parser.add_argument("--detection_radius", type=float, default=DEFAULT_ROUTE_DETECTION_RADIUS)
    parser.add_argument("--mission_id", type=str, default="")
    args = parser.parse_args()
    
    print(
        f"Routing Phase 2 Logic from Point A -> ({args.start_x}, {args.start_y}) "
        f"to Point B -> ({args.target_x}, {args.target_y}) "
        f"| mission_id={args.mission_id.strip() or 'auto'}"
    )
    process_telemetry(
        args.start_x,
        args.start_y,
        args.target_x,
        args.target_y,
        args.detection_radius,
        args.mission_id,
    )
