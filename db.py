import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.server_api import ServerApi

_CLIENT: Optional[MongoClient] = None


def get_phase1_telemetry_mode() -> str:
    """
    Controls Phase 1 telemetry ingestion behavior.

    - summary (default): write mission metadata only, skip heavy telemetry upload.
    - full: write all telemetry rows into telemetry_raw.
    """
    mode = str(os.environ.get("ATLAS_PHASE1_TELEMETRY_MODE", "summary")).strip().lower()
    return mode if mode in {"summary", "full"} else "summary"


def _load_dotenv_if_present(dotenv_path: str = ".env") -> None:
    """Load environment variables without adding new dependencies."""
    if not os.path.exists(dotenv_path):
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
    except Exception:
        # Keep runtime resilient if local dotenv parsing fails.
        return


def atlas_enabled() -> bool:
    _load_dotenv_if_present()
    return bool(os.environ.get("MONGODB_URI", "").strip())


def get_client() -> MongoClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    _load_dotenv_if_present()
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI missing. Atlas integration disabled.")

    app_name = os.environ.get("MONGODB_APP_NAME", "drone-recon-poc").strip() or "drone-recon-poc"
    _CLIENT = MongoClient(
        uri,
        appname=app_name,
        server_api=ServerApi("1"),
        retryWrites=True,
        maxPoolSize=30,
        minPoolSize=2,
    )
    return _CLIENT


def get_db():
    db_name = os.environ.get("MONGODB_DB", "aero_hacks").strip() or "aero_hacks"
    return get_client()[db_name]


def _safe_create_collection(db, name: str, **kwargs) -> None:
    existing = set(db.list_collection_names())
    if name in existing:
        return
    try:
        db.create_collection(name, **kwargs)
    except Exception:
        # Collection may already exist, or option unsupported on local test DB.
        return


def ensure_atlas_schema() -> Dict[str, bool]:
    """
    Ensure key collections and indexes exist.
    Returns a status map for observability in logs/UI.
    """
    status = {
        "collections_ready": False,
        "indexes_ready": False,
        "search_index_requested": False,
        "vector_index_requested": False,
    }
    db = get_db()

    # Native timeseries for telemetry streams.
    _safe_create_collection(
        db,
        "telemetry_raw",
        timeseries={
            "timeField": "timestamp",
            "metaField": "meta",
            "granularity": "seconds",
        },
        expireAfterSeconds=60 * 60 * 24 * 30,  # 30 days retention
    )
    _safe_create_collection(db, "missions")
    _safe_create_collection(db, "hazards")
    _safe_create_collection(db, "routes")
    _safe_create_collection(db, "terrain_cells")
    _safe_create_collection(db, "vision_events")
    _safe_create_collection(db, "tactical_outputs")
    _safe_create_collection(db, "alerts")
    status["collections_ready"] = True

    db.telemetry_raw.create_index([("mission_id", ASCENDING), ("elapsed_ms", ASCENDING)])
    db.telemetry_raw.create_index([("timestamp", DESCENDING)])
    db.missions.create_index([("created_at", DESCENDING)])
    db.missions.create_index([("status", ASCENDING)])
    db.hazards.create_index([("mission_id", ASCENDING), ("created_at", DESCENDING)])
    db.hazards.create_index([("location", "2dsphere")])
    db.routes.create_index([("mission_id", ASCENDING), ("created_at", DESCENDING)])
    db.routes.create_index([("geometry", "2dsphere")])
    db.terrain_cells.create_index([("mission_id", ASCENDING), ("x", ASCENDING), ("y", ASCENDING)])
    db.vision_events.create_index([("mission_id", ASCENDING), ("discovery_index", ASCENDING)])
    db.tactical_outputs.create_index([("mission_id", ASCENDING), ("created_at", DESCENDING)])
    db.alerts.create_index([("mission_id", ASCENDING), ("created_at", DESCENDING)])
    status["indexes_ready"] = True

    _ensure_search_indexes(db, status)
    return status


def _ensure_search_indexes(db, status: Dict[str, bool]) -> None:
    """
    Create Atlas Search and Vector Search indexes if supported.
    Works only on Atlas clusters with Search enabled.
    """
    # Full-text index for mission + vision diagnostics
    search_cmd = {
        "createSearchIndexes": "vision_events",
        "indexes": [
            {
                "name": "vision_text_search",
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "decision_label": {"type": "string"},
                            "status": {"type": "string"},
                            "source": {"type": "string"},
                            "model": {"type": "string"},
                            "error": {"type": "string"},
                            "frame_ref": {"type": "string"},
                        },
                    }
                },
            }
        ],
    }

    # Vector index for similarity search on optional embeddings
    vector_cmd = {
        "createSearchIndexes": "vision_events",
        "indexes": [
            {
                "name": "vision_embedding_vector",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 768,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "mission_id"},
                    ]
                },
            }
        ],
    }

    try:
        db.command(search_cmd)
        status["search_index_requested"] = True
    except Exception as exc:
        status["search_index_requested"] = "already exists" in str(exc).lower()

    try:
        db.command(vector_cmd)
        status["vector_index_requested"] = True
    except Exception as exc:
        status["vector_index_requested"] = "already exists" in str(exc).lower()


def load_telemetry_df_for_mission(mission_id: str):
    import pandas as pd

    db = get_db()
    cursor = db.telemetry_raw.find({"mission_id": mission_id}, {"_id": 0}).sort(
        [("elapsed_ms", ASCENDING), ("drone_id", ASCENDING)]
    )
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()

    rows = []
    for d in docs:
        pos = d.get("pos", {})
        rows.append(
            {
                "Drone_ID": d.get("drone_id", "drone_1"),
                "Elapsed_ms": int(d.get("elapsed_ms", 0)),
                "Pitch": float(d.get("pitch", 0.0)),
                "Roll": float(d.get("roll", 0.0)),
                "Yaw": float(d.get("yaw", 0.0)),
                "Velocity_ms": float(d.get("velocity_ms", 0.0)),
                "Altitude_m": float(d.get("altitude_m", 0.0)),
                "GPR_Score": float(d.get("gpr_score", 0.0)),
                "Deviation_Flag": int(d.get("deviation_flag", 0)),
                "Detected_Mine_ID": int(d.get("detected_mine_id", -1)),
                "Camera_Frame_URL": str(d.get("camera_frame_url", "")),
                "Pos_X": float(pos.get("x", 0.0)),
                "Pos_Y": float(pos.get("y", 0.0)),
                "Pos_Z": float(pos.get("z", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def write_phase1_mission(
    df,
    mission_id: str,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    detection_radius: float,
    num_drones: int,
    minefield_hash: Optional[str],
) -> None:
    db = get_db()
    created_at = datetime.now(timezone.utc)
    telemetry_mode = get_phase1_telemetry_mode()

    db.missions.update_one(
        {"_id": mission_id},
        {
            "$set": {
                "created_at": created_at,
                "phase": "phase1_completed",
                "status": "ready_for_phase2",
                "start": {"x": float(start_x), "y": float(start_y)},
                "target": {"x": float(target_x), "y": float(target_y)},
                "detection_radius": float(detection_radius),
                "num_drones": int(num_drones),
                "minefield_hash": str(minefield_hash or ""),
                "telemetry_count": int(len(df)),
                "telemetry_storage_mode": telemetry_mode,
                "updated_at": created_at,
            }
        },
        upsert=True,
    )

    if telemetry_mode != "full":
        return

    records: List[Dict] = []
    for row in df.itertuples(index=False):
        elapsed_ms = int(getattr(row, "Elapsed_ms", 0))
        ts = created_at + timedelta(milliseconds=elapsed_ms)
        drone_id = str(getattr(row, "Drone_ID", "drone_1"))
        records.append(
            {
                "mission_id": mission_id,
                "timestamp": ts,
                "meta": {"mission_id": mission_id, "drone_id": drone_id},
                "drone_id": drone_id,
                "elapsed_ms": elapsed_ms,
                "pitch": float(getattr(row, "Pitch", 0.0)),
                "roll": float(getattr(row, "Roll", 0.0)),
                "yaw": float(getattr(row, "Yaw", 0.0)),
                "velocity_ms": float(getattr(row, "Velocity_ms", 0.0)),
                "altitude_m": float(getattr(row, "Altitude_m", 0.0)),
                "gpr_score": float(getattr(row, "GPR_Score", 0.0)),
                "deviation_flag": int(getattr(row, "Deviation_Flag", 0)),
                "detected_mine_id": int(getattr(row, "Detected_Mine_ID", -1)),
                "camera_frame_url": str(getattr(row, "Camera_Frame_URL", "")),
                "pos": {
                    "x": float(getattr(row, "Pos_X", 0.0)),
                    "y": float(getattr(row, "Pos_Y", 0.0)),
                    "z": float(getattr(row, "Pos_Z", 0.0)),
                },
            }
        )

    if not records:
        return

    # Re-runs for a mission replace telemetry for deterministic replay.
    db.telemetry_raw.delete_many({"mission_id": mission_id})
    for i in range(0, len(records), 5000):
        db.telemetry_raw.insert_many(records[i : i + 5000], ordered=False)


def write_phase2_outputs(
    mission_id: str,
    output_data: Dict,
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    detection_radius: float,
) -> None:
    db = get_db()
    now = datetime.now(timezone.utc)

    db.hazards.delete_many({"mission_id": mission_id})
    db.routes.delete_many({"mission_id": mission_id})
    db.terrain_cells.delete_many({"mission_id": mission_id})
    db.vision_events.delete_many({"mission_id": mission_id})
    db.tactical_outputs.delete_many({"mission_id": mission_id})

    hazards = []
    for h in output_data.get("anomalies", []):
        x = float(h.get("x", 0.0))
        y = float(h.get("y", 0.0))
        hazards.append(
            {
                "mission_id": mission_id,
                "created_at": now,
                "discovery_index": int(h.get("discovery_index", -1)),
                "drone_id": str(h.get("drone_id", "unknown")),
                "mine_id": int(h.get("mine_id", -1)),
                "z": float(h.get("z", 0.0)),
                "location": {"type": "Point", "coordinates": [x, y]},
            }
        )
    if hazards:
        db.hazards.insert_many(hazards, ordered=False)

    route = output_data.get("safe_route", [])
    if route:
        line_coords = [[float(p.get("x", 0.0)), float(p.get("y", 0.0))] for p in route]
        db.routes.insert_one(
            {
                "mission_id": mission_id,
                "created_at": now,
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "safe_route": route,
            }
        )

    terrain_rows = []
    for t in output_data.get("terrain_grid", []):
        terrain_rows.append(
            {
                "mission_id": mission_id,
                "created_at": now,
                "x": float(t.get("x", 0.0)),
                "y": float(t.get("y", 0.0)),
                "z": float(t.get("z", 0.0)),
            }
        )
    if terrain_rows:
        for i in range(0, len(terrain_rows), 2000):
            db.terrain_cells.insert_many(terrain_rows[i : i + 2000], ordered=False)

    vision_events = []
    for v in output_data.get("llm_vision_events", []):
        event = dict(v)
        event["mission_id"] = mission_id
        event["created_at"] = now
        vision_events.append(event)
    if vision_events:
        db.vision_events.insert_many(vision_events, ordered=False)

    db.tactical_outputs.insert_one(
        {
            "mission_id": mission_id,
            "created_at": now,
            "start": {"x": float(start_x), "y": float(start_y)},
            "target": {"x": float(target_x), "y": float(target_y)},
            "detection_radius": float(detection_radius),
            "payload": output_data,
        }
    )

    trace = output_data.get("llm_vision_trace", {})
    summary = {
        "hazards": int(len(output_data.get("anomalies", []))),
        "route_waypoints": int(len(output_data.get("safe_route", []))),
        "gemini_calls": int(trace.get("gemini_calls", 0)) if isinstance(trace, dict) else 0,
        "gemini_failures": int(trace.get("gemini_failures", 0)) if isinstance(trace, dict) else 0,
    }
    db.missions.update_one(
        {"_id": mission_id},
        {
            "$set": {
                "status": "phase2_completed",
                "phase": "phase2_completed",
                "updated_at": now,
                "start": {"x": float(start_x), "y": float(start_y)},
                "target": {"x": float(target_x), "y": float(target_y)},
                "detection_radius": float(detection_radius),
                "summary": summary,
            }
        },
        upsert=True,
    )


def load_latest_tactical_output(mission_id: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    db = get_db()
    query = {"mission_id": mission_id} if mission_id else {}
    doc = db.tactical_outputs.find_one(query, sort=[("created_at", DESCENDING)])
    if not doc:
        return None, None
    payload = doc.get("payload")
    return payload if isinstance(payload, dict) else None, str(doc.get("mission_id"))


def append_alert_for_change_event(change_doc: Dict) -> None:
    db = get_db()
    mission_id = str(change_doc.get("fullDocument", {}).get("mission_id", "unknown"))
    db.alerts.insert_one(
        {
            "mission_id": mission_id,
            "created_at": datetime.now(timezone.utc),
            "kind": "telemetry_change",
            "change_summary": {
                "operationType": change_doc.get("operationType", ""),
                "ns": change_doc.get("ns", {}),
            },
        }
    )


def run_change_stream_once(max_events: int = 10, max_seconds: int = 5) -> int:
    """
    Consume a bounded number of telemetry change events.
    Useful for demos and lightweight local workers.
    """
    db = get_db()
    consumed = 0
    start = datetime.now(timezone.utc)
    with db.telemetry_raw.watch(full_document="updateLookup", max_await_time_ms=800) as stream:
        while consumed < int(max_events):
            event = stream.try_next()
            if event is None:
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                if elapsed >= float(max_seconds):
                    break
                continue
            append_alert_for_change_event(event)
            consumed += 1
    return consumed
