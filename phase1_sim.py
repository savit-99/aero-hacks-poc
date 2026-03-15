import pandas as pd
import numpy as np
import math
import hashlib
import argparse
import os
import json
import uuid

from db import atlas_enabled, ensure_atlas_schema, get_phase1_telemetry_mode, write_phase1_mission

# Base configurations
GRID_W = 500
GRID_H = 500
V_BASE = 5.0  # m/s
DT = 0.1      # 10Hz
DEFAULT_NUM_MINES = 40
DEFAULT_NUM_DRONES = 2
OUTBOUND_OFFSET = 16.0
RETURN_OFFSET = 70.0
ZIGZAG_AMPLITUDE = 18.0
ZIGZAG_SPACING = 10.0
DEFAULT_DETECTION_RADIUS = 20.0
MINE_SCAN_RADIUS = 10.0
MINE_SCAN_POINTS = 24
GROUND_TRUTH_PATH = "ground_truth_mines.csv"
TELEMETRY_META_PATH = "telemetry_metadata.json"

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
    """Clamp coordinates to the 500x500 tactical map bounds."""
    return (
        float(min(max(x, 0.0), GRID_W)),
        float(min(max(y, 0.0), GRID_H)),
    )

def build_leg_waypoints(start, end, base_offset, zigzag_amplitude=ZIGZAG_AMPLITUDE, spacing=ZIGZAG_SPACING):
    """
    Build a centerline-relative smooth-curve leg between start/end.
    base_offset controls how far the leg is from the center A<->B vector.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux  # left-normal

    waypoints = []
    num_samples = max(2, int(dist / max(spacing, 1.0)) + 1)
    oscillations = max(3, int(dist / 80.0))

    for i in range(num_samples + 1):
        progress = min(i / num_samples, 1.0)
        s = progress * dist
        # Smooth sine-wave sweep with zero-offset at leg start/end.
        theta = 2.0 * math.pi * oscillations * progress
        lateral = base_offset + zigzag_amplitude * math.sin(theta)
        wx = start[0] + ux * s + px * lateral
        wy = start[1] + uy * s + py * lateral
        cx, cy = clamp_xy(wx, wy)
        waypoints.append(np.array([cx, cy]))

    # ensure leg terminates near the requested endpoint while preserving offset side
    wx_end = end[0] + px * base_offset
    wy_end = end[1] + py * base_offset
    cx_end, cy_end = clamp_xy(wx_end, wy_end)
    waypoints.append(np.array([cx_end, cy_end]))
    return waypoints

def generate_dual_leg_waypoints(start, end, side_sign):
    """
    side_sign: -1 for left drone, +1 for right drone relative to A->B centerline.
    Outbound runs closer to center; return runs further from center.
    """
    outbound = build_leg_waypoints(start, end, base_offset=side_sign * OUTBOUND_OFFSET)
    inbound = build_leg_waypoints(end, start, base_offset=side_sign * RETURN_OFFSET)
    start_pt = np.array(clamp_xy(start[0], start[1]))
    return [start_pt] + outbound + inbound

def get_elevation(x, y):
    """
    Simulates Computer Vision Depth-Sensing / Terrain Elevation.
    Explicitly injects random variance mixed with rolling hills to ensure all Z-ranges (0-50m) are hit dynamically.
    """
    import random
    
    # Base undulating terrain
    hill_x = math.sin(x / 50.0) * 10.0
    hill_y = math.cos(y / 70.0) * 10.0
    
    # Steep central ridge
    ridge = math.exp(-((x - GRID_W/2)**2 + (y - GRID_H/2)**2) / 20000.0) * 20.0
    
    # Random terrain jitter so no values are identical or perfectly flat
    jitter = random.uniform(0.0, 15.0)
    
    # Combine signals and force base layer above 5m minimum
    raw_z = hill_x + hill_y + ridge + jitter + 15.0 
    
    # Clamp safely within the UI's 0-50m heatmap scale
    return max(1.0, min(50.0, raw_z))

def generate_mines(num_mines=DEFAULT_NUM_MINES, seed=None):
    """Generates random coordinates for simulated mines."""
    safe_num_mines = max(1, int(num_mines))
    rng = np.random.default_rng(seed)
    return rng.random((safe_num_mines, 2)) * [GRID_W, GRID_H]

def save_mines_to_csv(mines):
    pd.DataFrame(mines, columns=["X", "Y"]).to_csv(GROUND_TRUTH_PATH, index=False)

def load_mines_from_csv():
    if not os.path.exists(GROUND_TRUTH_PATH):
        return None
    try:
        df = pd.read_csv(GROUND_TRUTH_PATH)
        if not {"X", "Y"}.issubset(df.columns) or df.empty:
            return None
        mines = df[["X", "Y"]].to_numpy(dtype=float)
        return mines
    except Exception:
        return None

def min_distance_to_segment(points, seg_start, seg_end):
    """Return minimum distance from any point to segment [seg_start, seg_end]."""
    if len(points) == 0:
        return 999.0

    seg_vec = seg_end - seg_start
    seg_len_sq = float(np.dot(seg_vec, seg_vec))
    if seg_len_sq < 1e-9:
        return float(np.min(np.linalg.norm(points - seg_start, axis=1)))

    rel = points - seg_start
    t = np.clip((rel @ seg_vec) / seg_len_sq, 0.0, 1.0)
    projections = seg_start + t[:, None] * seg_vec
    dists = np.linalg.norm(points - projections, axis=1)
    return float(np.min(dists))

def nearest_distance_index_to_segment(points, seg_start, seg_end):
    """Return (min_distance, nearest_point_index) to segment [seg_start, seg_end]."""
    if len(points) == 0:
        return 999.0, -1

    seg_vec = seg_end - seg_start
    seg_len_sq = float(np.dot(seg_vec, seg_vec))
    if seg_len_sq < 1e-9:
        dists = np.linalg.norm(points - seg_start, axis=1)
        idx = int(np.argmin(dists))
        return float(dists[idx]), idx

    rel = points - seg_start
    t = np.clip((rel @ seg_vec) / seg_len_sq, 0.0, 1.0)
    projections = seg_start + t[:, None] * seg_vec
    dists = np.linalg.norm(points - projections, axis=1)
    idx = int(np.argmin(dists))
    return float(dists[idx]), idx

def build_circle_scan_waypoints(center_x, center_y, start_x, start_y, radius=MINE_SCAN_RADIUS, num_points=MINE_SCAN_POINTS):
    """Build circular scan waypoints around a mine center."""
    start_angle = math.atan2(start_y - center_y, start_x - center_x)
    circle_waypoints = []
    for i in range(1, num_points + 1):
        theta = start_angle + (2.0 * math.pi * i / num_points)
        wx = center_x + radius * math.cos(theta)
        wy = center_y + radius * math.sin(theta)
        cx, cy = clamp_xy(wx, wy)
        circle_waypoints.append(np.array([cx, cy], dtype=float))
    return circle_waypoints

def simulate_drone_track(drone_id, waypoints, mines, detection_radius=DEFAULT_DETECTION_RADIUS):
    """Simulate one drone moving through predefined zigzag waypoints."""
    telemetry = []
    elapsed_ms = 0
    current_pos = np.array(waypoints[0], dtype=float)
    wp_idx = 1
    scanned_mine_ids = set()

    while wp_idx < len(waypoints):
        target_pos = np.array(waypoints[wp_idx], dtype=float)
        target_pos = np.array(clamp_xy(target_pos[0], target_pos[1]), dtype=float)
        dist = np.linalg.norm(target_pos - current_pos)

        if dist < (V_BASE * DT):
            current_pos = target_pos
            wp_idx += 1
            continue

        direction = (target_pos - current_pos) / max(dist, 1e-6)
        next_pos = current_pos + direction * (V_BASE * DT)
        next_pos = np.array(clamp_xy(next_pos[0], next_pos[1]), dtype=float)

        prev_pos = current_pos.copy()
        movement_vec = next_pos - prev_pos
        move_dist = np.linalg.norm(movement_vec)
        true_yaw = math.atan2(movement_vec[1], movement_vec[0]) if move_dist > 1e-3 else 0.0

        yaw_noise = np.random.normal(0, 0.03)
        vel_noise = np.random.normal(0, 0.15)
        noisy_yaw = true_yaw + yaw_noise
        noisy_vel = max(0.0, V_BASE + vel_noise)

        pitch = np.random.normal(0, 0.02)
        roll = np.random.normal(0, 0.02)
        altitude = get_elevation(next_pos[0], next_pos[1]) + np.random.normal(0, 0.4)

        # Detect against full motion segment so mines near route are not missed between timesteps.
        min_dist, nearest_mine_idx = nearest_distance_index_to_segment(mines, prev_pos, next_pos)
        current_pos = next_pos
        is_deviating = 0
        if min_dist <= float(detection_radius):
            gpr = np.random.uniform(0.96, 1.0)
            detected_mine_id = nearest_mine_idx

            # First time this mine is detected by this drone: insert a 10m circular scan,
            # then continue along the original route waypoints.
            if detected_mine_id >= 0 and detected_mine_id not in scanned_mine_ids:
                scanned_mine_ids.add(detected_mine_id)
                mine_x, mine_y = mines[detected_mine_id]
                circle_scan = build_circle_scan_waypoints(mine_x, mine_y, current_pos[0], current_pos[1])
                waypoints[wp_idx:wp_idx] = circle_scan
                is_deviating = 1
        else:
            gpr = np.random.uniform(0.0, 0.25)
            detected_mine_id = -1

        url = f"http://internal-drone-net/cam/{drone_id}/frame_{elapsed_ms:08d}.jpg"

        telemetry.append({
            "Drone_ID": drone_id,
            "Elapsed_ms": int(elapsed_ms),
            "Pitch": round(float(pitch), 4),
            "Roll": round(float(roll), 4),
            "Yaw": round(float(noisy_yaw), 4),
            "Velocity_ms": round(float(noisy_vel), 4),
            "Altitude_m": round(float(altitude), 2),
            "GPR_Score": round(float(gpr), 4),
            "Deviation_Flag": is_deviating,
            "Detected_Mine_ID": int(detected_mine_id),
            "Camera_Frame_URL": url,
            "Pos_X": round(float(current_pos[0]), 4),
            "Pos_Y": round(float(current_pos[1]), 4),
            "Pos_Z": round(float(altitude), 2),
        })

        elapsed_ms += int(DT * 1000)

    return pd.DataFrame(telemetry)

def simulate_data(
    start_x=0.0,
    start_y=0.0,
    target_x=500.0,
    target_y=500.0,
    num_mines=DEFAULT_NUM_MINES,
    num_drones=DEFAULT_NUM_DRONES,
    detection_radius=DEFAULT_DETECTION_RADIUS,
    use_existing_mines=False,
    mine_seed=None,
):
    """Simulate default 2-drone centerline-offset zigzag scans from A->B and B->A."""
    start = clamp_xy(start_x, start_y)
    target = clamp_xy(target_x, target_y)
    if use_existing_mines:
        mines = load_mines_from_csv()
        if mines is None:
            mines = generate_mines(num_mines, seed=mine_seed)
    else:
        mines = generate_mines(num_mines, seed=mine_seed)

    active_drones = max(1, min(2, int(num_drones)))
    drone_sides = [-1, 1][:active_drones]
    drone_frames = []
    for idx, side_sign in enumerate(drone_sides, start=1):
        waypoints = generate_dual_leg_waypoints(start, target, side_sign=side_sign)
        drone_df = simulate_drone_track(
            f"drone_{idx}",
            waypoints,
            mines,
            detection_radius=detection_radius,
        )
        drone_frames.append(drone_df)

    df = pd.concat(drone_frames, ignore_index=True)
    df = df.sort_values(["Elapsed_ms", "Drone_ID"]).reset_index(drop=True)
    
    # Save coordinates of ground truth mines for later diagnostic evaluation 
    save_mines_to_csv(mines)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_x", type=float, default=0.0)
    parser.add_argument("--start_y", type=float, default=0.0)
    parser.add_argument("--target_x", type=float, default=500.0)
    parser.add_argument("--target_y", type=float, default=500.0)
    parser.add_argument("--num_mines", type=int, default=DEFAULT_NUM_MINES)
    parser.add_argument("--num_drones", type=int, default=DEFAULT_NUM_DRONES)
    parser.add_argument("--detection_radius", type=float, default=DEFAULT_DETECTION_RADIUS)
    parser.add_argument("--mine_seed", type=int, default=None)
    parser.add_argument("--use_existing_mines", action="store_true")
    parser.add_argument("--generate_mines_only", action="store_true")
    parser.add_argument("--mission_id", type=str, default="")
    args = parser.parse_args()
    
    safe_num_mines = max(1, args.num_mines)
    safe_num_drones = max(1, min(2, args.num_drones))
    safe_detection_radius = max(1.0, float(args.detection_radius))

    if args.generate_mines_only:
        mines = generate_mines(safe_num_mines, seed=args.mine_seed)
        save_mines_to_csv(mines)
        print(
            f"Generated minefield with {len(mines)} simulated mines "
            f"(seed={args.mine_seed if args.mine_seed is not None else 'random'})"
        )
        raise SystemExit(0)

    print(
        f"Initializing Phase 1 Simulator. Vectoring Point A -> ({args.start_x}, {args.start_y}) "
        f"to Point B -> ({args.target_x}, {args.target_y}) "
        f"with {safe_num_mines} simulated mines using {safe_num_drones} drone(s), "
        f"detection radius {safe_detection_radius:.1f}m"
    )
    mission_id = args.mission_id.strip() or str(uuid.uuid4())
    df = simulate_data(
        args.start_x,
        args.start_y,
        args.target_x,
        args.target_y,
        safe_num_mines,
        safe_num_drones,
        detection_radius=safe_detection_radius,
        use_existing_mines=args.use_existing_mines,
        mine_seed=args.mine_seed,
    )
    output_path = "telemetry_log.csv"
    df.to_csv(output_path, index=False)

    telemetry_meta = {
        "mission_id": mission_id,
        "start_x": float(args.start_x),
        "start_y": float(args.start_y),
        "target_x": float(args.target_x),
        "target_y": float(args.target_y),
        "detection_radius": float(safe_detection_radius),
        "num_drones": int(safe_num_drones),
        "minefield_hash": file_sha256(GROUND_TRUTH_PATH),
    }
    with open(TELEMETRY_META_PATH, "w") as f:
        json.dump(telemetry_meta, f)

    if atlas_enabled():
        try:
            telemetry_mode = get_phase1_telemetry_mode()
            status = {
                "collections_ready": "skipped",
                "indexes_ready": "skipped",
                "search_index_requested": "skipped",
                "vector_index_requested": "skipped",
            }
            # Full telemetry ingestion relies on Atlas-specific collection/index setup.
            if telemetry_mode == "full":
                status = ensure_atlas_schema()
            write_phase1_mission(
                df=df,
                mission_id=mission_id,
                start_x=float(args.start_x),
                start_y=float(args.start_y),
                target_x=float(args.target_x),
                target_y=float(args.target_y),
                detection_radius=float(safe_detection_radius),
                num_drones=int(safe_num_drones),
                minefield_hash=telemetry_meta.get("minefield_hash"),
            )
            print(f"Atlas mission ingestion complete (mission_id={mission_id})")
            print(f"Atlas telemetry mode: {telemetry_mode}")
            print(
                "Atlas setup status: "
                f"collections={status.get('collections_ready')}, "
                f"indexes={status.get('indexes_ready')}, "
                f"search_index={status.get('search_index_requested')}, "
                f"vector_index={status.get('vector_index_requested')}"
            )
        except Exception as exc:
            print(f"WARNING: Atlas Phase 1 export failed ({exc}). Continuing with local artifacts.")

    print(f"Successfully generated {len(df)} telemetry logs representing dynamic flight.")
    print(f"Secured Data saved to {output_path}")
    print(f"Mission ID: {mission_id}")
