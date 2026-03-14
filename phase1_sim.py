import pandas as pd
import numpy as np
import math
import hashlib
import argparse

# Base configurations
GRID_W = 500
GRID_H = 500
SWEEP_SPACING = 50
V_BASE = 5.0  # m/s
DT = 0.1      # 10Hz
NUM_MINES = 25

def generate_corridor_waypoints(start, end, width=80, spacing=20):
    """Generates a zig-zag corridor sweep to maximize surface area along the route to Point B."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    
    waypoints_local = []
    x = 0
    direction = 1
    while x <= dist:
        y_max = width / 2
        y_min = -width / 2
        if direction == 1:
            waypoints_local.append((x, y_min))
            waypoints_local.append((x, y_max))
        else:
            waypoints_local.append((x, y_max))
            waypoints_local.append((x, y_min))
        x += spacing
        direction *= -1
    
    waypoints_local.append((dist, 0)) # End point
    
    # Rotate and translate back to global coords
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    waypoints = []
    for lx, ly in waypoints_local:
        wx = start[0] + lx * cos_a - ly * sin_a
        wy = start[1] + lx * sin_a + ly * cos_a
        waypoints.append(np.array([wx, wy]))
        
    return waypoints

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

def generate_mines():
    """Generates random coordinates for simulated mines."""
    np.random.seed(42) # Reproducible mines
    return np.random.rand(NUM_MINES, 2) * [GRID_W, GRID_H]

def simulate_data(target_x=500.0, target_y=500.0):
    """Simulates drone flight data to Point B and back, including real-time obstacles."""
    start = (0.0, 0.0)
    target = (target_x, target_y)
    
    # Outward sweep
    outward_wp = generate_corridor_waypoints(start, target)
    # Direct return home
    return_wp = [np.array(target), np.array(start)]
    waypoints = outward_wp + return_wp
    
    mines = generate_mines()
    
    current_pos = np.array(waypoints[0])
    wp_idx = 1
    
    elapsed_ms = 0
    telemetry = []
    
    while wp_idx < len(waypoints):
        target_pos = np.array(waypoints[wp_idx])
        dist = np.linalg.norm(target_pos - current_pos)
        
        # Advance waypoint if reached
        if dist < (V_BASE * DT):
            current_pos = target_pos
            wp_idx += 1
            if wp_idx >= len(waypoints):
                break
            continue
            
        # Real-time State: Obstacle Avoidance (Maintain >5m from mines)
        dists_to_mines = np.linalg.norm(mines - next_pos, axis=1)
        min_dist = np.min(dists_to_mines)
        closest_mine_idx = np.argmin(dists_to_mines)
        
        is_deviating = 0
        if min_dist < 5.0:
            # Dodge! Slide along the 5m exclusion radius boundary
            mine_pos = mines[closest_mine_idx]
            escape_vec = next_pos - mine_pos
            escape_dist = max(np.linalg.norm(escape_vec), 0.001)
            next_pos = mine_pos + (escape_vec / escape_dist) * 5.0
            is_deviating = 1
            min_dist = 5.0 # Reset min dist for GPR logic below
            
        # Calculate actual movement after potential dodging
        movement_vec = next_pos - current_pos
        move_dist = np.linalg.norm(movement_vec)
        true_yaw = math.atan2(movement_vec[1], movement_vec[0]) if move_dist > 0.001 else 0.0
        
        current_pos = next_pos
        
        # Inject realistic INS noise
        yaw_noise = np.random.normal(0, 0.05) # ~2.8 degrees
        vel_noise = np.random.normal(0, 0.2)  # m/s drift
        
        noisy_yaw = true_yaw + yaw_noise
        noisy_vel = max(0.0, V_BASE + vel_noise)
        
        # Pitch and Roll hold relatively steady with slight drift noise
        pitch = np.random.normal(0, 0.02)
        roll = np.random.normal(0, 0.02)
        
        # Determine GPR score (0 to 1 based on distance to nearest mine)
        if min_dist < 6.0:
            gpr = np.random.uniform(0.95, 1.0)
        elif min_dist < 10.0:
            gpr = np.random.uniform(0.4, 0.94)
        else:
            gpr = np.random.uniform(0.0, 0.3)
            
        # Add Occasional False Positives (like metallic debris)
        if np.random.rand() < 0.005: 
            gpr = np.random.uniform(0.95, 1.0)
            
        # Simulate Vision-Based Depth Sensing Altitude Returns
        altitude = get_elevation(current_pos[0], current_pos[1])
        altitude += np.random.normal(0, 0.5) # Sensor noise
            
        url = f"http://internal-drone-net/cam/frame_{elapsed_ms:08d}.jpg"
            
        telemetry.append({
            "Elapsed_ms": int(elapsed_ms),
            "Pitch": round(float(pitch), 4),
            "Roll": round(float(roll), 4),
            "Yaw": round(float(noisy_yaw), 4),
            "Velocity_ms": round(float(noisy_vel), 4),
            "Altitude_m": round(float(altitude), 2),
            "GPR_Score": round(float(gpr), 4),
            "Deviation_Flag": is_deviating,
            "Camera_Frame_URL": url
        })
        
        elapsed_ms += int(DT * 1000)
        
    df = pd.DataFrame(telemetry)
    
    # Save coordinates of ground truth mines for later diagnostic evaluation 
    pd.DataFrame(mines, columns=['X', 'Y']).to_csv('ground_truth_mines.csv', index=False)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_x", type=float, default=500.0)
    parser.add_argument("--target_y", type=float, default=500.0)
    args = parser.parse_args()
    
    print(f"Initializing Phase 1 Simulator. Vectoring Point B -> ({args.target_x}, {args.target_y})")
    df = simulate_data(args.target_x, args.target_y)
    output_path = "telemetry_log.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} telemetry logs representing dynamic flight.")
    print(f"Secured Data saved to {output_path}")
