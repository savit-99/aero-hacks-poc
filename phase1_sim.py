import pandas as pd
import numpy as np
import math
import hashlib

# Base configurations
GRID_W = 500
GRID_H = 500
SWEEP_SPACING = 50
V_BASE = 5.0  # m/s
DT = 0.1      # 10Hz
NUM_MINES = 25

def generate_lawnmower_waypoints():
    """Generates the coordinates for a non-overlapping lawnmower path."""
    waypoints = []
    y = 0
    direction = 1
    while y <= GRID_H:
        if direction == 1:
            waypoints.append((0.0, float(y)))
            waypoints.append((float(GRID_W), float(y)))
        else:
            waypoints.append((float(GRID_W), float(y)))
            waypoints.append((0.0, float(y)))
        y += SWEEP_SPACING
        direction *= -1
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

def simulate_data():
    """Simulates drone flight data including noisy INS measurements and GPR scores."""
    waypoints = generate_lawnmower_waypoints()
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
            
        # Move drone kinematically
        dir_vec = (target_pos - current_pos) / dist
        current_pos += dir_vec * V_BASE * DT
        
        # Calculate true yaw and velocity
        true_yaw = math.atan2(dir_vec[1], dir_vec[0])
        
        # Inject realistic INS noise
        yaw_noise = np.random.normal(0, 0.05) # ~2.8 degrees
        vel_noise = np.random.normal(0, 0.2)  # m/s drift
        
        noisy_yaw = true_yaw + yaw_noise
        noisy_vel = max(0.0, V_BASE + vel_noise)
        
        # Pitch and Roll hold relatively steady with slight drift noise
        pitch = np.random.normal(0, 0.02)
        roll = np.random.normal(0, 0.02)
        
        # Determine GPR score (0 to 1 based on distance to nearest mine)
        dists_to_mines = np.linalg.norm(mines - current_pos, axis=1)
        min_dist = np.min(dists_to_mines)
        
        if min_dist < 3.0:
            # Proximate to a mine: high detection probability
            gpr = np.random.uniform(0.95, 1.0)
        elif min_dist < 8.0:
            # Fringe read
            gpr = np.random.uniform(0.4, 0.94)
        else:
            # Background signal
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
            "Camera_Frame_URL": url
        })
        
        elapsed_ms += int(DT * 1000)
        
    df = pd.DataFrame(telemetry)
    
    # Save coordinates of ground truth mines for later diagnostic evaluation 
    pd.DataFrame(mines, columns=['X', 'Y']).to_csv('ground_truth_mines.csv', index=False)
    
    return df

if __name__ == "__main__":
    print("Initializing Phase 1 Mission Planning & Data Simulation...")
    df = simulate_data()
    output_path = "telemetry_log.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} telemetry logs representing drone sweep flight.")
    print(f"Secured Data saved to {output_path}")
