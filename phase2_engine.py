import pandas as pd
import numpy as np
import math
import json
import heapq
import random

# Core simulation settings matching Phase 1
GRID_W = 500
GRID_H = 500
ANOMALY_THRESHOLD = 0.95
HAZARD_RADIUS = 15.0 # Mined area safety buffer

def mock_llm_vision_api(frame_url):
    """
    Mocks a call to a multimodal LLM to verify if a drone camera frame contains a mine.
    For the hackathon PoC, we will simulate this returning True for most high GPR scores.
    """
    # Simply simulate a 90% true positive rate from the LLM based on GPR flags
    return np.random.rand() < 0.90

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
                for hx, hy, hz in hazards:
                    if math.hypot(nx - hx, ny - hy) < HAZARD_RADIUS:
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

def process_telemetry():
    print("Loading encrypted telemetry logs...")
    df = pd.read_csv("telemetry_log.csv")
    
    engine = DeadReckoningEngine()
    
    calculated_positions = []
    anomalies = []
    
    # Simulating iteration through real-time streaming data over time
    prev_time = df.iloc[0]["Elapsed_ms"]
    
    for _, row in df.iterrows():
        current_time = row["Elapsed_ms"]
        dt = (current_time - prev_time) / 1000.0 # Convert ms to seconds
        prev_time = current_time
        
        # Dead Reckon position based purely on yaw, velocity, and Vision System Depth sensing
        x, y, z = engine.update_position(dt, row["Yaw"], row["Velocity_ms"], row["Altitude_m"])
        calculated_positions.append({"X": x, "Y": y, "Z": z})
        
        # Check GPR Threshold
        if row["GPR_Score"] > ANOMALY_THRESHOLD:
            anomalies.append({
                "X": x, 
                "Y": y, 
                "Z": z,
                "GPR": row["GPR_Score"], 
                "URL": row["Camera_Frame_URL"]
            })
            
    print(f"INS filtering complete. Raw anomalies detected: {len(anomalies)}")
    
    # Deduplicate anomalies (cluster points that are extremely close to one another)
    confirmed_mines = []
    for anomaly in anomalies:
        is_new = True
        for mine in confirmed_mines:
            if math.hypot(anomaly["X"] - mine[0], anomaly["Y"] - mine[1]) < 5.0:
                is_new = False
                break
        
        if is_new:
            # Invoke LLM Vision API verification
            if mock_llm_vision_api(anomaly["URL"]):
                confirmed_mines.append((anomaly["X"], anomaly["Y"], anomaly["Z"]))
                
    print(f"LLM Vision verified mines: {len(confirmed_mines)}")
    
    print("Calculating Topography Grid...")
    terrain_map = generate_global_terrain(GRID_W, GRID_H, 10.0, engine.path)
    
    # Calculate Safe Path (Base Camp to Point B)
    start_point = (0.0, 0.0)
    end_point = (GRID_W, GRID_H)
    
    print("Calculating safe 3D convoy route using A* and Topography constraints...")
    safe_path = run_a_star(start_point, end_point, confirmed_mines, terrain_map, GRID_W, GRID_H)
    
    if not safe_path:
        print("CRITICAL: No safe path could be mapped through the minefield.")
    else:
        print(f"Safe route generated with {len(safe_path)} waypoints.")
    
    # Consolidate Output for Phase 3 UI (including the global terrain mapping)
    export_terrain = [{"x": k[0], "y": k[1], "z": v} for k, v in terrain_map.items()]
    
    output_data = {
        "trajectory": engine.path,
        "anomalies": confirmed_mines,
        "safe_route": safe_path,
        "terrain_grid": export_terrain
    }
    
    with open("tactical_map_data.json", "w") as f:
        json.dump(output_data, f)
        
    print("Tactical data exported to tactical_map_data.json")

if __name__ == "__main__":
    process_telemetry()
