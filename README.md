# Technical Architecture & Walkthrough

This Proof of Concept (PoC) simulates a fully autonomous, **GPS-Denied military reconnaissance drone** mapping a safe convoy route through a heavily mined area. 

Because we operate under strict Electronic Warfare (EW) rules where GNSS/GPS protocols are jammed, this pipeline relies entirely on internal **Inertial Navigation Systems (INS)** via Dead Reckoning and **Computer Vision Depth Sensing**.

Our execution is divided into three distinct execution phases. Below is the step-by-step breakdown of exactly how each component operates under the hood.

---

## 🏗️ System Architecture: Simulated vs. Executed Reality

Because this is a PoC meant to run locally without expensive hardware (like an actual physical drone or live Vision/Radar arrays), certain hardware sensors are **simulated**, while the math, data processing, and tactical logic are **actually executing**.

| Component | Status | How it works in this PoC |
| :--- | :--- | :--- |
| **Drone Hardware / Flight** | 🟡 **Simulated** | There is no physical drone. `phase1_sim.py` generates a mathematically perfect "Lawnmower" flight path matrix. |
| **GPS Denial Constraints** | 🟢 **Actual** | The system strictly enforces GPS denial. At no point in Phase 2 or 3 does the logic use X/Y coordinates from the map. |
| **IMU Sensors (Velocity/Yaw)** | 🟡 **Simulated** | We generate baseline Velocity and Yaw, but inject **Actual** mathematical Gaussian noise to mimic cheap hardware sensor drift. |
| **Dead Reckoning (INS Engine)** | 🟢 **Actual** | `phase2_engine.py` **actually executes** Moving Average filters and Trigonometric integration (`dx = v * cos(yaw) * dt`) to recreate flight paths. |
| **Ground Penetrating Radar (GPR)** | 🟡 **Simulated** | When the simulated drone passes over the artificial mine matrix, a random float `> 0.95` is injected into the CSV. |
| **Anomaly Clustering** | 🟢 **Actual** | `phase2_engine.py` runs a legitimate spatial aggregation algorithm using KDTree logic to deduplicate physical threats within a 5-meter radius. |
| **Multimodal Vision LLM** | 🟡 **Simulated** | We are *not* sending real images bounding boxes to an API. We trigger a mock function representing a 90% vision confirmation probability. |
| **Depth Sensing (Z-Axis)** | 🟡 **Simulated** | Complex 3D sine wave and exponential ridge functions construct realistic terrain rather than relying on a 2GB Lidar point cloud. |
| **A\* Pathfinding AI** | 🟢 **Actual** | The A* algorithm is a **100% real** matrix. It successfully paths through a 3D volume, avoiding blast radii and punishing steep Z-axis climbs. |

---

## Phase 1: Mission Planning & Data Simulation
**Script:** `phase1_sim.py`  
**Output:** `telemetry_log.csv` (12,000+ rows)

In lieu of flying a physical drone for the hackathon, we simulate the environment and sensor readings.

**What it does:**
1. **Lawnmower Path Generation**: The drone calculates a non-overlapping扫 sweep pattern across a 500m x 500m local Cartesian grid.
2. **Kinematic Movement**: It generates a flight path by moving at a base velocity of `5.0 m/s` at a simulated polling rate of `10Hz` (`100ms`).
3. **INS Sensor Drift Injection**: Real IMU sensors are incredibly noisy. Rather than outputting perfect `(X, Y)` coordinates, we strip the coordinates out entirely. Instead, we generate standard IMU outputs (`Yaw`/Heading and `Velocity`) and inject randomized Gaussian noise to mimic real-world hardware drift.
4. **Threat Placement**: We scatter 75 invisible mines randomly across the grid. As the drone passes near a mine, its simulated Ground Penetrating Radar (GPR) generates high anomaly detection scores.
5. **Computer Vision Depth Map**: To simulate visual topology, the system mathematically calculates a dynamic terrain combining rolling hills, deep craters, and a steep central ridge across the 500x500 operational grid.

### 📋 Example Telemetry Output `(telemetry_log.csv)`
```csv
Elapsed_ms, Pitch, Roll, Yaw, Velocity_ms, Altitude_m, GPR_Score, Camera_Frame_URL
100,        0.012, -0.01, 1.571, 5.21,       10.45,       0.12,     http://net/frame_00000100.jpg
200,        0.009, 0.02,  1.568, 4.95,       11.02,       0.98,     http://net/frame_00000200.jpg
```
*(Notice: There are no X/Y/Z coordinates! The drone does not know where it is in the world).*

---

## Phase 2: Data Consolidation & Threat Detection
**Script:** `phase2_engine.py`  
**Output:** `tactical_map_data.json`

This is the core tactical engine running on the base-camp servers. It ingests the encrypted `telemetry_log.csv` and recalculates the operation purely using trigonometry.

**What it does:**
1. **Moving Average Dead Reckoning**: The engine streams the CSV telemetry chronologically. It applies a rolling window (size=5) to smooth out the noisy `Yaw` and `Velocity` variables.
2. **Trigonometric Coordinate Integration**: For every `100ms` tick, it calculates how far the drone moved on the local X/Y axis using:
   - `dx = Smoothed_Velocity * cos(Smoothed_Yaw) * dt`
   - `dy = Smoothed_Velocity * sin(Smoothed_Yaw) * dt`
   By accumulating these deltas from `[Base Camp X:0, Y:0]`, we plot the entire flight path **without ever utilizing a satellite map**.
3. **Anomaly Clustering**: Any interpolated X/Y coordinate tied to a GPR score `> 0.95` is flagged. Because GPR signals often trigger sequentially on the same object, the engine clusters all points within 5 meters of each other to deduplicate the threat matrix.
4. **LLM Verification API**: For every clustered anomaly, the system extracts the associated `Camera_Frame_URL` and sends a mock payload to a multimodal LLM to visually confirm the threat. 
5. **Terrain-Aware A\* (A-Star) Pathfinding**: 
   - We generate a global **Topological Grid**, attaching a randomly varying `Z` altitude to every `X/Y` grid block.
   - We assign a `15-meter` blast exclusion radius around every confirmed mine.
   - The A* algorithm runs from Base Camp `(0,0)` to Point B `(500,500)`. It balances the base distance of the path against a **Heavy Elevation Penalty**. The penalty forces the route to seek out deep, flat valleys (indicated by dark purples in the UI) and avoids traversing steep mountain ridges to prevent vehicle rollovers!

### 📋 Example Engine Output `(tactical_map_data.json)`
```json
{
  "trajectory": [[0.0, 0.0, 10.0], [0.5, 0.02, 10.45]],
  "anomalies": [[245.2, 112.5, 23.4]],
  "terrain_grid": [{"x": 0, "y": 0, "z": 12.5}],
  "safe_route": [{"x": 0, "y": 0, "z": 12.5}, {"x": 10, "y": 0, "z": 13.0}]
}
```

---

## Phase 3: Tactical UI Dashboard
**Script:** `phase3_dashboard.py`  

We built a custom web-based tracking dashboard using `Streamlit` and `Altair` to visualize the operation.

**What it does:**
1. **Interactive Loading Sequence**: Utilizing Streamlit `st.empty` and expansion modules, the dashboard simulates the terminal ingestion pipeline—visually decrypting data, running the progress bars for the A* math, and verifying threats live in the sidebar before clearing the payload into the map renderer.
2. **The Topological Depth Heatmap**: It parses the `terrain_grid` array from JSON. It maps those explicitly defined altitude points cleanly onto an Ordinal grid layer using a scientific `viridis` color scale (Dark Purple = Low Valleys | Bright Yellow = High Ridges).
3. **The Layered Execution Plan**:
   - Drops the calculated **INS Drone Path** as 3D tooltipped grey dashed lines over the map.
   - Distinctly renders the **Confirmed Hazards** as heavy red markers.
   - Draws the ultimate **A* Safe Convoy Route** as a thickened, glowing trail.
4. **Live Color-Encoding**: The Safe Convoy Route itself is not a static color! The `turbo` color mapping adjusts dynamically across the line based on the exact `Z: Altitude` slope the convoy will be navigating at that specific meter along the track.

*The result is a tactical PoC demonstrating how internal Dead Reckoning, Vision Sensing, and multi-cost Pathfinding logic can execute complex drone logistics inside a GPS jammed environment.*
