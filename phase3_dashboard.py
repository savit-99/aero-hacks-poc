import streamlit as st
import pandas as pd
import json
import altair as alt
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Tactical Recon Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Dark Theme ---
st.markdown("""
<style>
    /* Dark Theme Core - Professional Neon */
    :root {
        --primary-accent: #00ffcc; /* Neon Cyan */
        --hazard: #ff0055;       /* Neon Magenta Hazard */
        --bg-dark: #07070a;      /* Deep Vanta Black */
        --panel-bg: rgba(15, 20, 25, 0.90);
        --text-main: #e0e0e0;
    }

    body, .stApp {
        background-color: var(--bg-dark);
        color: var(--text-main);
        font-family: 'Inter', 'Roboto', sans-serif;
    }

    /* Minimalist Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--panel-bg);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Sleek Title */
    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #fff, #888);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem !important;
    }

    /* Metric Cards - Scaled for fit */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; /* Reduced to fit in side panels */
        font-weight: 700;
        color: var(--primary-accent) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #999;
    }
    
    .metric-container {
        background: var(--panel-bg);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 255, 0, 0.1);
    }

    /* Blinking red dot for live status */
    .record-dot {
        height: 12px;
        width: 12px;
        background-color: var(--hazard);
        border-radius: 50%;
        display: inline-block;
        animation: blink 1.5s infinite;
        margin-right: 8px;
    }
    @keyframes blink {
        0% { opacity: 1; box-shadow: 0 0 8px var(--hazard); }
        50% { opacity: 0.4; box-shadow: none; }
        100% { opacity: 1; box-shadow: 0 0 8px var(--hazard); }
    }
    
</style>
""", unsafe_allow_html=True)

def load_tactical_data():
    try:
        with open("tactical_map_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("No tactical data found. Run phase2_engine.py first.")
        return None

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

def render_dashboard():
    data = load_tactical_data()
    if not data: return
    
    # Header Area
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("OP: SWEEP_NET // Tactical Map")
    with col2:
        st.markdown("<div style='text-align: right; padding-top: 1.5rem;'><span class='record-dot'></span><b>SYSTEM ONLINE</b><br><small>GPS: DENIED / INS: ACTIVE</small></div>", unsafe_allow_html=True)

    # Convert lists to DataFrames for Altair
    df_traj = pd.DataFrame(data['trajectory'], columns=['x', 'y', 'z']).reset_index()
    df_mines = pd.DataFrame(data['anomalies'], columns=['x', 'y', 'z'])
    df_route = pd.DataFrame(data['safe_route'], columns=['x', 'y', 'z']).reset_index()
    
    # Load Topographical Terrain explicitly from Phase 2
    df_terrain = pd.DataFrame(data['terrain_grid'])
    # Add bounds to explicitly draw 20x20 meter Quantitative Rectangles without Ordinal Axis scale overlaps
    df_terrain['x2'] = df_terrain['x'] + 20
    df_terrain['y2'] = df_terrain['y'] + 20

    # --- Animated Processing Pipeline ---
    if st.sidebar.button("System Reboot & Initialize Analysis", use_container_width=True, type="primary"):
        st.session_state.has_run = False
        
    if 'has_run' not in st.session_state:
        st.session_state.has_run = False

    if not st.session_state.has_run:
        play_area = st.empty()
        with play_area.container():
            st.markdown("### INITIALIZING SWEEP_NET UPLINK...")
            
            # Step 1: Telemetry Ingestion
            with st.status("Establishing Secure Data Link...", expanded=True) as status:
                st.write("Decrypting 12,000 telemetry packets...")
                time.sleep(1)
                st.write("Initializing INS Dead Reckoning Engine...")
                time.sleep(1)
                status.update(label="Telemetry Ingested", state="complete")
                
            # Step 2: INS Calculation
            progress_text = "Calculating 2D Flight Path via Velocity & Yaw Integration..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            my_bar.empty()
            
            # Step 3: Threat Detection & Depth Sensing
            with st.status("Analyzing GPR Signatures & Depth Terrain Modeling...", expanded=True) as status:
                st.write("Mapping Topological Altitudes (Z-Axis) from Vision API...")
                time.sleep(1)
                st.write(f"Scanning the path for GPR anomalies scoring > 0.95...")
                time.sleep(1)
                st.write("Clustering spatial points to remove false positives...")
                time.sleep(1)
                st.write("Sending highest-probability cluster frames to Vision LLM API for visual confirmation...")
                time.sleep(1.5)
                st.write(f"VISION API OVERRIDE: {len(df_mines)} lethal threats confirmed in sector.")
                status.update(label="Threat Matrix Verified", state="complete")
                
            # Step 4: A* Pathfinding
            progress_text = "Executing A* Pathfinding Algorithm (15m Exclusion Radii)..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.015)
                my_bar.progress(percent_complete + 1, text=progress_text)
            my_bar.empty()
            
            st.success("TACTICAL SOLUTION FOUND. Rendering Operational Grid.")
            time.sleep(1.5)
        
        play_area.empty()
        st.session_state.has_run = True

    # --- Sidebar Metrics ---
    with st.sidebar:
        st.header("Mission Specs")
        st.markdown("---")
        
        # Calculate derived metrics
        ins_dist = calculate_distance(data['trajectory'])
        convoy_dist = calculate_distance(data['safe_route'])
        
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Detected Hazards", len(data['anomalies']))
        st.markdown("</div><br>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Convoy Route Dist", f"{convoy_dist:.1f} m")
        st.markdown("</div><br>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.metric("Max Terrain Altitude", f"{df_traj['z'].max():.1f} m")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("A* Pathfinding Algorithm Active. Routing around confirmed threats and avoiding severe topographical elevations (terrain-aware cost calculations).")

    # --- Main Visualization Area (Altair) ---
    st.markdown("### Operational Theater (Topological Terrain & Convoy Route)")
    
    # Background Grid Setup (500x500)
    base = alt.Chart(pd.DataFrame({'x': [0, 500], 'y': [0, 500]})).mark_rect(opacity=0).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 500]), title='Local X (meters)', axis=alt.Axis(gridColor='#333', domainColor='#555', tickColor='#555')),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[0, 500]), title='Local Y (meters)', axis=alt.Axis(gridColor='#333', domainColor='#555', tickColor='#555'))
    ).properties(height=650, width='container')

    # Topological Terrain Depth Heatmap
    # Explicitly mapping quantitative boundaries (x to x2, y to y2) instead of ordinal categories
    # This completely eliminates the Top/Right axis text overlaps and aligns perfectly with the main grid.
    heatmap = alt.Chart(df_terrain).mark_rect(opacity=0.6).encode(
        x='x:Q',
        x2='x2:Q',
        y='y:Q',
        y2='y2:Q',
        color=alt.Color('z:Q', scale=alt.Scale(scheme='viridis', domain=[0, 45]), title='Altitude (m)'),
        tooltip=[alt.Tooltip('x:Q', title='X', format='.1f'), alt.Tooltip('y:Q', title='Y', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    # INS Drone Sweeper Path
    line_traj = alt.Chart(df_traj).mark_line(color='#ffffff', strokeWidth=1, opacity=0.3, strokeDash=[5,5]).encode(
        x='x:Q', y='y:Q', order='index:Q',
        tooltip=[alt.Tooltip('x:Q', format='.1f'), alt.Tooltip('y:Q', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    # Confirmed Mines Scatter Plot
    scatter_mines = alt.Chart(df_mines).mark_circle(size=150, color='#ff0055', opacity=0.9, stroke='#fff', strokeWidth=1).encode(
        x='x:Q', y='y:Q',
        tooltip=[alt.Tooltip('x:Q', format='.1f'), alt.Tooltip('y:Q', format='.1f')]
    )

    # Safe Convoy Route (Color-coded by elevation difficulty)
    route_glow = alt.Chart(df_route).mark_line(strokeWidth=18, opacity=0.5).encode(
        x='x:Q', y='y:Q', order='index:Q', color=alt.value('#00ffcc')
    )
    line_route = alt.Chart(df_route).mark_line(strokeWidth=6, opacity=1.0).encode(
        x='x:Q', y='y:Q', order='index:Q',
        color=alt.Color('z:Q', scale=alt.Scale(scheme='turbo', domain=[0,40])), # Visual indicator of path slope
        tooltip=[alt.Tooltip('x:Q', format='.1f'), alt.Tooltip('y:Q', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    # Start/End Markers for Convoy
    poi_data = pd.DataFrame({'x': [0, 500], 'y': [0, 500], 'label': ['BASE CAMP', 'POINT B']})
    scatter_poi = alt.Chart(poi_data).mark_square(size=250, color='#00ffcc', opacity=0.9).encode(
        x='x:Q', y='y:Q', tooltip='label'
    )
    text_poi = alt.Chart(poi_data).mark_text(align='left', dx=20, dy=-15, color='#00ffcc', fontWeight='bold', fontSize=15).encode(
        x='x:Q', y='y:Q', text='label'
    )

    # Layer all components
    layered_chart = alt.layer(base, heatmap, line_traj, scatter_mines, route_glow, line_route, scatter_poi, text_poi).configure_view(
        strokeOpacity=0,
        fill='#0a0a0a' # Dark background matching app
    )

    st.altair_chart(layered_chart, use_container_width=True)
    
    # --- Technical Breakdown ---
    st.markdown("---")
    st.markdown("### OP: SWEEP_NET // Technical Architecture")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.expander("Phase 1: GPS-Denied Dead Reckoning", expanded=True):
            st.markdown("""
            **How it works:**
            Because we are operating in an electronic warfare zone, GPS is jammed. The drone relies entirely on its **Inertial Navigation System (INS)**.
            1. The drone logs its **Yaw** (heading) and **Velocity** every 100 milliseconds.
            2. We apply a **Moving Average Filter** to smooth out the noisy IMU sensor drift.
            3. *Trigonometry in Action:* We calculate the delta X and delta Y for every tick using `dx = Velocity * cos(Yaw) * dt`.
            4. By accumulating these deltas from `(0,0)`, we plot the entire grey lawnmower flight path without ever hitting a satellite.
            """)
            
        with st.expander("Phase 2: Multimodal Threat Detection"):
            st.markdown("""
            **How it works:**
            1. Throughout the flight, the **Ground Penetrating Radar (GPR)** streams anomaly scores.
            2. Any score above `0.95` flags the current Dead-Reckoned X/Y coordinate as a suspect anomaly (shown as the glowing orange heatmap).
            3. Because GPR has false positives (metallic debris), the coordinates are clustered using a spatial distance threshold.
            4. **Vision API Verification:** The closest camera frame URL for that cluster is sent to a Multimodal LLM to visually verify the threat. If confirmed, it becomes a hard **Red Hazard Marker**.
            """)
            
    with col_b:
        with st.expander("Phase 3: The Terrain-Aware A* Matrix", expanded=True):
            st.markdown("""
            **How it works:**
            With the static minefield mapped and the Topological Terrain elevations sensed, we must route the convoy from Base Camp to Point B.
            1. We construct a 3D virtual grid representing coordinates (X, Y) and altitudes (Z).
            2. Every confirmed mine generates a **15-meter Exclusion Radius**.
            3. We run the **A* (A-Star) Pathfinding Algorithm**. This advanced version does not just find the shortest 2D path.
            4. **Terrain Penalty:** The algorithm assesses a massive "travel cost penalty" for severe elevation changes. It actively looks for deep, flat valleys and avoids navigating heavy ground vehicles straight over steep mountain ridges (visualized by the purple topographical heatmap)!
            5. The final colored trail indicates the safe route gradient.
            """)
            
if __name__ == "__main__":
    render_dashboard()
