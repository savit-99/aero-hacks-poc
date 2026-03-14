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
    # --- State Initialization & Sidebar Inputs ---
    if 'has_run' not in st.session_state:
        st.session_state.has_run = False
        st.session_state.target_x = 500.0
        st.session_state.target_y = 500.0

    with st.sidebar:
        st.header("Mission Specs")
        st.markdown("---")
        
        st.subheader("Point B Targeting")
        demo_routes = {
            "Operation Alpha (Default)": (500.0, 500.0),
            "Operation Bravo (Deep Valley)": (100.0, 480.0),
            "Operation Charlie (Steep Ridge)": (480.0, 100.0),
            "Custom Target": None
        }
        
        selected_demo = st.selectbox("Select Target Point B:", list(demo_routes.keys()))
        
        if selected_demo == "Custom Target":
            col_x, col_y = st.columns(2)
            pt_b_x = col_x.number_input("Target X", min_value=10.0, max_value=500.0, value=300.0, step=10.0)
            pt_b_y = col_y.number_input("Target Y", min_value=10.0, max_value=500.0, value=300.0, step=10.0)
        else:
            pt_b_x, pt_b_y = demo_routes[selected_demo]
            
        if st.button("System Reboot & Initialize Analysis", use_container_width=True, type="primary"):
            st.session_state.has_run = False
            st.session_state.do_animation = True
            st.session_state.target_x = pt_b_x
            st.session_state.target_y = pt_b_y

    # --- Animated Processing Pipeline ---
    if not st.session_state.has_run:
        st.session_state.do_animation = True  # Force animation on fresh run
        play_area = st.empty()
        import subprocess
        with play_area.container():
            st.markdown("### INITIALIZING SWEEP_NET UPLINK...")
            
            # Step 1: Telemetry Ingestion & Real-Time Setup
            with st.status("Establishing Secure Data Link...", expanded=True) as status:
                st.write(f"Transmitting target coordinates ({st.session_state.target_x}, {st.session_state.target_y}) to Drone...")
                subprocess.run(["python3", "phase1_sim.py", "--target_x", str(st.session_state.target_x), "--target_y", str(st.session_state.target_y)])
                st.write("Decrypting telemetry packets...")
                status.update(label="Telemetry Ingested", state="complete")
                
            # Step 2: Path Generation Bar
            progress_text = "Tracking Adaptive 2D Flight Path & Generating Exclusions..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            my_bar.empty()
            
            # Step 3: Threat Detection & Engine Analysis
            with st.status("Analyzing GPR Signatures & Off-Road Viability...", expanded=True) as status:
                st.write("Initializing INS Dead Reckoning Engine & Mapping Altitudes...")
                subprocess.run(["python3", "phase2_engine.py", "--target_x", str(st.session_state.target_x), "--target_y", str(st.session_state.target_y)])
                st.write("Extracting Mine Deviation Anchors...")
                st.write("Evaluating Z-Altitude variance around dodging deviations...")
                st.write("Sending highest-probability cluster frames to Vision LLM API for visual confirmation...")
                time.sleep(1.5)
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

    # --- Data Loading (Post-Calculation) ---
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
    df_terrain = pd.DataFrame(data['terrain_grid'])
    df_offroad = pd.DataFrame(data.get('off_road_viability', []))
    
    df_terrain['x2'] = df_terrain['x'] + 20
    df_terrain['y2'] = df_terrain['y'] + 20

    # --- Sidebar Metrics ---
    with st.sidebar:
        st.markdown("---")
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
        st.caption("A* Pathfinding Algorithm Active. Routing around confirmed threats and avoiding severe topographical elevations.")

    # --- Main Visualization Area (Altair) ---
    st.markdown("### 🚁 Phase 1: Autonomous UAV Search Vectoring")
    st.caption("Visualizing the drone's initial flight path as it vectors towards Point B, scanning topography and dodging detected hazards.")
    
    base = alt.Chart(pd.DataFrame({'x': [0, 500], 'y': [0, 500]})).mark_rect(opacity=0).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 500]), title='Local X (meters)', axis=alt.Axis(gridColor='#333', domainColor='#555', tickColor='#555')),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[0, 500]), title='Local Y (meters)', axis=alt.Axis(gridColor='#333', domainColor='#555', tickColor='#555'))
    ).properties(height=650, width='container')

    heatmap = alt.Chart(df_terrain).mark_rect(opacity=0.6).encode(
        x='x:Q', x2='x2:Q', y='y:Q', y2='y2:Q',
        color=alt.Color('z:Q', scale=alt.Scale(scheme='viridis', domain=[0, 45]), title='Altitude (m)'),
        tooltip=[alt.Tooltip('x:Q', title='X', format='.1f'), alt.Tooltip('y:Q', title='Y', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    line_traj = alt.Chart(df_traj).mark_line(color='#ffffff', strokeWidth=2, opacity=0.4, strokeDash=[4,4]).encode(
        x='x:Q', y='y:Q', order='index:Q',
        tooltip=[alt.Tooltip('x:Q', format='.1f'), alt.Tooltip('y:Q', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    scatter_mines = alt.Chart(df_mines).mark_circle(size=150, color='#ff0055', opacity=0.9, stroke='#fff', strokeWidth=1).encode(
        x='x:Q', y='y:Q',
        tooltip=[alt.Tooltip('x:Q', format='.1f'), alt.Tooltip('y:Q', format='.1f')]
    )

    # Dynamic Safe Convoy Route
    route_glow = alt.Chart(df_route).mark_line(strokeWidth=18, opacity=0.5).encode(
        x='x:Q', y='y:Q', order='index:Q', color=alt.value('#00ffcc')
    )
    line_route = alt.Chart(df_route).mark_line(strokeWidth=6, opacity=1.0).encode(
        x='x:Q', y='y:Q', order='index:Q',
        color=alt.Color('z:Q', scale=alt.Scale(scheme='turbo', domain=[0,40])), 
        tooltip=[alt.Tooltip('x:Q', title='X', format='.1f'), alt.Tooltip('y:Q', title='Y', format='.1f'), alt.Tooltip('z:Q', title='Altitude', format='.1f')]
    )

    poi_data = pd.DataFrame({'x': [0, st.session_state.target_x], 'y': [0, st.session_state.target_y], 'label': ['BASE CAMP', 'POINT B']})
    scatter_poi = alt.Chart(poi_data).mark_square(size=250, color='#00ffcc', opacity=0.9).encode(
        x='x:Q', y='y:Q', tooltip='label'
    )
    text_poi = alt.Chart(poi_data).mark_text(align='left', dx=20, dy=-15, color='#00ffcc', fontWeight='bold', fontSize=15).encode(
        x='x:Q', y='y:Q', text='label'
    )

    # Compile layer sets
    layers_convoy = [base, heatmap, line_traj, scatter_mines, route_glow, line_route, scatter_poi, text_poi]
    
    # Phase 1 Search doesn't show the terrain or final path yet!
    layers_search = [base, line_traj, scatter_mines, scatter_poi, text_poi]
    
    # Mount off-road viability evaluation points if present
    if not df_offroad.empty:
        scatter_offroad = alt.Chart(df_offroad).mark_square(size=120, opacity=0.9, stroke='#fff', strokeWidth=2).encode(
            x='x:Q', y='y:Q',
            color=alt.condition(
                alt.datum.status == 'VIABLE',
                alt.value('#00ffcc'), # Cyan for viable detour
                alt.value('#ff0055') # Magenta for impassable detour
            ),
            tooltip=['x', 'y', 'status', 'variance']
        )
        layers_convoy.insert(4, scatter_offroad)
        layers_search.insert(4, scatter_offroad)

    chart_convoy = alt.layer(*layers_convoy).configure_view(strokeOpacity=0, fill='#0a0a0a')
    chart_search_static = alt.layer(*layers_search).configure_view(strokeOpacity=0, fill='#0a0a0a')

    search_placeholder = st.empty()
    
    # --- ANIMATION LOGIC ---
    if st.session_state.get('do_animation', False):
        st.session_state.do_animation = False
        import numpy as np
        
        # Precompute Discovery Indices (when the drone first gets near the mine/anchor)
        mines_discovery = []
        for _, m in df_mines.iterrows():
            dists = np.hypot(df_traj['x'] - m['x'], df_traj['y'] - m['y'])
            mines_discovery.append(dists.idxmin() if not dists.empty else 0)
        df_mines['discovery_idx'] = mines_discovery
        
        if not df_offroad.empty:
            offroad_discovery = []
            for _, o in df_offroad.iterrows():
                dists = np.hypot(df_traj['x'] - o['x'], df_traj['y'] - o['y'])
                offroad_discovery.append(dists.idxmin() if not dists.empty else 0)
            df_offroad['discovery_idx'] = offroad_discovery
            
        num_frames = 100
        chunk_size = max(1, len(df_traj) // num_frames)
        
        for frame_idx in range(1, len(df_traj) + chunk_size, chunk_size):
            current_step = min(frame_idx, len(df_traj) - 1)
            
            # The drone's path includes its history so far, PLUS a theoretical straight line to the end goal.
            history_pts = df_traj.iloc[:current_step+1]
            current_x, current_y = history_pts.iloc[-1]['x'], history_pts.iloc[-1]['y']
            
            # Create a dynamic dataframe for the current perceived path
            frame_path_data = history_pts[['x', 'y', 'z']].copy()
            # Append the direct vector to Point B from wherever the drone currently is
            frame_path_data = pd.concat([frame_path_data, pd.DataFrame([{'x': st.session_state.target_x, 'y': st.session_state.target_y, 'z': 0.0}])], ignore_index=True)
            frame_path_data['index'] = range(len(frame_path_data))
            
            f_line_traj = alt.Chart(frame_path_data).mark_line(color='#ffffff', strokeWidth=4, opacity=1.0, strokeDash=[4,4]).encode(
                x='x:Q', y='y:Q', order='index:Q'
            )
            
            frame_mines = df_mines[df_mines['discovery_idx'] <= current_step]
            if not frame_mines.empty:
                f_scatter_mines = alt.Chart(frame_mines).mark_circle(size=250, color='#ff0055', opacity=1.0, stroke='#fff', strokeWidth=3).encode(
                    x='x:Q', y='y:Q'
                )
            else:
                f_scatter_mines = alt.Chart(pd.DataFrame({'x':[], 'y':[]})).mark_circle().encode(x='x:Q', y='y:Q')
                
            frame_layers = [base, f_line_traj, f_scatter_mines, scatter_poi, text_poi]
            
            if not df_offroad.empty:
                frame_offroad = df_offroad[df_offroad['discovery_idx'] <= current_step]
                if not frame_offroad.empty:
                    f_scatter_offroad = alt.Chart(frame_offroad).mark_square(size=180, opacity=1.0, stroke='#fff', strokeWidth=3).encode(
                        x='x:Q', y='y:Q',
                        color=alt.condition(alt.datum.status == 'VIABLE', alt.value('#00ffcc'), alt.value('#ff0055'))
                    )
                    frame_layers.append(f_scatter_offroad)
            
            frame_chart = alt.layer(*frame_layers).configure_view(strokeOpacity=0, fill='#0a0a0a')
            
            # Force Streamlit to render the precise frame by clearing the placeholder empty block and drawing
            search_placeholder.empty()
            search_placeholder.altair_chart(frame_chart, use_container_width=True)
            time.sleep(0.1) # Renders over exactly 10 seconds (100 frames * 0.1s)
            
        time.sleep(1.0)

    # Render final static search chart
    search_placeholder.altair_chart(chart_search_static, use_container_width=True)
    
    # --- PHASE 2 GROUND CONVOY CHART ---
    st.markdown("---")
    st.markdown("### 🚚 Phase 2: Tactical Convoy Route (A* Solution)")
    st.caption("The fully computed safe ground route. The algorithm actively navigates through flatter valleys and avoids the 15m blast radius of all verified hazards.")
    st.altair_chart(chart_convoy, use_container_width=True)
    
    # --- Technical Breakdown ---
    st.markdown("---")
    st.markdown("### OP: SWEEP_NET // Technical Architecture")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        with st.expander("Phase 1: Adaptive Flight & Collision Avoidance", expanded=True):
            st.markdown("""
            **Dynamic Obstacle Avoidance:**
            Rather than a static lawnmower grid, the simulated drone evaluates the shortest vector to Point B and initiates a localized data-collection sweep alongside it to maximize surface intelligence. 
            Crucially, it executes an **Active Real-Time Collision Logic**: If its flight path encroaches within a 5-meter exclusion radius of a detected mine, it mathematically overrides to trace safely around the perimeter border of the threat.
            """)
            
        with st.expander("Phase 2: Off-Road Viability Evaluation"):
            st.markdown("""
            **Off-Road Terrain Scoring:**
            When the drone is forced to physically divert its flight path around a mine, *it drops a Deviation Anchor marker*. 
            The Tactical Engine later isolates these exact deviation anchors on the Topological Z-grid and analyzes the surrounding variance (steepness/gradient) of the raw terrain. 
            If the variance is minor, the detour is marked **VIABLE** (Cyan Square). If the terrain slopes aggressively, it flags the detour as **IMPASSABLE** (Magenta Square) for ground trucks.
            """)
            
    with col_b:
        with st.expander("Phase 3: The Terrain-Aware A* Matrix", expanded=True):
            st.markdown("""
            **A* Pathfinding Constraints:**
            By plotting the newly detected `Point B` destination, the **A* (A-Star) Pathfinding Algorithm** generates a safe convoy route across the tactical plane.
            It utilizes the Z-Axis topographical heatmap to punish steep terrain slopes, resulting in organic vehicle navigation that completely avoids the 15-meter hard exclusion zones around verified hazards while remaining inside the flatter valleys.
            """)
            
if __name__ == "__main__":
    render_dashboard()
