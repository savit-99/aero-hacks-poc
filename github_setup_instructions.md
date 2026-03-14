# 🚀 Sharing SWEEP_NET with your Team on GitHub

To share this hackathon codebase with your team so they can run the simulator on their own machines, follow these steps exactly:

### Step 1: Create the GitHub Repository
1. Go to your GitHub account in your web browser.
2. Click the `+` in the top right corner and select `New repository`.
3. Name the repository (e.g., `sweep-net-poc`).
4. Keep it **Public** (or **Private** if you want to restrict access to your team).
5. Do **NOT** check "Add a README file" or "Add .gitignore" (we have already created them locally).
6. Click **Create repository**.

### Step 2: Push Your Code via Terminal
Once the repo is created, GitHub will give you a list of commands. Open a new terminal specifically aimed at your project folder (`/Users/savit/Desktop/Aero Hacks`) and run these exact commands (replace `YOUR_USERNAME` and `YOUR_REPO` with the actual URL from step 1):

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

*(Note: The local Git repository has already been initialized by the AI, and the files have been staged and committed. You only need to link the remote `origin` and push!)*

---

# 💻 Instructions for Your Teammates (How to Run It)

Once your teammates clone the repository from GitHub, they need to set up the Python environment exactly as we did to avoid missing dependency errors. 

**Tell them to run these exact commands in their terminal inside the cloned folder:**

### 1. Set up the Virtual Environment
Running isolated environments is critical for Streamlit and Altair.

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Tactical Dashboard (Phase 3)
Because the `tactical_map_data.json` and `telemetry_log.csv` are included in the repository, they don't *need* to re-run the simulators unless they want to generate new random terrain. 

To launch the dashboard immediately:
```bash
streamlit run phase3_dashboard.py
```

### Optional: Re-Running the Deep Simulation (Phase 1 & 2)
If your teammates want to modify the Python code to change how many mines there are, or how the altitude is calculated, tell them to run the pipeline again:

```bash
python phase1_sim.py   # Generates new telemetry_log.csv
python phase2_engine.py # Generates new tactical_map_data.json
```
