# 🛰️ AETHER_OS — 5G Network Slicing Research Platform

A full-stack research platform that **benchmarks 5G network slicing algorithms** and visualises them in a live cinematic 3D dashboard.

---

## 📁 Project Structure

```
5G-project/
├── backend/          ← Python research engine + REST API
├── next_frontend/    ← ✅ Main website (Next.js 3D dashboard)
└── frontend/         ← Legacy Vite prototype (not needed)
```

The **backend** runs algorithms and serves data. The **next_frontend** is the website that visualises everything. They talk to each other over HTTP on your local machine.

---

## ✅ What You Need Installed

| Tool | Version | Check Command |
|------|---------|--------------|
| Python | 3.10 or newer | `python --version` |
| Node.js | 18 or newer | `node --version` |
| npm | 9 or newer | `npm --version` |
| pip | any | `pip --version` |

> **Download links:**
> - Python → [python.org/downloads](https://www.python.org/downloads/)
> - Node.js → [nodejs.org](https://nodejs.org/) (pick the LTS version)

---

## 🐍 PART 1 — Backend Setup

> Do everything in this section in **Terminal 1**.

### 1.1 — Open a terminal in the backend folder

```powershell
cd C:\Users\Ojas\Desktop\5G-project\backend
```

---

### 1.2 — Create a Python virtual environment

A virtual environment keeps your project's packages separate from the rest of your system. You only need to do this **once**.

```powershell
python -m venv .venv
```

---

### 1.3 — Activate the virtual environment

You need to do this **every time** you open a new terminal window before running the backend.

**Windows — PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

> ⚠️ If you see a red "cannot be loaded because running scripts is disabled" error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then run the activate command again.

**Windows — Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

✅ **Success indicator:** Your prompt will now show `(.venv)` at the beginning, like:
```
(.venv) PS C:\Users\Ojas\Desktop\5G-project\backend>
```

---

### 1.4 — Install Python dependencies

```powershell
pip install -r requirements.txt
```

This installs:

| Package | Used for |
|---------|---------|
| `numpy` | Array math, channel simulation, PRB scheduling |
| `pandas` | Result tables, CSV read/write |
| `matplotlib` | Generating all PNG charts |
| `torch` | Neural network training (MAAN, MAPPO algorithms) |
| `scipy` | Statistical significance tests, confidence intervals |
| `fastapi` | REST API that the frontend connects to |
| `uvicorn` | Web server that runs FastAPI |

> ⚠️ `torch` (PyTorch) is the largest download (~200–800 MB). This is normal.
>
> If the install fails with a torch error, try:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

**Verify everything installed correctly:**
```powershell
python -c "import numpy, pandas, matplotlib, torch, scipy, fastapi; print('✅ All packages OK')"
```

---

### 1.5 — Start the Backend API Server

```powershell
python main.py
```

You should see output like:
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **The backend is now running at `http://localhost:8000`**

**Verify it's working** — open a browser or run:
```powershell
curl http://localhost:8000/api/health
# Should return: {"status":"ok"}
```

> 💡 **Interactive API docs** are available at: `http://localhost:8000/docs`

**Keep this terminal open.** The API stops if you close it.

---

## 🌐 PART 2 — Frontend Setup

> Open a **new, second terminal window** for this. Leave Terminal 1 running the backend.

### 2.1 — Open a new terminal in the frontend folder

```powershell
cd C:\Users\Ojas\Desktop\5G-project\next_frontend
```

---

### 2.2 — Install Node.js dependencies

You only need to do this **once** (or after pulling new changes).

```powershell
npm install
```

This will download packages into a `node_modules/` folder. It may take 1–3 minutes.

> ⚠️ If you see peer dependency errors, use:
> ```powershell
> npm install --legacy-peer-deps
> ```

---

### 2.3 — Start the Frontend Dev Server

```powershell
npm run dev
```

You should see:
```
▲ Next.js 14.x.x
- Local:        http://localhost:3000
- Ready in Xs
```

✅ **The dashboard is now running at `http://localhost:3000`**

Open `http://localhost:3000` in your browser.

---

## 🖥️ PART 3 — Using the Dashboard

With both servers running, open **`http://localhost:3000`** in your browser.

### What you'll see

```
┌─────────────────────────────────────────────────────────┐
│  AETHER_OS           C_ADMM  MAAN  STATIC   [Run Full Research] [Result Plots] ● SIMULATION ACTIVE
├─────────────────────────────────────────────────────────┤
│                                                         │
│           3D SPACE — Three glowing orbs orbit           │
│           a central constellation node.                 │
│                                                         │
│           ● Green orb  = C_ADMM algorithm               │
│           ● Red orb    = MAAN algorithm                 │
│           ● Grey orb   = Static Greedy (baseline)       │
│                                                         │
│           An astronaut floats in zero-gravity,          │
│           fleeing from your cursor.                     │
│                                                         │
│                ↓  scroll to explore  ↓                  │
└─────────────────────────────────────────────────────────┘
```

### The 6 Scroll Sections (Beats)

Scroll down through the page. Each full-screen section introduces one concept:

| Beat | What you'll see |
|------|----------------|
| **Beat 0** — Orientation | Overview of all 3 algorithms with live score badges updating every 500ms |
| **Beat 1** — C_ADMM | Deep-dive card with live sparkline + a **slider to control number of network slices** |
| **Beat 2** — MAAN | Deep-dive card with live sparkline + a **slider to control network load** |
| **Beat 3** — Static Greedy | Performance comparison bar showing why this is the baseline to beat |
| **Beat 4** — Full System | Combined dashboard with scores, sparklines, and average utility for all algorithms |
| **Beat 5** — Connect API | Input field to connect your own live 5G telemetry endpoint |

### Navigation

- **Scroll** normally to move between beats
- **Dot indicators** on the right — click any dot to jump to that beat
- **Top nav links** (C_ADMM / MAAN / STATIC_GREEDY) — click to jump directly to that algorithm's beat

---

## 🔬 PART 4 — Running a Research Benchmark

This triggers the actual Python research engine to run a full experiment and generate results.

### From the Dashboard (Recommended)

1. Click the **"Run Full Research"** button in the top navigation bar
2. A green progress bar appears next to the button showing `0% → 100%`
3. The status message below the nav updates in real-time (e.g. *"Completed 3/60: seed=0 load=1.0 alg=C_ADMM"*)
4. When complete, the **Result Plots gallery** opens automatically
5. You can also click **"Result Plots"** at any time to view previously generated charts

### From the Terminal (Alternative)

In Terminal 1 (backend, venv active):

**Quick benchmark — Phase 1** (~2–5 min):
```powershell
python -m src.experiments.run_benchmark
```

**Full research benchmark — Phase 2** (~5–20 min):
```powershell
python -m src.experiments.run_benchmark_phase2
```

---

## 📊 The Algorithms Being Compared

| Algorithm | What it does | Role |
|-----------|-------------|------|
| **MAAN_PPO** | Neural network agent trained with PPO. Uses dual price signals to learn resource allocation. | Main algorithm under test |
| **Ind. MAPPO_PPO** | Separate PPO agent per slice, no coordination or price signals | Decentralised baseline |
| **C_ADMM** | Consensus ADMM — splits the problem across slices and iterates toward a shared feasible solution | Distributed optimiser |
| **Static Greedy** | Fixed proportional rules that never adapt to network conditions | Baseline floor |
| **OMD Bandit** | Online Mirror Descent with bandit-style gradient estimation. No neural networks. | Black-box baseline |

---

## 📁 Output Files (After a Benchmark Run)

```
backend/
├── outputs/                              ← Phase 1 results
│   ├── benchmark_results.csv
│   └── plots/*.png                       (14 charts)
│
└── outputs_phase2/                       ← Phase 2 results (full research)
    ├── benchmark_results_phase2.csv      ← raw per-timestep data for all runs
    ├── summary_with_ci95.csv             ← per-algorithm means + 95% CI
    ├── statistical_significance.csv      ← p-values vs MAAN_PPO
    ├── config_used.json                  ← exact experiment settings
    ├── plots/*.png                       (14 diagnostic charts)
    └── plots_publication/*.png           (6 publication-quality figures)
```

These PNG files are automatically served by the backend API and viewable in the **Result Plots** overlay in the dashboard.

---

## ⚙️ Configuration

### Backend — Experiment Parameters

Controlled by `ExpConfig` in `backend/src/experiments/run_benchmark_phase2.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `horizon` | 500 | Time slots per episode. Reduce to 100 for a quick test. |
| `seeds` | 6 | Independent random runs per config. Reduce to 2 for speed. |
| `load_scales` | `(0.8, 1.0, 1.2, 1.4, 1.6)` | Traffic load multipliers to sweep over. |
| `n_mc_urlcc` | 64 | SAA samples for URLLC chance-constraint. Reduce to 16 for speed. |
| `num_slices` | 3 | Number of network slices (eMBB + URLLC + mMTC). |

### Frontend — Backend URL

By default the frontend connects to `http://localhost:8000`. To change this, create a `.env.local` file in `next_frontend/`:

```env
NEXT_PUBLIC_BACKEND_URL=http://your-backend-host:8000
```

---

## 🔄 Simulation vs Live Mode

| Mode | When it's active | Data source |
|------|-----------------|-------------|
| **Simulation** (default) | Always — no backend needed | Browser generates fake sine-wave telemetry |
| **Live** | After clicking "Run Full Research" | Backend serves real benchmark results |

In Simulation Mode, the 3D orbs and sparklines still animate — the utilisation values are mathematically generated in the browser using sine functions that respond to the sliders.

---

## 🛠️ Troubleshooting

### Backend won't start — "Port 8000 already in use"

```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill it (replace 12345 with the actual PID shown)
taskkill /PID 12345 /F
```

Then restart: `python main.py`

---

### Frontend can't connect to backend (fetch errors in browser console)

1. Make sure the backend is actually running — check Terminal 1
2. Visit `http://localhost:8000/api/health` in your browser — should show `{"status":"ok"}`
3. Make sure both are on the same machine (backend on 8000, frontend on 3000)
4. The backend has CORS fully open (`allow_origins=["*"]`), so CORS is not the issue

---

### `No module named 'torch'` when starting backend

```powershell
# Activate venv first, then:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### `No module named 'src'` error

You're running the Python command from the wrong folder. Must be inside `backend/`:
```powershell
cd C:\Users\Ojas\Desktop\5G-project\backend
python -m src.experiments.run_benchmark_phase2
```

---

### Virtual environment activation blocked by PowerShell

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### `npm install` fails with peer dependency errors

```powershell
npm install --legacy-peer-deps
```

---

### Result Plots gallery is empty / shows nothing

You need to run a benchmark first. The gallery only shows files that exist in `outputs_phase2/plots/`. Click **"Run Full Research"** in the nav bar and wait for it to complete.

---

### Benchmark is too slow

Edit `run_benchmark_phase2.py` and temporarily use smaller values:
```python
cfg = ExpConfig(
    horizon=100,       # was 500
    seeds=2,           # was 6
    n_mc_urlcc=16,     # was 64
    load_scales=(0.8, 1.2, 1.6),  # was 5 values
)
```

---

## ⚡ Quick Reference — All Commands

```powershell
# ─── BACKEND (Terminal 1) ─────────────────────────────────

# Navigate to backend
cd C:\Users\Ojas\Desktop\5G-project\backend

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install packages (first time only)
pip install -r requirements.txt

# Start the API server
python main.py

# ─── ALTERNATIVELY: run experiments directly ──────────────

# Phase 1 quick benchmark
python -m src.experiments.run_benchmark

# Phase 2 full benchmark (recommended)
python -m src.experiments.run_benchmark_phase2


# ─── FRONTEND (Terminal 2) ────────────────────────────────

# Navigate to frontend
cd C:\Users\Ojas\Desktop\5G-project\next_frontend

# Install packages (first time only)
npm install

# Start the dashboard
npm run dev


# ─── OPEN IN BROWSER ──────────────────────────────────────

# Dashboard
http://localhost:3000

# API health check
http://localhost:8000/api/health

# API interactive docs
http://localhost:8000/docs
```

---

## 📌 Summary — Normal Workflow

```
 Terminal 1                        Terminal 2                  Browser
─────────────                     ─────────────               ────────────────
cd backend                        cd next_frontend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   npm install
python main.py          →         npm run dev        →        http://localhost:3000
[API running]           →         [Site running]     →        Click "Run Full Research"
[Benchmark running...]                               ←        [Progress bar updates]
[Done → plots saved]              ←                 ←        [Plot gallery opens]
```

---

*Stack: Python 3.10+ · FastAPI · Uvicorn · Next.js 14 · Three.js · React Three Fiber · Framer Motion · PyTorch · TailwindCSS*
