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

The **backend** runs algorithms and serves data. The **next_frontend** is the website that visualises everything. They talk to each other over HTTP — locally, or across two deployed services.

> **Deploying?** Jump to [PART 6 — Deploying to Render](#-part-6--deploying-to-render). The repo ships a `render.yaml` blueprint.

---

## 🔬 The Model, in Brief

Each slice is served by three queues in series — radio, transport, and compute — and every
quantity carries an explicit unit:

| Symbol | Unit | Meaning |
|--------|------|---------|
| `λ_s` | bit/s | offered traffic of slice *s* |
| `b_{s,k}` | PRB | resource blocks of gNB *k* given to slice *s* |
| `R_s` | bit/s | achieved radio rate, `Σ_k b_{s,k} · W_prb · log₂(1+SINR)` |
| `τ_s` | bit/s | transport capacity given to slice *s* |
| `c_{s,m}` | cycle/s | MEC compute given to slice *s* on host *m* |
| `ω_s` | cycle/bit | processing density of slice *s* |
| `L` | bit | mean packet size (12 000 bit = 1500 B) |

Each domain is an M/M/1 queue, so its mean sojourn time is

```
W = L / (μ − λ)        [s]
```

with `μ = R_s` for radio, `τ_s` for transport, and `c_s/ω_s` for compute. End-to-end delay
is `d_radio + d_trans + d_comp`, and a slice meets its SLA when `R_s ≥ r_min` **and**
`D_s ≤ d_max`. Unstable queues (`μ ≤ λ`) saturate at 1 s rather than diverging.

Utility is proportional-fair with a bounded delay cost:

```
u_s = α_s·log(1 + R_s/r_min_s) − β_s·min(D_s/d_max_s, 3) − γ_s·σ(20·(D_s/d_max_s − 1))
```

The log rate term prevents the degenerate solution (hand every PRB to one slice) that a
linear term produces; the cap on the delay term keeps utility O(1) instead of letting a
saturated queue against an 8 ms budget swamp everything else.

---

## ✅ What You Need Installed

| Tool | Version | Check Command |
|------|---------|--------------|
| Python | 3.12 recommended (3.10+ works) | `python --version` |
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
cd <path-to-repo>\backend
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
(.venv) PS <path-to-repo>\backend>
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

> ℹ️ `requirements.txt` pulls the **CPU-only** PyTorch build via an `--extra-index-url`,
> and pins every version. Both matter:
> - The default PyPI `torch` bundles the CUDA runtime (~2.5 GB with its `nvidia-*` deps),
>   which is dead weight on any machine without an NVIDIA GPU. The CPU build is ~500 MB.
> - Unpinned versions is how this project silently picked up matplotlib 3.11, which
>   removed the `boxplot(labels=)` argument the plotting code used — the research run
>   completed the whole benchmark and then died at 90%.
>
> Expect ~890 MB installed and a few minutes on a first install.

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
# {"status":"ok","service":"5g-slicing-benchmark","results_available":true,"research_running":false}
```

> 💡 **Interactive API docs** are available at: `http://localhost:8000/docs`

**Keep this terminal open.** The API stops if you close it.

---

## 🌐 PART 2 — Frontend Setup

> Open a **new, second terminal window** for this. Leave Terminal 1 running the backend.

### 2.1 — Open a new terminal in the frontend folder

```powershell
cd <path-to-repo>\next_frontend
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
│  AETHER_OS           C_ADMM  MAAN  STATIC   [Run Demo Benchmark] [Result Plots] ● SIMULATION ACTIVE
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
| **Beat 5** — Real Results | Button that opens the benchmark figure gallery served by the backend. Ingesting telemetry from an external 5G network is **not** implemented. |

### Navigation

- **Scroll** normally to move between beats
- **Dot indicators** on the right — click any dot to jump to that beat
- **Top nav links** (C_ADMM / MAAN / STATIC_GREEDY) — click to jump directly to that algorithm's beat

---

## 🔬 PART 4 — Running a Research Benchmark

This triggers the actual Python research engine to run a full experiment and generate results.

### From the Dashboard (Recommended)

1. Click the **"Run Demo Benchmark"** button in the top navigation bar
2. A green progress bar appears next to the button showing `0% → 100%`
3. The status message below the nav updates in real-time (e.g. *"Completed 3/60: seed=0 load=1.0 alg=C_ADMM"*)
4. When complete, the **Result Plots gallery** opens automatically on its **Demo Run** tab
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
| **Ind. MAPPO_PPO** | Separate PPO agent per slice, no coordination or price signals | Ablation of the price mechanism |
| **C_ADMM** | Consensus ADMM. Per-slice primal steps use closed-form gradients of a **smooth relaxation** of the scored utility — the log-rate reward and the three M/M/1 delay terms — then project onto the shared capacity constraints. The relaxation omits the `γ·σ(·)` violation penalty (γ is not passed to the allocator at all) and ignores the `min(·, 3)` delay-cost saturation, so it is *not* the exact scored objective. | Distributed optimiser |
| **Static Greedy** | Fixed proportional rules plus a greedy QoS repair loop; does not learn | Baseline floor |
| **OGD_Bandit** | Projected online gradient ascent with one-point bandit feedback. No neural networks. | Black-box baseline |

> **Naming note:** this baseline was previously called `OMD_BF` ("Online Mirror Descent").
> The implementation uses a Euclidean projection with no Bregman divergence or mirror map,
> so it is online *gradient* descent with bandit feedback, not mirror descent. Renamed to
> match what the code does.

### What the current results actually show

From the committed run (6 seeds × 5 loads × horizon 500, paired *t*-tests with
Holm-Bonferroni correction — see `outputs_phase2/statistical_significance.csv`):

QoS success ratio, lowest load → highest load:

| Algorithm | 0.8 → 1.6 |
|---|---|
| C_ADMM | 0.758 → 0.635 |
| MAAN_PPO | 0.642 → 0.557 |
| Independent_MAPPO_PPO | 0.635 → 0.552 |
| Static_Greedy | 0.487 → 0.262 |
| OGD_Bandit | 0.350 → 0.310 |

* All five algorithms degrade monotonically as offered load rises, as expected.
* **C_ADMM leads on utility and QoS success**, beating MAAN_PPO at **all 5** load points
  on utility (Holm-adjusted *p* ≤ 0.0011; utility margin +0.26 to +0.38) and at all 5 on
  QoS success (Holm-adjusted *p* ≤ 0.00034). This is **not** a clean sweep of the three
  tested metrics: on **mean delay C_ADMM is significantly *worse*** than MAAN_PPO at every
  load (Holm-adjusted *p* between 1.6e-04 and 5.3e-04). Working from closed-form gradients
  of a smooth relaxation of the scored objective is a real advantage over learning it
  online — but see the C_ADMM row above for what that relaxation leaves out.
* **MAAN_PPO shows no significant advantage over the Independent MAPPO ablation** at any
  load, on any tested metric (Holm-adjusted *p* ≥ 0.26 for utility, ≥ 0.78 for QoS
  success, ≥ 0.98 for mean delay; utility differences ≤ 0.053). On this benchmark the
  dual-price coordination mechanism is **not** demonstrably helping. That is a negative
  result, and it is reported rather than tuned away.
* **MAAN_PPO** clearly beats Static_Greedy and OGD_Bandit at every load on all three
  tested metrics (Holm-adjusted *p* ≤ 0.0015). **The Independent MAPPO ablation is not
  covered by this statement.** `_significance_table` in `run_benchmark_phase2.py` only
  ever tests algorithms against `target_alg="MAAN_PPO"`, so Ind-MAPPO vs Static_Greedy and
  Ind-MAPPO vs OGD_Bandit were **never tested**. The summary means point the same way, but
  that is directional only — not statistically confirmed.

> These conclusions differ from earlier versions of this README, which described MAAN_PPO
> as the winner. Those numbers came from a delay model whose units did not cancel: QoS
> success was identically 0.000 and URLLC violation probability identically 1.000 for
> every algorithm at every load, so nothing was actually being compared. See
> [The Model, in Brief](#-the-model-in-brief) for the corrected formulation.

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

The **"Run Demo Benchmark" button does not use these defaults.** It posts a much smaller
job (2 seeds, 3 loads, horizon 120) sized to finish in a few minutes on a small shared-CPU
instance; the accepted ranges are enforced by `ResearchRunRequest` in `backend/main.py`.
`PLOT_DPI` (default 140) can be lowered further to cut memory during plotting.

> **The demo button cannot overwrite the study.** It writes to `backend/outputs_demo/`
> (gitignored, served at `/artifacts_demo`), never to `outputs_phase2/`. Its figures appear
> under the gallery's amber **Demo Run** tab with a banner stating the reduced parameters.
> This used to write straight into `outputs_phase2/`, so one click silently replaced the
> committed 5-load publication figures with 3-load ones and nothing in the UI said so.
> To regenerate the real study, run `python -m src.experiments.run_benchmark_phase2` from
> a terminal — that is the only thing that writes `outputs_phase2/`.

### Frontend — Backend URL

Copy `next_frontend/.env.example` to `.env.local` and set:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

> ⚠️ `NEXT_PUBLIC_*` values are **inlined into the JavaScript bundle at build time**, not
> read at runtime. Changing this on a deployed service requires a **rebuild**, not just a
> restart.

---

## 🚀 PART 6 — Deploying to Render

The two halves run as **two separate Render Web Services** — different runtimes, different
build commands. They cannot share one service. A `render.yaml` blueprint at the repo root
defines both.

### Option A — Blueprint (recommended)

1. Render Dashboard → **New** → **Blueprint** → select this repo.
2. Render reads `render.yaml` and proposes two services.
3. It will prompt for `NEXT_PUBLIC_BACKEND_URL`. You don't know the backend URL yet, so
   put anything, deploy, then come back to step 5.
4. Wait for **`aether-5g-backend`** to go live and copy its URL
   (e.g. `https://aether-5g-backend.onrender.com`).
5. Set `NEXT_PUBLIC_BACKEND_URL` on the **frontend** service to that URL (no trailing
   slash) and **Manual Deploy → Clear build cache & deploy**. The rebuild is required —
   see the warning above.

### Option B — Two services by hand

**Backend** — New → Web Service:

| Field | Value |
|-------|-------|
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health Check Path | `/api/health` |
| Env: `PYTHON_VERSION` | `3.12.7` |
| Env: `OMP_NUM_THREADS` | `1` |

**Frontend** — New → Web Service:

| Field | Value |
|-------|-------|
| Root Directory | `next_frontend` |
| Runtime | Node |
| Build Command | `npm ci && npm run build` |
| Start Command | `npm run start -- --port $PORT` |
| Env: `NODE_VERSION` | `20.18.0` |
| Env: `NEXT_PUBLIC_BACKEND_URL` | the backend's URL |

### Verifying the deploy

```bash
curl https://<your-backend>.onrender.com/api/health
# {"status":"ok","service":"5g-slicing-benchmark","results_available":true,"research_running":false}
```

Then open the frontend URL. It starts in **Simulation Mode**, which needs no backend at
all — so the page rendering is *not* proof the backend is reachable. Click **Result
Plots**: if charts appear, the frontend is genuinely talking to the backend.

### Free-tier constraints worth knowing

| Constraint | Consequence |
|------------|-------------|
| **512 MB RAM** | Importing torch + pandas + scipy + matplotlib costs ~305 MB before serving a request. A benchmark run peaks near **415 MB**. Do not raise the run parameters much, and do not add a second worker. |
| **Ephemeral disk** | Anything written at runtime — new CSVs, new plots — is lost on restart, redeploy, or sleep/wake. The repo therefore **commits** a pre-generated result set so a cold instance has something to serve immediately. |
| **Sleeps after ~15 min idle** | First request after sleep takes ~30–60 s. A research run in progress is killed; on wake the job is reported as `failed` with "Server restarted while this run was in progress" rather than leaving the progress bar spinning forever. |
| **Shared CPU** | The benchmark is single-threaded. A run that takes ~25 s locally can take several minutes there. |
| **CORS** | `allow_origins=["*"]` — deliberately open so any frontend origin works. |

---

## 🔄 Simulation vs Live Mode

| Mode | When it's active | Data source |
|------|-----------------|-------------|
| **Simulation** (default) | Always — no backend needed | Browser generates fake sine-wave telemetry |
| **Live** | After clicking "Run Demo Benchmark" | Backend serves real benchmark results |

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
2. Visit `http://localhost:8000/api/health` in your browser — the `status` field should be `"ok"`
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
cd <path-to-repo>\backend
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

The committed study figures live in `outputs_phase2/plots/` and ship with the repo, so the **Core 14** and **Publication Pack** tabs should never be empty. An empty **Demo Run** tab just means you have not clicked **"Run Demo Benchmark"** yet.

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

Total work scales as `seeds × len(load_scales) × 5 algorithms × horizon`, so halving the
seeds and dropping two loads is roughly a 3× saving.

---

## ⚡ Quick Reference — All Commands

```powershell
# ─── BACKEND (Terminal 1) ─────────────────────────────────

# Navigate to backend
cd <path-to-repo>\backend

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
cd <path-to-repo>\next_frontend

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
[API running]           →         [Site running]     →        Click "Run Demo Benchmark"
[Benchmark running...]                               ←        [Progress bar updates]
[Done → plots saved]              ←                 ←        [Plot gallery opens]
```

---

*Stack: Python 3.12 · FastAPI · Uvicorn · Next.js 14 · Three.js · React Three Fiber · Framer Motion · PyTorch · TailwindCSS*
