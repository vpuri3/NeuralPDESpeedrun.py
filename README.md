# NeuralPDESpeedrun.py

A speedrun for **data-free** PDE solving. Each track is one PDE problem: solve it as fast as possible on one
H100, from the governing equations alone, to a fixed accuracy against a closed-form or high-resolution
reference.

**Goals.** The classical solvers here are reference points, not opponents to beat on every track. We want to
(1) improve hybrid neural methods, measured honestly on wall-clock time-to-accuracy, and (2) find problems
where hybrid neural methods give outsized leverage, i.e. where classical methods are slow. New tracks that
expose such regimes are as valuable as new records. We do not expect one method to win every track; each
submission competes per track.

## Quickstart

```bash
uv sync
uv run python -m speedrun.run --track 01                            # time + score track 01's baseline
uv run python -m speedrun.run_submission --submission submission_PR0001 --gpus 0,1,2,3 \
    --name "PirateNet + SOAP" --kind neural --pr 1                  # one submission on all its tracks
uv run python -m speedrun.records --track all                       # leaderboards, plots, READMEs
```

The environment targets H100 GPUs (NVIDIA driver ≥ 580): `uv sync` installs Python 3.13 and torch 2.15 built for CUDA 13.2;
submissions that use JAX add it with its CUDA 13 wheels (`jax[cuda13]`).

Reference solutions are not in git. They are assets of the GitHub release
[`refs-v1`](https://github.com/vpuri3/NeuralPDESpeedrun.py/releases/tag/refs-v1), downloaded into
`tracks/<track>/reference/ref.npz` the first time a track is scored and checked against the sha256 in its
`meta.json`. `uv run python -m speedrun.make_reference --track <track|all> --device cuda` rebuilds them (and
the figures) from scratch; it downloads the published benchmark data (~300 MB, cached in
`~/.cache/neuralpdespeedrun`). `run_submission` runs one job per GPU at a time, writes each track's
`result.json`, and with `--name` adds the results to the records and regenerates the plots and tables. Tracks
can be named in full (`track03_1DT_Burgers`) or by key (`03`).

## Scoring

Score = wall-clock time of `solve` + `query`, provided the output meets the track's
accuracy bar against the reference. All fields and evaluation points are flattened into one vector u
(`(n_fields, ...)` → `(n_fields · N,)`), and both errors are relative to the reference vector:

- **RelL2** `‖u − u_ref‖₂ / ‖u_ref‖₂ ≤ 0.5%`
- **RelL∞** `max |u − u_ref| / max |u_ref| ≤ 1%`

The bar is the same on every track; a track may exclude a wall margin from RelL∞. Pressure is never scored. Each track has a
wall-clock budget for the solve (`problem.budget_s`, 1000 s unless the track says otherwise), enforced by the
harness: once it is used up, the harness interrupts `solve` and scores what `query` returns for the last
checkpointed state.

## Rules

1. **Data-free.** Use only what `Problem` provides. No reference or solver outputs, precomputed solutions,
   downloaded data, or pretraining of any kind. A closed form used to manufacture forcing (09) is off limits.
2. **Anything else goes.** Any model, ansatz, loss, optimizer, discretization, precision, or mix of neural and
   classical parts. PyTorch, JAX, and Julia are all fine (the entry point is Python; `Problem` callables accept
   torch or JAX arrays), and so are hand-written GPU kernels (Triton, CUDA, ...).
3. **No oracle.** The method decides when to stop (return from `solve`), within the budget; it is scored once,
   on what it returns or last checkpointed. The reference is loaded only after the clock stops.
4. **Timing.** `setup` is untimed and may build and compile, but not train on the problem or compute any part
   of its solution. `solve` and `query` are each timed, with the device (torch and JAX) synchronized before the
   clock stops; `query` must finish in ≤ 5 s. The harness, not the submission, keeps the clock and the budget.
5. **One run, seed 0, on the GPU.** The harness reruns the submission once (untimed, with a 10 s budget) to see
   where it computes: torch ops are counted by device, JAX by the device of its arrays. Other work (Julia,
   custom kernels) is checked by the maintainers.
6. **Official times** are maintainer re-timings on one H100 80GB. Record #0 on each track is an honest but
   deliberately unoptimized classical baseline.

## Submission API

Per track, `tracks/<track>/submissions/submission_PR<NNNN>/solve.py` defines up to three functions:

| function | timed | required | role |
|---|---|---|---|
| `setup(problem) -> state` | no | no | build, allocate, compile; no training, no part of the solution |
| `solve(problem, state, checkpoint) -> state` | yes | yes | train / iterate for as long as you like; `checkpoint(state)` often |
| `query(problem, state) -> array` | yes, ≤ 5 s | unless `solve` returns the output | evaluate the solution at the output points |

```python
def setup(problem):
    return MySolver(problem).compile()               # untimed

def solve(problem, solver, checkpoint):
    while True:                                      # open-ended: the harness enforces the budget
        solver.train(iterations=50)
        checkpoint(solver.snapshot())                # what query() gets if the budget runs out now
        if solver.converged():
            return solver.snapshot()                 # or stop early yourself

def query(problem, state):
    return state.predict(problem.axes)               # torch / JAX / numpy, shape problem.out_shape
```

What the harness does (`speedrun/run.py`):

1. Seeds everything (seed 0), builds the `Problem` on the GPU, imports `solve.py`, calls `setup` (untimed).
2. Starts the clock and calls `solve(problem, state, checkpoint)`.
3. `solve` ends when it returns, or when the track's budget (`problem.budget_s`, 1000 s by default) is used up:
   then the next `checkpoint` call raises `BudgetExceeded`, and if `solve` does not checkpoint within 1 s more,
   a timer interrupts it. The state is the returned one, or else the last checkpointed one; a run that has not
   checkpointed by the budget fails.
4. Synchronizes the device (torch and JAX), stops the clock; then times `query` on that state the same way.
5. Only then loads the reference and scores the output; finally reruns setup + solve (10 s budget) + query
   untimed to check where the work ran.

Notes:

- `checkpoint(state)` only stores a reference, so it is cheap; call it every few iterations. Never modify a
  checkpointed state in place afterwards (JAX arrays are immutable; snapshot torch tensors or parameters).
  `BudgetExceeded` is a `BaseException`, so `except Exception` does not swallow it.
- The state can be any Python object; no module-level globals. Without `setup`, `solve` receives `None`.
- A one-shot solver may define `solve(problem, state)` without `checkpoint`. If it returns an array of shape
  `problem.out_shape`, `query` is skipped (this is how the classical baselines work).
- Learning need not happen on the evaluation points: `query` projects onto them, an array of shape
  `problem.out_shape = (n_fields, len(axis_0), ...)` on the tensor grid of `problem.axes` (`indexing="ij"`).
- `--budget-s` on `speedrun.run` overrides the budget for development runs; records use the track's budget.
  `--no-impose-budget` lifts it entirely (`solve` runs until it returns by itself), for studies; such
  results cannot be recorded.

`Problem` holds the statement and `params`, IC/BC/forcing/coefficient callables (`fns`; array-API, so they take
torch tensors or JAX arrays; on tracks whose initial state is only known as data, `fns["ic"]` holds it as
samples on the output grid), the `domain` bounds, per-axis `periodic` flags, the evaluation `axes` and output
layout, `device`/`dtype` (CUDA, float64; compute in any precision), the track's bar, and its `budget_s`.

**Shared code.** Code a submission reuses across tracks goes in `speedrun/submissions/submission_PR<NNNN>/`, and
the per-track `solve.py` files import it (e.g. `speedrun/submissions/submission_PR0001/pinn.py`). The classical
toolbox `speedrun/numerics/` is available to everyone.

### Submitting

1. **Open a draft PR.** Its number names your directories, zero-padded (PR #42 → `submission_PR0042`).
2. Add your files, run the harness on your GPU, and push as often as you like:
   ```
   speedrun/submissions/submission_PR<NNNN>/        shared code (optional)
   tracks/<track>/submissions/submission_PR<NNNN>/
     solve.py  result.json  previous.json (current record, same GPU)  README.md (method, what changed)
   ```
   ```bash
   uv run python -m speedrun.run_submission --submission submission_PR0042 --gpus 0 \
       --name "my method" --kind hybrid --pr 42       # all tracks the submission covers
   d=tracks/track07_2D_HighContrastDiffusion/submissions/submission_PR0042      # or one track at a time:
   uv run python -m speedrun.run --track 07 --submission $d --out $d/result.json
   ```
3. **Mark the PR ready for review** when you want the maintainers to re-time it.

Results that miss the bar stay as points on the track's accuracy-vs-time plot, but they are not records.

## Tracks

The table lists the best submission on each track. Until some submission meets the track's accuracy bar, the
best is the most accurate one (lowest RelL2); once the bar is met, it is the fastest one that meets it. Classical
baselines (record #0, where a track has one) are reference points listed in each `tracks/<track>/README.md`, not
in this table. Times are pending H100 re-timing. Tracks 04, 05, 06, 11 and 15–18 are the benchmarks of the
PirateNet / SOAP-for-PINNs study (Wang et al.); their references are the published solutions
(`speedrun/external.py`).

Tracks are ordered by dimension: the ODE, then 1D + time, 2D steady, 2D + time, and 3D + time. The folder name
carries the tag (`track03_1DT_Burgers`: 1D + time; `track07_2D_HighContrastDiffusion`: 2D, steady).

Each track below shows its reference solution (animation where time-dependent) and, once it has submissions,
its tracker: RelL2 vs wall-clock time, one point per submission, bar dashed.

<!-- tracks-table:begin -->
| track | best submission | time [s] | RelL2 | RelL∞ | bar met |
|---|---|---|---|---|---|
| 01 VanDerPol (ODE) | — | | | | |
| 02 DampedWave (1D + Time) | — | | | | |
| 03 Burgers (1D + Time) | — | | | | |
| 04 KuramotoSivashinsky (1D + Time) | — | | | | |
| 05 AllenCahn (1D + Time) | — | | | | |
| 06 KdV (1D + Time) | — | | | | |
| 07 HighContrastDiffusion (2D) | — | | | | |
| 08 Helmholtz (2D) | *on hold* | | | | |
| 09 AdvectionDiffusion (2D) | — | | | | |
| 10 Kovasznay (2D) | — | | | | |
| 11 LidDrivenCavity (2D) | — | | | | |
| 12 Burgers (2D + Time) | — | | | | |
| 13 DecayingEddies (2D + Time) | — | | | | |
| 14 CylinderFlow (2D + Time) | *TODO* | | | | |
| 15 GreyScott (2D + Time) | — | | | | |
| 16 GinzburgLandau (2D + Time) | — | | | | |
| 17 RayleighTaylor (2D + Time) | — | | | | |
| 18 KolmogorovFlow (2D + Time) | — | | | | |
| 19 BeltramiFlow (3D + Time) | — | | | | |
<!-- tracks-table:end -->

### 01 VanDerPol (ODE)
Stiff ODE: `y'' − μ(1 − y²) y' + y = 0`, μ = 1000, `y(0) = 2`, `y'(0) = 0`, t ∈ [0, 3000] (almost two relaxation
periods: slow drifts on a time scale ~μ, jumps lasting ~1/μ). Output: y at 3001 uniform times. Reference: Radau
IIA, rtol 1e-12.

<img src="tracks/track01_0DT_VanDerPol/figures/reference.png" width="100%" alt="01 reference">

<img src="tracks/track01_0DT_VanDerPol/figures/accuracy_vs_time.png" width="100%" alt="01 accuracy_vs_time">

### 02 DampedWave (1D + Time)
`u_tt = u_xx − 0.5 u_t` on (t, x) ∈ [0, 4] × [0, 1], `u = 0` at x = 0, 1, a narrow Gaussian pulse
`u(0, x) = exp(−(x − 0.3)²/(2·0.02²))`, `u_t(0, x) = 0`. The two halves cross the domain four times, reflecting
off the walls. Output: 801 t × 1001 x. Reference: exact sine series (closed-form coefficients).

<img src="tracks/track02_1DT_DampedWave/figures/reference.png" width="100%" alt="02 reference">

<img src="tracks/track02_1DT_DampedWave/figures/reference.gif" width="100%" alt="02 reference">

<img src="tracks/track02_1DT_DampedWave/figures/accuracy_vs_time.png" width="100%" alt="02 accuracy_vs_time">

### 03 Burgers (1D + Time)
`u_t + u u_x = ν u_xx`, ν = 1e-4, periodic x ∈ [0, 2), t ∈ [0, 0.5], smooth bump IC steepening toward a shock at
t ≈ 0.53. Output: 500 t × 1024 x. Reference: Fourier + ETDRK4, N = 8192.

<img src="tracks/track03_1DT_Burgers/figures/reference.png" width="100%" alt="03 reference">

<img src="tracks/track03_1DT_Burgers/figures/reference.gif" width="100%" alt="03 reference">

<img src="tracks/track03_1DT_Burgers/figures/accuracy_vs_time.png" width="100%" alt="03 accuracy_vs_time">

### 04 KuramotoSivashinsky (1D + Time)
`u_t + (100/16) u u_x + (100/16²) u_xx + (100/16⁴) u_xxxx = 0`, x ∈ [0, 2π] periodic, t ∈ [0, 1], chaotic regime;
the initial state is given as samples. Output: 250 t × 512 x. Reference: published data.

<img src="tracks/track04_1DT_KuramotoSivashinsky/figures/reference.png" width="100%" alt="04 reference">

<img src="tracks/track04_1DT_KuramotoSivashinsky/figures/reference.gif" width="100%" alt="04 reference">

### 05 AllenCahn (1D + Time)
`u_t − 1e-4 u_xx + 5u³ − 5u = 0`, x ∈ [−1, 1] periodic, t ∈ [0, 1], `u(0, x) = x² cos(πx)`. Output: 200 t × 512 x.
Reference: published data.

<img src="tracks/track05_1DT_AllenCahn/figures/reference.png" width="100%" alt="05 reference">

<img src="tracks/track05_1DT_AllenCahn/figures/reference.gif" width="100%" alt="05 reference">

### 06 KdV (1D + Time)
`u_t + u u_x + 0.022² u_xxx = 0`, x ∈ [−1, 1] periodic, t ∈ [0, 1], `u(0, x) = cos(πx)`; the cosine breaks into
solitons. Output: 200 t × 512 x. Reference: published data.

<img src="tracks/track06_1DT_KdV/figures/reference.png" width="100%" alt="06 reference">

<img src="tracks/track06_1DT_KdV/figures/reference.gif" width="100%" alt="06 reference">

### 07 HighContrastDiffusion (2D)
`−∇·(a∇u) = 18` on [0, 1]², `u = 0` on the boundary, `a = 1e3` in a 3 × 3 array of square inclusions, 1
elsewhere. The inclusion corners make u singular (~ρ^(2/3)). Output: 256² cell centers. Reference: FV on 3840²
and 5376², Richardson-extrapolated at the measured order 1.34.

<img src="tracks/track07_2D_HighContrastDiffusion/figures/reference.png" width="100%" alt="07 reference">

<img src="tracks/track07_2D_HighContrastDiffusion/figures/accuracy_vs_time.png" width="100%" alt="07 accuracy_vs_time">

### 08 Helmholtz (2D): on hold
The original setup (lossless Dirichlet box with a slow lens, ω = 40) is resonant. It needs absorption (sponge,
PML, or impedance BC) first.

### 09 AdvectionDiffusion (2D)
`−ε Δu + u_x + u_y = f` on [0, 1]², ε = 1e-3, `u = 0` on the boundary, with boundary layers of width ε at x = 1
and y = 1. Output: 257² Chebyshev–Lobatto grid. Reference: closed form (f was manufactured from it; rule 1).

<img src="tracks/track09_2D_AdvectionDiffusion/figures/reference.png" width="100%" alt="09 reference">

<img src="tracks/track09_2D_AdvectionDiffusion/figures/accuracy_vs_time.png" width="100%" alt="09 accuracy_vs_time">

### 10 Kovasznay (2D)
Steady incompressible NS, Re = 40, on [−0.5, 2] × [−0.5, 1.5], with the Kovasznay velocity as Dirichlet data on the
whole boundary. Output: (u, v) on 251 × 201. Reference: closed form.

<img src="tracks/track10_2D_Kovasznay/figures/reference.png" width="100%" alt="10 reference">

<img src="tracks/track10_2D_Kovasznay/figures/accuracy_vs_time.png" width="100%" alt="10 accuracy_vs_time">

### 11 LidDrivenCavity (2D)
Steady NS, Re = 5000, in [0, 1]², lid `u = 1 − cosh(50(x − 0.5))/cosh(25)` on y = 1, no-slip elsewhere. Output: (u, v)
on 256². RelL∞ is scored at points ≥ 0.05 from the walls. Reference: published data.

<img src="tracks/track11_2D_LidDrivenCavity/figures/reference.png" width="100%" alt="11 reference">

### 12 Burgers (2D + Time)
`u_t + (u·∇)u = ν Δu`, ν = 1e-3, periodic [−0.25, 0.75)², t ∈ [0, 0.5], `u = v = sin(2πx) sin(2πy)` on [0, 0.5]²,
0 elsewhere; shocks of width ~ν form. Output: (u, v) at 11 t × 512². Reference: Fourier + ETDRK4, 2048².

<img src="tracks/track12_2DT_Burgers/figures/reference.png" width="100%" alt="12 reference">

<img src="tracks/track12_2DT_Burgers/figures/reference.gif" width="100%" alt="12 reference">

<img src="tracks/track12_2DT_Burgers/figures/accuracy_vs_time.png" width="100%" alt="12 accuracy_vs_time">

### 13 DecayingEddies (2D + Time)
Unsteady NS, ν = 0.05, periodic [0, 2π)², t ∈ [0, 5], from `u = sin 5y + cos 3x cos 4y`,
`v = cos 5x + 0.75 sin 3x sin 4y` (Walsh 1992). Output: (u, v) at 11 t × 128². Reference: closed form.

<img src="tracks/track13_2DT_DecayingEddies/figures/reference.png" width="100%" alt="13 reference">

<img src="tracks/track13_2DT_DecayingEddies/figures/reference.gif" width="100%" alt="13 reference">

<img src="tracks/track13_2DT_DecayingEddies/figures/accuracy_vs_time.png" width="100%" alt="13 accuracy_vs_time">

### 14 CylinderFlow (2D + Time): TODO
Flow past a cylinder; see `tracks/track14_2DT_CylinderFlow/TODO.md`.

### 15 GreyScott (2D + Time)
`u_t = 0.2 Δu + 40(1 − u) − 1000 u v²`, `v_t = 0.1 Δv − 100 v + 1000 u v²` on [−1, 1]² periodic, t ∈ [0, 2], initial
state given as samples. Output: (u, v) at 100 t × 200². Reference: published data.

<img src="tracks/track15_2DT_GreyScott/figures/reference.png" width="100%" alt="15 reference">

<img src="tracks/track15_2DT_GreyScott/figures/reference.gif" width="100%" alt="15 reference">

### 16 GinzburgLandau (2D + Time)
Complex Ginzburg–Landau for A = u + iv, ε = 0.004, k = 10, on [−1, 1]² periodic, t ∈ [0, 1], initial state given as
samples. Output: (u, v) at 100 t × 200². Reference: published data.

<img src="tracks/track16_2DT_GinzburgLandau/figures/reference.png" width="100%" alt="16 reference">

<img src="tracks/track16_2DT_GinzburgLandau/figures/reference.gif" width="100%" alt="16 reference">

### 17 RayleighTaylor (2D + Time)
Boussinesq Rayleigh–Taylor instability, Ra = 1e6, Pr = 0.71, on [0, 1] × [0, 2], t ∈ [0, 8.5], initial state given as
samples. Output: (u, v, T) at 34 t on a 100 × 200 grid. Reference: published data.

<img src="tracks/track17_2DT_RayleighTaylor/figures/reference.png" width="100%" alt="17 reference">

<img src="tracks/track17_2DT_RayleighTaylor/figures/reference.gif" width="100%" alt="17 reference">

### 18 KolmogorovFlow (2D + Time)
Forced 2D NS, Re = 1e4, forcing `2 sin(4πy)` in x, on [0, 1)² periodic, t ∈ [0, 1.96], initial state given as
samples; turbulent. Output: (u, v) at 49 t × 256². Reference: published data.

<img src="tracks/track18_2DT_KolmogorovFlow/figures/reference.png" width="100%" alt="18 reference">

<img src="tracks/track18_2DT_KolmogorovFlow/figures/reference.gif" width="100%" alt="18 reference">

### 19 BeltramiFlow (3D + Time)
3D unsteady Navier–Stokes, Re = 1, on [−1, 1]³ × [0, 1]: the Ethier–Steinman Beltrami flow (a = d = 1). Initial and
Dirichlet velocity on all six faces come from the exact solution. Output: (u, v, w) at t = 0.25, 0.5, 0.75, 1 on 33³.
Reference: closed form.

<img src="tracks/track19_3DT_BeltramiFlow/figures/reference.png" width="100%" alt="19 reference">

## Layout

```
speedrun/
  problem.py  metrics.py  run.py (harness)  run_submission.py  devicecheck.py  make_reference.py  records.py
  tracks.py (registry, reference download)  external.py (published benchmark data)  plotting.py
  numerics/       classical toolbox: Chebyshev, Fourier/ETDRK4, Krylov, FV + multigrid, PN–PN-2 spectral elements
  submissions/    code shared across tracks by a submission: submission_PR<NNNN>/
tracks/<track>/
  problem.py  reference.py  baseline.py (record #0)  records.json  README.md
  figures/  reference/ (meta.json; ref.npz downloaded from the release)  data/ (initial states given as data)
  submissions/submission_PR<NNNN>/
```
