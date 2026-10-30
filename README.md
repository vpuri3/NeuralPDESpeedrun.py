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
uv run python -m speedrun.make_reference --track all --device cuda   # references + figures
uv run python -m speedrun.run --track 01                            # time + score track 01's baseline
uv run python -m speedrun.records --track all                       # leaderboards, plots, per-track READMEs
```

Tracks can be named in full (`track05B_Burgers2D`) or by key (`05B`).

## Scoring

Score = wall-clock time of `solve` + `query`, provided the output meets the track's accuracy bar against the
reference on all fields and evaluation points:

- **RelL2** `‖u − u_ref‖₂ / ‖u_ref‖₂ ≤ 1e-4`
- **L∞** `max |u − u_ref| ≤ 1e-3`

Tracks may override the bar or exclude a wall margin from L∞. Pressure is never scored.

## Rules

1. **Data-free.** Use only what `Problem` provides. No reference or solver outputs, precomputed solutions,
   downloaded data, or pretraining of any kind. A closed form used to manufacture forcing (04B) is off limits.
2. **Anything else goes.** Any model, ansatz, loss, optimizer, discretization, precision, or mix of neural and
   classical parts. Python, JAX, and Julia are all fine (the entry point is Python; call into JAX or Julia from
   it), and so are hand-written GPU kernels (Triton, CUDA, ...).
3. **No oracle.** The method decides when to stop; it is scored once, on what it returns. The reference is
   loaded only after the clock stops.
4. **Timing.** `warmup` is untimed and may build and compile, but not train on the problem or compute any part
   of its solution. `solve` and `query` are each timed with GPU syncs; `query` must finish in ≤ 5 s.
5. **One run, seed 0, on the GPU.** The harness reruns the submission once (untimed) and counts torch ops by
   device. Work it cannot see (JAX, Julia, custom kernels) is checked by the maintainers.
6. **Official times** are maintainer re-timings on one H100 80GB. Record #0 on each track is an honest but
   deliberately unoptimized classical baseline.

## Submission API

Per track, `tracks/<track>/submissions/submission_PR<NNNN>/solve.py` defines:

```python
def warmup(problem):              # optional, untimed: build + compile
    return MySolver(problem).compile()

def solve(problem, solver):       # timed: train / iterate / solve
    solver.train()
    return solver

def query(problem, solver):       # timed separately, <= 5 s: evaluate at the output points
    return solver.predict(problem.axes)    # torch.Tensor of shape problem.out_shape
```

- All state lives in the solver object; no module-level globals. Without `warmup`, `solver` is `None`.
- Learning need not happen on the evaluation points: `query` projects onto them, a tensor of shape
  `problem.out_shape = (n_fields, len(axis_0), ...)` on the tensor grid of `problem.axes` (`indexing="ij"`).
- If `solve` returns that tensor directly, `query` is skipped (this is how the classical baselines work).

`Problem` holds the statement and `params`, IC/BC/forcing/coefficient callables (`fns`), the `domain` bounds,
per-axis `periodic` flags, the evaluation `axes` and output layout, `device`/`dtype` (CUDA, float64; compute in
any precision), and the track's bar.

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
   d=tracks/track03_HighContrastDiffusion/submissions/submission_PR0042
   uv run python -m speedrun.run --track 03 --submission $d --out $d/result.json
   uv run python -m speedrun.records --track 03 --add $d/result.json --name "my method" --kind hybrid --pr 42
   ```
3. **Mark the PR ready for review** when you want the maintainers to re-time it.

Results that miss the bar stay as points on the track's accuracy-vs-time plot, but they are not records.

## Tracks

Each track shows its reference solution (left) and its tracker (right): RelL2 vs wall-clock time, one point per
submission, bar dashed. Record #0 times are on an RTX 2080 Ti (fp64, GPU-checked) pending H100 re-timing.
`submission_PR0001` on every active track is a vanilla PINN (standard MLP + Adam, 1000 s budget). Each
`tracks/<track>/README.md` has the full statement and leaderboard.

| track | record #0 | time |
|---|---|---|
| 01 VanDerPol | LSODA, rtol 1e-8 (CPU) | 37 ms |
| 02 DampedWave | sine modes N=256, exact in time | 2.9 ms |
| 03 HighContrastDiffusion | FV 3328² + MG-PCG | 8.36 s |
| 04A Helmholtz | *on hold* | — |
| 04B AdvectionDiffusion | Chebyshev N=96 | 1.16 s |
| 05A Burgers1D | Fourier N=1024 + ETDRK4 | 0.23 s |
| 05B Burgers2D | Fourier 1024² + ETDRK4 | 7.83 s |
| 05C KuramotoSivashinsky | Fourier N=256 + ETDRK4 | 6.24 s |
| 06 Kovasznay | PN–PN-2 N=16, Newton | 0.19 s |
| 07 CavityRe1000 | *TODO* | — |
| 08 DecayingEddies | Fourier N=32 + ETDRK4 | 0.13 s |
| 09 CylinderFlow | *TODO* | — |
| 10 BeltramiFlow | *TODO* | — |

### 01 VanDerPol
Stiff ODE: `y'' − μ(1 − y²) y' + y = 0`, μ = 1000, `y(0) = 2`, `y'(0) = 0`, t ∈ [0, 3000] (almost two relaxation
periods: slow drifts on a time scale ~μ, jumps lasting ~1/μ). Output: y at 3001 uniform times. Reference: Radau
IIA, rtol 1e-12.

<p><img src="tracks/track01_VanDerPol/figures/reference.png" width="54%" alt="01 reference"> <img src="tracks/track01_VanDerPol/figures/accuracy_vs_time.png" width="42%" alt="01 RelL2 vs time"></p>

### 02 DampedWave
`u_tt = u_xx − 0.5 u_t` on (t, x) ∈ [0, 4] × [0, 1], `u = 0` at x = 0, 1, a narrow Gaussian pulse
`u(0, x) = exp(−(x − 0.3)²/(2·0.02²))`, `u_t(0, x) = 0`. The two halves cross the domain four times, reflecting
off the walls. Output: 801 t × 1001 x. Reference: exact sine series (closed-form coefficients).

<p><img src="tracks/track02_DampedWave/figures/reference.png" width="54%" alt="02 reference"> <img src="tracks/track02_DampedWave/figures/accuracy_vs_time.png" width="42%" alt="02 RelL2 vs time"></p>

### 03 HighContrastDiffusion
`−∇·(a∇u) = 18` on [0, 1]², `u = 0` on the boundary, `a = 1e3` in a 3 × 3 array of square inclusions, 1
elsewhere. The inclusion corners make u singular (~ρ^(2/3)). Output: 256² cell centers. Reference: FV on 3840²
and 5376², Richardson-extrapolated at the measured order 1.34.

<p><img src="tracks/track03_HighContrastDiffusion/figures/reference.png" width="54%" alt="03 reference"> <img src="tracks/track03_HighContrastDiffusion/figures/accuracy_vs_time.png" width="42%" alt="03 RelL2 vs time"></p>

### 04A Helmholtz: on hold
The original setup (lossless Dirichlet box with a slow lens, ω = 40) is resonant. It needs absorption (sponge,
PML, or impedance BC) first.

### 04B AdvectionDiffusion
`−ε Δu + u_x + u_y = f` on [0, 1]², ε = 1e-3, `u = 0` on the boundary, with boundary layers of width ε at x = 1
and y = 1. Output: 257² Chebyshev–Lobatto grid. Reference: closed form (f was manufactured from it; rule 1).

<p><img src="tracks/track04B_AdvectionDiffusion/figures/reference.png" width="54%" alt="04B reference"> <img src="tracks/track04B_AdvectionDiffusion/figures/accuracy_vs_time.png" width="42%" alt="04B RelL2 vs time"></p>

### 05A Burgers1D
`u_t + u u_x = ν u_xx`, ν = 1e-4, periodic x ∈ [0, 2), t ∈ [0, 0.5], smooth bump IC steepening toward a shock at
t ≈ 0.53. Output: 500 t × 1024 x. Reference: Fourier + ETDRK4, N = 8192.

<p><img src="tracks/track05A_Burgers1D/figures/reference.png" width="54%" alt="05A reference"> <img src="tracks/track05A_Burgers1D/figures/accuracy_vs_time.png" width="42%" alt="05A RelL2 vs time"></p>

### 05B Burgers2D
`u_t + (u·∇)u = ν Δu`, ν = 1e-3, periodic [−0.25, 0.75)², t ∈ [0, 0.5], `u = v = sin(2πx) sin(2πy)` on [0, 0.5]²,
0 elsewhere; shocks of width ~ν form. Output: (u, v) at 11 t × 512². Reference: Fourier + ETDRK4, 2048².

<p><img src="tracks/track05B_Burgers2D/figures/reference_t0.50.png" width="54%" alt="05B reference"> <img src="tracks/track05B_Burgers2D/figures/accuracy_vs_time.png" width="42%" alt="05B RelL2 vs time"></p>

### 05C KuramotoSivashinsky
`u_t + u u_x + u_xx + ν u_xxxx = 0`, ν = 0.01, x ∈ [−π, π) periodic, t ∈ [0, 1], Gaussian IC (σ = 0.2).
Output: 1000 t × 256 x. Reference: Fourier + ETDRK4, N = 1024, 64 steps per snapshot.

<p><img src="tracks/track05C_KuramotoSivashinsky/figures/reference.png" width="54%" alt="05C reference"> <img src="tracks/track05C_KuramotoSivashinsky/figures/accuracy_vs_time.png" width="42%" alt="05C RelL2 vs time"></p>

### 06 Kovasznay
Steady incompressible NS, Re = 40, on [−0.5, 2] × [−0.5, 1.5], with the Kovasznay velocity as Dirichlet data on the
whole boundary. Output: (u, v) on 251 × 201. Reference: closed form.

<p><img src="tracks/track06_Kovasznay/figures/reference.png" width="54%" alt="06 reference"> <img src="tracks/track06_Kovasznay/figures/accuracy_vs_time.png" width="42%" alt="06 RelL2 vs time"></p>

### 07 CavityRe1000: TODO
Unsteady NS in [−1, 1]², Re = 1000, lid `u = 1` on y = 1 (corners included), no-slip elsewhere, at rest at
t = 0, t ∈ [0, 5]. Output: (u, v) at t = 0.5, 1, …, 5 on 129². The corner singularities limit the reference;
it is not finalized (see PLAN.md).

### 08 DecayingEddies
Unsteady NS, ν = 0.05, periodic [0, 2π)², t ∈ [0, 5], from `u = sin 5y + cos 3x cos 4y`,
`v = cos 5x + 0.75 sin 3x sin 4y` (Walsh 1992). Output: (u, v) at 11 t × 128². Reference: closed form.

<p><img src="tracks/track08_DecayingEddies/figures/reference_t5.0.png" width="54%" alt="08 reference"> <img src="tracks/track08_DecayingEddies/figures/accuracy_vs_time.png" width="42%" alt="08 RelL2 vs time"></p>

### 09 CylinderFlow: TODO
Flow past a cylinder; see `tracks/track09_CylinderFlow/TODO.md`.

### 10 BeltramiFlow: TODO
3D unsteady Navier–Stokes, Re = 1, on [−1, 1]³ × [0, 1] with the Ethier–Steinman exact solution (as in
arXiv:2402.10680, §4.2); see `tracks/track10_BeltramiFlow/TODO.md`.

## Layout

```
speedrun/
  problem.py  metrics.py  run.py (harness)  devicecheck.py  make_reference.py  records.py  tracks.py
  numerics/       classical toolbox: Chebyshev, Fourier/ETDRK4, Krylov, FV + multigrid, PN–PN-2 spectral elements
  submissions/    code shared across tracks by a submission: submission_PR<NNNN>/
tracks/<track>/
  problem.py  reference.py  baseline.py (record #0)  records.json  README.md
  figures/  reference/  submissions/submission_PR<NNNN>/
```
