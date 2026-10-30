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

Tracks can be named in full (`track07A_CavityRe100`) or by key (`07A`).

## Scoring

Score = wall-clock time of `solve` + `query`, provided the output meets the track's accuracy bar against the
reference on all fields and evaluation points:

- **RelL2** `‖u − u_ref‖₂ / ‖u_ref‖₂ ≤ 1e-4`
- **L∞** `max |u − u_ref| ≤ 1e-3`

Tracks may override the bar or exclude a wall margin from L∞ (noted per track). Pressure is never scored.

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

`Problem` holds the statement and `params`, IC/BC/forcing/coefficient callables (`fns`, plus the domain map on
curved domains), the `domain`, the evaluation `axes` and output layout, `device`/`dtype` (CUDA, float64; compute
in any precision), and the track's bar.

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
| 01 ThirdOrderODE | Chebyshev N=16 | 1.5 ms |
| 02A Wave | sine spectral N=32 | 1.2 ms |
| 02B DampedWave | sine spectral N=512 | 1.9 ms |
| 03 HighContrastDiffusion | FV 3328² + MG-PCG | 8.39 s |
| 04A Helmholtz | *on hold* | — |
| 04B AdvectionDiffusion | Chebyshev N=96 | 1.16 s |
| 05A Burgers1D | Fourier N=1024 + ETDRK4 | 0.23 s |
| 05B Burgers2D | Fourier 1024² + ETDRK4 | 7.83 s |
| 05C KuramotoSivashinsky | Fourier N=256 + ETDRK4 | 0.46 s |
| 06A Kovasznay | PN–PN-2 N=16, Newton | 0.19 s |
| 06B KovasznayCurved | PN–PN-2 N=16, Newton | 0.22 s |
| 07A CavityRe100 | *TODO* | — |
| 07B CavityRe1000 | *TODO* | — |
| 08 StokesCavity | PN–PN-2 N=192, Uzawa | 0.22 s |
| 09 DecayingEddies | Fourier N=32 + ETDRK4 | 0.14 s |
| 10 CylinderFlow | *TODO* | — |

### 01 ThirdOrderODE
`u''' = cos(πx)` on [0, 1], `u(0) = 0`, `u(1) = −1`, `u'(1) = 1`. Output: u at 1001 uniform points. Reference:
closed form.

<p><img src="tracks/track01_ThirdOrderODE/figures/reference.png" width="54%" alt="01 reference"> <img src="tracks/track01_ThirdOrderODE/figures/accuracy_vs_time.png" width="42%" alt="01 RelL2 vs time"></p>

### 02A Wave
`u_tt = u_xx` on (t, x) ∈ [0, 1]², `u = 0` at x = 0, 1, `u(0, x) = x(1 − x)`, `u_t(0, x) = 0`. Output: 201 × 201
(t, x). Reference: d'Alembert (exact).

<p><img src="tracks/track02A_Wave/figures/reference.png" width="54%" alt="02A reference"> <img src="tracks/track02A_Wave/figures/accuracy_vs_time.png" width="42%" alt="02A RelL2 vs time"></p>

### 02B DampedWave
`u_tt = 9 u_xx − 18 u_t` on [0, 1]², Dirichlet walls, `u(0, x) = x(1 − x)`, `u_t(0, x) = 1 − 2x`. The initial
velocity does not vanish at the walls, so u_t jumps along the corner characteristics. Output: 201 × 201 (t, x).
Reference: exact modal sine series, K = 32768.

<p><img src="tracks/track02B_DampedWave/figures/reference.png" width="54%" alt="02B reference"> <img src="tracks/track02B_DampedWave/figures/accuracy_vs_time.png" width="42%" alt="02B RelL2 vs time"></p>

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
`u_t + u u_x + u_xx + ν u_xxxx = 0`, ν = 0.01, periodic x ∈ [−π, π), t ∈ [0, 0.1], Gaussian IC (σ = 0.2). Output:
1000 t × 256 x. Reference: Fourier + ETDRK4, N = 1024.

<p><img src="tracks/track05C_KuramotoSivashinsky/figures/reference.png" width="54%" alt="05C reference"> <img src="tracks/track05C_KuramotoSivashinsky/figures/accuracy_vs_time.png" width="42%" alt="05C RelL2 vs time"></p>

### 06A Kovasznay
Steady incompressible NS, Re = 40, on [−0.5, 2] × [−0.5, 1.5], with the Kovasznay velocity as Dirichlet data on the
whole boundary. Output: (u, v) on 251 × 201. Reference: closed form.

<p><img src="tracks/track06A_Kovasznay/figures/reference.png" width="54%" alt="06A reference"> <img src="tracks/track06A_Kovasznay/figures/accuracy_vs_time.png" width="42%" alt="06A RelL2 vs time"></p>

### 06B KovasznayCurved
The same flow on a trapezoid with a parabolic top: the image of [−1, 1]² under `x = r(3 + s)/4`,
`y = (1 + s)/2 · (1 − r²/2) − (1 − s)/4` (`problem.fns["map"]`). Output: (u, v) at the mapped 201 × 201 (r, s) grid.
Reference: closed form.

<p><img src="tracks/track06B_KovasznayCurved/figures/reference.png" width="54%" alt="06B reference"> <img src="tracks/track06B_KovasznayCurved/figures/accuracy_vs_time.png" width="42%" alt="06B RelL2 vs time"></p>

### 07A CavityRe100 / 07B CavityRe1000: TODO
Unsteady NS in [−1, 1]², lid `u = 1` on y = 1 (corners included), no-slip elsewhere, at rest at t = 0, t ∈ [0, 5],
Re = 100 / 1000. Output: (u, v) at t = 0.5, 1, …, 5 on 129². The reference is not finalized (see PLAN.md).

### 08 StokesCavity
Steady Stokes flow in the 07 cavity. Output: (u, v) on 129². Reference: PN–PN-2, N = 768. **Bar: RelL2 ≤ 1e-3**,
L∞ ≤ 1e-3 at points ≥ 0.05 from the walls: the lid corners are singular, which limits every reference.

<p><img src="tracks/track08_StokesCavity/figures/reference.png" width="54%" alt="08 reference"> <img src="tracks/track08_StokesCavity/figures/accuracy_vs_time.png" width="42%" alt="08 RelL2 vs time"></p>

### 09 DecayingEddies
Unsteady NS, ν = 0.05, periodic [0, 2π)², t ∈ [0, 5], from `u = sin 5y + cos 3x cos 4y`,
`v = cos 5x + 0.75 sin 3x sin 4y` (Walsh 1992). Output: (u, v) at 11 t × 128². Reference: closed form.

<p><img src="tracks/track09_DecayingEddies/figures/reference_t5.0.png" width="54%" alt="09 reference"> <img src="tracks/track09_DecayingEddies/figures/accuracy_vs_time.png" width="42%" alt="09 RelL2 vs time"></p>

### 10 CylinderFlow: TODO
Flow past a cylinder; see `tracks/track10_CylinderFlow/TODO.md`.

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
