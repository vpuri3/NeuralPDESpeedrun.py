# SPEEDRUN.md — rules, status, and plan of action

## 1. Goal

Find out whether neural or hybrid neural/classical methods can match or beat traditional PDE solvers
on wall-clock **time-to-accuracy**, using **only the governing equations**: no solution data of any kind.

Each track is a single PDE solve. The score is the wall-clock time of `solve(problem)` on one
**H100 80GB**, subject to:

| metric | definition (over all fields and all evaluation points) | tolerance |
|---|---|---|
| RelL2 | `‖u − u_ref‖₂ / ‖u_ref‖₂` | **≤ 5e-3** |
| L∞ | `max |u − u_ref|` (absolute) | **≤ 1e-2** |

`u_ref` is a closed-form solution where one exists, otherwise a high-resolution classical solve
(FEM/FV, Fourier or Chebyshev spectral), evaluated at the fixed points given in `problem.axes`.

## 2. Rules

1. **Fully data-free.** A submission can use only what `Problem` contains: equations and parameters
   (`statement`, `params`), domain, IC/BC/forcing/coefficient callables (`fns`), and the evaluation points
   (`axes`). Not allowed: reference or solver outputs, precomputed solutions, pretrained weights fitted to
   solutions of this problem, or downloaded data. Pretraining on *other* data is also out.
   - On tracks whose forcing was manufactured from a closed form (4B), the solver must not use
     knowledge of that closed form.
2. **Free choice of everything else.** Model, ansatz, loss, optimizer, discretization, precision, and
   kernels are all open. Classical solvers may be used inside the method (hybrid is fine), with no
   restriction on mixing neural and classical parts.
3. **No oracle stopping.** The method decides on its own when to stop (residual, loss, iteration
   budget…). It is scored once, on whatever `solve` returns. It cannot see the reference, and the harness
   loads the reference only after the clock stops.
4. **Timing.** `warmup(problem)` (optional) is untimed and may compile/autotune on anything. It must not
   leave solution state behind for `solve`: no caching of solved fields or of weights trained on this
   problem. `solve` starts from scratch. The clock wraps `solve` with CUDA syncs on both ends.
5. **One solve per record.** Seed 0. There is no multi-instance evaluation.
6. **Submitting a record (PR).** Develop on any GPU. In the PR, include:
   - the submission file `tracks/<track>/records/<YYYYMMDD>_<handle>_<short-name>.py`
   - the harness result json for **your method and the current record**, both run on your GPU
     (`uv run python -m speedrun.run --track <t> --submission <file> --out <json>`), with GPU type,
     time, RelL2, L∞, and peak memory
   - an entry appended to `tracks/<track>/records.json`
   - a short write-up (method, what changed vs. previous record)

   Maintainers re-time on a single H100 80GB before merging. The official time is that H100 number.
7. **The baseline classical solvers are deliberately not aggressively optimized.** They exist to give a
   real, honest bar and a record #0, not to be the best classical method.

## 3. Repository layout

```
speedrun/
  problem.py         Problem dataclass (the only input a submission receives)
  metrics.py         RelL2 / L∞ / pass criteria
  run.py             timing + scoring harness     (python -m speedrun.run --track trackK [--submission f.py])
  make_reference.py  generate references + plots  (python -m speedrun.make_reference --track trackK|all)
  tracks.py          track registry, reference I/O (tracks/<t>/reference/ref.npz + meta.json)
  plotting.py        reference figures / gifs
  numerics/          classical building blocks, all PyTorch:
    cheb.py            Chebyshev nodes, diff. matrices, barycentric interp, Clenshaw-Curtis
    fourier.py         wavenumbers, 2/3 dealiasing, ETDRK4, DST-I sine transforms
    periodic.py        Fourier pseudo-spectral Burgers 1D/2D, KS
    krylov.py          PCG, restarted right-preconditioned GMRES (complex)
    fv_diffusion.py    cell-centered FV + aggregation-multigrid PCG
    streamfunction.py  Chebyshev streamfunction Newton solver for steady (Navier-)Stokes
tracks/
  cavity_common.py   shared regularized-lid cavity definition (tracks 7A, 7B, 8)
  trackK/
    problem.py     make_problem(device, dtype) -> Problem
    reference.py   compute(device) -> (u_ref, meta, extras);  plot(...)
    baseline.py    record #0: classical solver, `solve(problem)`
    records.json   leaderboard
    figures/       reference.png / .gif, activity.png
    reference/     ref.npz + meta.json
```

## 4. Tracks

| track | problem | eval points / output shape | reference | baseline (record #0) |
|---|---|---|---|---|
| 1 | u''' = cos(πx) on [0,1], u(0)=0, u(1)=−1, u'(1)=1 (NeuralPDE.jl) | x: 1001 → (1,1001) | closed form | Chebyshev N=16 |
| 2A | wave u_tt = u_xx, Dirichlet, u0 = x(1−x) (NeuralPDE.jl) | t×x 201×201 | d'Alembert (exact) | DST spectral N=32, exact modal time |
| 2B | damped wave u_tt = 9u_xx − 18u_t, u0 = x(1−x), u_t0 = 1−2x (NeuralPDE.jl) | t×x 201×201 | exact sine series, K=32768 | DST spectral N=64 |
| 3 | −∇·(a∇u) = 18, 3×3 inclusions a=1e3 (contrast 1e3) | 256² cell centers | FV 2304² + MG-PCG | FV 256² + MG-PCG |
| 4A | Helmholtz, heterogeneous medium (**TODO**, see §6) | 257² | sine pseudo-spectral 768² + GMRES | same at 256² |
| 4B | −εΔu + u_x + u_y = f, ε=1e-3 (Pe ~ 1e3), boundary layers | 257² Chebyshev–Lobatto | closed form g(x)g(y) | 2D Chebyshev N=80, dense LU |
| 5A | Burgers 1D, ν=1e-4, μ=0.6, T=0.5 (SNF-ROM §6.3) | 500 t × 1024 x | Fourier+ETDRK4, N=8192 | Fourier+ETDRK4 on N=1024 |
| 5B | Burgers 2D (vector), ν=1e-3, μ=1, T=0.5 (SNF-ROM §6.4) | 11 t × 512² × 2 fields | Fourier+ETDRK4 1024², dt=5e-4 | Fourier+ETDRK4 512², dt=1e-3 |
| 5C | Kuramoto–Sivashinsky, ν=0.01, T=0.1 (SNF-ROM §6.5) | 1000 t × 256 x | Fourier+ETDRK4, N=1024 | Fourier+ETDRK4 on N=256 |
| 6 | Kovasznay flow, Re=20 | 151×201 × (u,v) | closed form | Chebyshev streamfunction N=16 |
| 7A | lid-driven cavity, regularized lid, Re=100 | 129² × (u,v) | Chebyshev streamfunction N=48 | N=24 |
| 7B | lid-driven cavity, regularized lid, Re=1000 | 129² × (u,v) | Chebyshev streamfunction N=80 | N=32 |
| 8 | steady Stokes cavity, regularized lid | 129² × (u,v) | Chebyshev streamfunction N=64 | N=16 |

Pressure is never scored. In NS/Stokes it is defined only up to a constant.

### Problem-spec decisions to confirm
- **5A.** The paper prints ν=1e-4 and IC "0 otherwise". The NeuralROMs.jl datagen script uses ν=1e-3, tspan
  (0,1), domain [0,2.25), and IC "1 otherwise". I took **ν=1e-4, T=0.5, domain [0,2), N=1024 (paper)**
  with the **continuous IC (1 otherwise, from the code)**. The shock forms at t* = 1/(πμ) ≈ 0.53, just after T.
  The 1024 grid resolves the solution: N=4096 vs N=8192 agree to 2e-8.
- **5A/5B/5C.** These are single instances (μ=0.6, μ=1.0, fixed IC) because each record is one solve.
- **5B.** The 512² output grid is the paper's. Shock width ~ν ≈ 1e-3 is about half the 512 grid spacing,
  so check the 512-vs-1024 difference in `reference/meta.json`. If the baseline fails the tolerance
  against the 1024² reference, either raise the reference to 2048² or accept that the classical bar
  needs 1024².
- **2B.** Equation as implemented in the NeuralPDE.jl code (u_xx = u_tt/v² + b u_t, v=3, b=2). The
  page's displayed equation differs.
- **7/8.** Regularized lid u = 16x²(1−x)², so pointwise L∞ is meaningful. The classic u=1 lid has corner
  singularities.
- **3, 4B.** Forcing scaled so max|u| ≈ 1, which makes the absolute L∞ tolerance meaningful.

## 5. Status (as of this hand-off)

Everything below was executed with a **numpy-backed torch shim** in a sandbox without PyTorch
(float64 throughout). Same code paths, but it has not yet run under real torch/CUDA.

| track | problem/reference/baseline written | reference generated | baseline passes | notes |
|---|---|---|---|---|
| 1 | ✅ | ✅ | ✅ (relL2 2e-13) | |
| 2A | ✅ | ✅ | ✅ (7e-5) | |
| 2B | ✅ | ✅ | ✅ (3.8e-4) | |
| 3 | ✅ | ❌ (was running) | ? | MG-PCG works: 51 its @256², 98 @768², ~200 expected @2304² |
| 4A | ❌ | ❌ | ❌ | spec + design in §6 |
| 4B | ✅ | ✅ | ✅ (5.8e-4 @N=80) | N=64 fails, N=72 passes |
| 5A | ✅ | ✅ | ✅ (4.4e-7) | 0.1 s on CPU |
| 5B | ✅ | ❌ (was at t=0.11 of 0.5) | ? | ~1.5 s/step @1024² on 2 CPU cores; seconds on GPU |
| 5C | ✅ | ✅ | ✅ (1.5e-12) | |
| 6 | ✅ | ❌ (not run) | ✅ in study (relL2 6.7e-6 @N=16) | solver validated vs exact: 5e-10 @N=24 |
| 7A | ✅ | ❌ | ✅ in study | N=24 vs 48: 1.2e-6 |
| 7B | ✅ | ❌ | ✅ in study | N=32 vs 80: 8.3e-5, N=64 vs 80: 3e-9 |
| 8 | ✅ | ❌ | ✅ in study | N=16 vs 64: 8e-6 |

## 6. Plan of action (on a machine with PyTorch + GPU)

```bash
git clone <repo> && cd NeuralPDESpeedrun.py
uv sync                                    # torch, numpy, matplotlib, pillow
uv run python -m speedrun.make_reference --track all --device cuda   # regenerate every reference
for t in track1 track2A track2B track3 track4B track5A track5B track5C track6 track7A track7B track8; do
  uv run python -m speedrun.run --track $t --out tracks/$t/logs/baseline.json
done
```

**Step 1: verify under real torch** (expect small fixes only):
- [ ] `make_reference --track all` on CUDA. Compare `meta.json` `sha256_float32` and `norm_l2`
      against the committed CPU/shim versions (tracks 1, 2A, 2B, 4B, 5A, 5C). They should agree to ~1e-12.
      Possible torch-vs-numpy differences to watch: `torch.fft.*` dims, `Tensor.amax(dim=)`,
      `repeat_interleave`, in-place slice updates in `FVLevel.apply`, and complex scalar math in `krylov.gmres`.
- [ ] Device consistency. Some helpers allocate on the default device (`torch.zeros(...)` in `krylov.gmres`
      H/cs/sn/g, `etdrk4_coefficients` contour points use `L.device` ✓). Run with `--device cuda` and fix
      any CPU/CUDA mismatch.
- [ ] For tracks 6/7/8, the dense Kronecker matrices are fine on GPU (≤ 6561² for N=80).

**Step 2: finish missing pieces:**
- [ ] **Track 3 reference.** `make_reference --track track3` (FV 2304², ~5.3M unknowns). Check
      `check_N768_vs_N2304_*`, then run the baseline (FV 256²).
- [ ] **Track 5B reference.** 1024², dt=5e-4. Verify the 512² baseline passes. If not, see §4.
- [ ] **Track 4A Helmholtz** (to implement):
      `−Δu − (ω/c(x,y))² u = f` on [0,1]², u=0 on the boundary, ω = 40,
      `c = 1 − 0.4·exp(−((x−0.65)² + (y−0.6)²)/(2·0.12²))` (slow lens, k_max ≈ 67),
      `f = A·exp(−((x−0.35)² + (y−0.4)²)/(2·0.05²))`, with A set so max|u| ≈ 1; eval on the 257² grid j/256.
      Reference: sine pseudo-spectral (DST-I via `numerics.fourier.dst2/idst2`) at N=768, solved by
      right-preconditioned GMRES (`numerics.krylov.gmres`) with the complex-shifted-Laplacian preconditioner
      `M = −Δ − (1 − 0.5i) k̄²`, inverted exactly in sine space. Check vs N=512. Check that ω is not near a
      cavity resonance (GMRES iteration count, solution norm). Baseline: same at N=256.
      Alternative if you prefer non-resonant physics: replace the Dirichlet walls with a PML or impedance BC.
- [ ] **Leaderboard plotting.** `speedrun/records.py`: read `tracks/<t>/records.json` entries
      `{date, author, name, time_s (H100), rel_l2, linf, hardware, kind: classical|neural|hybrid, pr}`
      and draw `figures/activity.png`: time vs date (log-y), a step line for the record, with classical
      records marked separately. Seed every track with its baseline as record #0, timed on H100.
- [ ] Per-track `README.md` generated from `problem.statement` + reference figure + current record.

**Step 3: first neural records** (see IDEAS.md for the reasoning):
- [ ] Vanilla PINN + Adam + L-BFGS on tracks 1, 2A, 6, 7A. This is the honest starting point and shows
      the gap.
- [ ] PINN + SOAP / natural-gradient (Gauss–Newton with CG) on the same tracks.
- [ ] Multilevel explicit ansatz (dense multiresolution B-spline/Hermite grid, no hashing) + sparse
      Gauss–Newton with level-preconditioned CG. Exact analytic derivatives, no FD.
      Targets: tracks 3, 4A, 4B.
- [ ] Hybrid: classical smoother + learned coarse correction trained in-run with residual loss only
      (tracks 3, 4A).

## 7. Open questions for later
- Second precision tier (e.g. RelL2 ≤ 1e-5) once the 5e-3 tier saturates?
- Separate "neural-field-only" leaderboard (no mesh solver inside) alongside the open hybrid board?
- Multi-instance amortized track (train data-free on an instance sampler, score inference time)?
