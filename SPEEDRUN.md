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
| 6A | Kovasznay flow, Re=40, rectangle [-0.5,2]×[-0.5,1.5] (spec) | 251×201 × (u,v) | closed form | PN–PN-2 N=16, Newton |
| 6B | Kovasznay flow, Re=40, trapezoid with parabolic top (spec `geom/para.m`) | 201×201 mapped (r,s) × (u,v) | closed form | PN–PN-2 N=16 on the mapped element, Newton |
| 7A | lid-driven cavity [-1,1]², u=1 lid, Re=100, impulsive start, T=5 (spec) | 10 t × 129² × (u,v) | PN–PN-2 N=384, BDF3/EXT3 | N=96 |
| 7B | same, Re=1000 | 10 t × 129² × (u,v) | PN–PN-2 N=384, BDF3/EXT3 | N=128 |
| 8 | steady Stokes cavity, [-1,1]², u=1 lid (spec geometry) | 129² × (u,v) | PN–PN-2 N=512, Uzawa CG | N=64 |
| 9 | decaying eddies (Walsh 1992), periodic [0,2π)², ν=0.05, T=5 (spec) | 11 t × 128² × (u,v) | closed form | Fourier vorticity N=32, ETDRK4 |

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
- **6–9 follow [vpuri3/spec](https://github.com/vpuri3/spec)** (MATLAB): Kovasznay `kov` (Re=40, rectangle
  and the Gordon–Hall trapezoid with parabolic top), lid-driven cavity `LDC` ([-1,1]², u=1 lid including the top
  corners, ν=1e-2, fluid at rest at t=0, T=5), decaying eddies `wls` (ν=5e-2, T=5). Reference/baseline solvers
  are a PyTorch port of the spec's single-element Legendre PN–PN-2 method (`speedrun/numerics/sem.py`).
  7B (Re=1000) and 8 (Stokes) reuse the spec's cavity setup. The spec's T=0 "steady" mode divides by
  b0=0 in the pressure projection, so steady tracks are solved directly (Newton / Uzawa).
- **7/8 corner singularities.** The u=1 lid is discontinuous at the top corners. Spectral convergence there is
  only algebraic (≈N⁻²), so references use large N. RelL2 is scored everywhere; **L∞ is not scored within
  0.05 of the walls** (`Problem.linf_margin`, applied by the harness via `Problem.linf_mask()`).
- **3, 4B.** Forcing scaled so max|u| ≈ 1, which makes the absolute L∞ tolerance meaningful.

## 5. Status (2026-10-30)

Everything runs under real PyTorch 2.15 + CUDA. Times below are record #0 on an **RTX 2080 Ti** (fp64), not
yet re-timed on the official H100. References regenerated on CUDA agree with the earlier CPU/numpy-shim ones
to ≤ 8e-15.

| track | reference | baseline (record #0) | time (2080 Ti) | RelL2 | L∞ |
|---|---|---|---|---|---|
| 1 | closed form | Chebyshev N=16 | 1.6 ms | 3.4e-13 | 6.2e-13 |
| 2A | d'Alembert | DST N=32 | 1.3 ms | 7.4e-05 | 7.4e-05 |
| 2B | sine series K=32768 | DST N=64 | 1.6 ms | 3.8e-04 | 9.2e-04 |
| 3 | FV 2304² (768 vs 2304: L∞ 8.9e-4) | FV 256² + MG-PCG | 0.17 s | 1.1e-03 | 4.6e-03 |
| 4A | **on hold** (see §6) | | | | |
| 4B | closed form | Chebyshev N=80 | 0.43 s | 5.8e-04 | 6.7e-04 |
| 5A | Fourier N=8192 | Fourier N=1024 | 0.23 s | 4.4e-07 | 1.1e-04 |
| 5B | Fourier 1024², dt=5e-4 | Fourier 512², dt=1e-3 | 3.0 s | 1.2e-04 | 1.6e-03 |
| 5C | Fourier N=1024 | Fourier N=256 | 0.45 s | 1.5e-12 | 2.1e-11 |
| 6A | closed form | PN–PN-2 N=16, Newton | 0.19 s | 1.4e-07 | 4.7e-07 |
| 6B | closed form | PN–PN-2 N=16 (mapped), Newton | 0.23 s | 4.8e-07 | 2.9e-06 |
| 7A | PN–PN-2 N=384 | PN–PN-2 N=96, BDF3/EXT3 | @7A_T@ | @7A_R@ | @7A_L@ |
| 7B | PN–PN-2 N=384 | PN–PN-2 N=128, BDF3/EXT3 | @7B_T@ | @7B_R@ | @7B_L@ |
| 8 | PN–PN-2 N=512 (384 vs 512: 2.1e-4) | PN–PN-2 N=64, Uzawa | 0.11 s | 3.1e-03 | 3.7e-03 |
| 9 | closed form | Fourier vorticity N=32 | 0.14 s | 1.9e-15 | 7.7e-15 |

Harness notes:
- `speedrun.run` sets `CUDA_MODULE_LOADING=EAGER` and initializes cuBLAS/cuSOLVER/cuFFT before the clock.
  Under torch's default lazy loading, first-use kernel loading added ~0.1 s to every small track
  (track 1: 95 ms → 1.5 ms).
- Baselines that use `torch.func` (6A, 6B) run a throwaway N=6 solve in `warmup` to initialize it.
- Cavity time-step study (N=64): CFL 0.25 / 0.5 / 1 / 2 agree to ≤ 5e-6 RelL2, so references and
  baselines use dt = 2 × (smallest GLL spacing). Spatial convergence is algebraic (≈N⁻²) because of the
  lid-corner singularities: Re=100 N=96/128/192 vs 256 give RelL2 1.9e-3 / 8.3e-4 / 5.4e-4; Re=1000
  4.6e-3 / 2.2e-3 / 9.3e-4.

## 6. Plan of action (on a machine with PyTorch + GPU)

```bash
git clone <repo> && cd NeuralPDESpeedrun.py
uv sync                                    # torch, numpy, matplotlib, pillow
uv run python -m speedrun.make_reference --track all --device cuda   # regenerate every reference
for t in track1 track2A track2B track3 track4B track5A track5B track5C track6A track6B track7A track7B track8 track9; do
  uv run python -m speedrun.run --track $t --out tracks/$t/logs/baseline.json
done
```

**Step 1: verify under real torch** ✅ done 2026-10-30 (see §5):
- [x] `make_reference --track all` on CUDA. Compare `meta.json` `sha256_float32` and `norm_l2`
      against the committed CPU/shim versions (tracks 1, 2A, 2B, 4B, 5A, 5C). They should agree to ~1e-12.
      Possible torch-vs-numpy differences to watch: `torch.fft.*` dims, `Tensor.amax(dim=)`,
      `repeat_interleave`, in-place slice updates in `FVLevel.apply`, and complex scalar math in `krylov.gmres`.
- [x] Device consistency. Some helpers allocate on the default device (`torch.zeros(...)` in `krylov.gmres`
      H/cs/sn/g, `etdrk4_coefficients` contour points use `L.device` ✓). Run with `--device cuda` and fix
      any CPU/CUDA mismatch.
- [x] Tracks 6–9 were re-specified from vpuri3/spec (see §4) and now use `numerics/sem.py`.

**Step 2: finish missing pieces:**
- [x] **Track 3 reference.** `make_reference --track track3` (FV 2304², ~5.3M unknowns). Check
      `check_N768_vs_N2304_*`, then run the baseline (FV 256²).
- [x] **Track 5B reference.** 1024², dt=5e-4. Verify the 512² baseline passes. If not, see §4.
- [ ] **Track 4A Helmholtz** — **ON HOLD.** The spec below is resonant: a lossless Dirichlet cavity with a
      lens. Shifted-Laplacian GMRES does not converge in 3000 iterations at N=256, and max|u| changes 7×
      between ω=39.5 and ω=40. It needs absorption (sponge layer / PML / impedance BC) before it is a
      usable benchmark. Original spec:
      `−Δu − (ω/c(x,y))² u = f` on [0,1]², u=0 on the boundary, ω = 40,
      `c = 1 − 0.4·exp(−((x−0.65)² + (y−0.6)²)/(2·0.12²))` (slow lens, k_max ≈ 67),
      `f = A·exp(−((x−0.35)² + (y−0.4)²)/(2·0.05²))`, with A set so max|u| ≈ 1; eval on the 257² grid j/256.
      Reference: sine pseudo-spectral (DST-I via `numerics.fourier.dst2/idst2`) at N=768, solved by
      right-preconditioned GMRES (`numerics.krylov.gmres`) with the complex-shifted-Laplacian preconditioner
      `M = −Δ − (1 − 0.5i) k̄²`, inverted exactly in sine space. Check vs N=512. Check that ω is not near a
      cavity resonance (GMRES iteration count, solution norm). Baseline: same at N=256.
      Alternative if you prefer non-resonant physics: replace the Dirichlet walls with a PML or impedance BC.
- [x] **Leaderboard plotting.** `speedrun/records.py`: read `tracks/<t>/records.json` entries
      `{date, author, name, time_s (H100), rel_l2, linf, hardware, kind: classical|neural|hybrid, pr}`
      and draw `figures/activity.png`: time vs date (log-y), a step line for the record, with classical
      records marked separately. Seed every track with its baseline as record #0, timed on H100.
- [x] Per-track `README.md` generated from `problem.statement` + reference figure + current record.

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
