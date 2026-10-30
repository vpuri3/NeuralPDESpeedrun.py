# PLAN.md: status, decisions, and next steps

Rules, accuracy bars, and track definitions live in the README. This file tracks what is done, why the tracks
look the way they do, and what comes next. Research reasoning is in IDEAS.md.

## 1. Status (2026-10-30)

- Harness: default bar RelL2 ≤ 1e-4, L∞ ≤ 1e-3 (overridable per track); API `warmup(problem) -> solver`,
  `solve(problem, solver)`, `query(problem, solver)` (≤ 5 s; skipped if `solve` returns the output grid); GPU
  check by torch-op device accounting. Per-axis `Problem.periodic` flags.
- Track set (two-digit numbers): 01 VanDerPol, 02 DampedWave, 03 HighContrastDiffusion, 04A Helmholtz (on hold),
  04B AdvectionDiffusion, 05A Burgers1D, 05B Burgers2D, 05C KuramotoSivashinsky, 06 Kovasznay, 07 LidDrivenCavity,
  08 DecayingEddies, 09 CylinderFlow (TODO), 10 BeltramiFlow (TODO), 11 AllenCahn, 12 KdV, 13 GreyScott,
  14 GinzburgLandau, 15 RayleighTaylor, 16 KolmogorovFlow.
- **Benchmark tracks (05C, 07, 11–16)** come from the PirateNet / SOAP-for-PINNs study (Wang et al.; data from
  github.com/PredictiveIntelligenceLab/jaxpi, branch `pirate`, commit 9b5196b). References are their published
  solutions, downloaded and sha256-checked by `speedrun/external.py` (pickled .npy files go through a restricted
  unpickler). Initial states that are only known as data are given to solvers as samples (`fns["ic"]`), and the
  scored times exclude t = 0. No classical baselines on these tracks. Per-track bars are set near what SOAP
  reports in the paper (their references do not support 1e-4 everywhere): 05C 5e-2, 07 5e-2, 11 1e-4, 12 1e-3,
  13 1e-3, 14 1e-2, 15 2e-2, 16 5e-2 (RelL2).
- The vanilla PINN was a workflow draft, not a submission: it now lives in `submissions/draft_vanilla_pinn/` and
  is not in the records.
- Times are on an **RTX 2080 Ti** (fp64 for the baselines, fp32 for the PINNs); H100 re-timing is pending.

| track | reference (self-check) | record #0 | time | RelL2 | L∞ |
|---|---|---|---|---|---|
| 01 | Radau IIA rtol 1e-12 (rtol 1e-10: 7e-12) | LSODA rtol 1e-8 (CPU) | 37 ms | 8.7e-07 | 5.4e-05 |
| 02 | exact sine series, K=512 (K=256: 0) | sine modes N=256 | 2.9 ms | 8.0e-16 | 7.8e-16 |
| 03 | FV 3840²/5376², Richardson p=1.34 (1e-7) | FV 3328² + MG-PCG | 8.36 s | 3.8e-05 | 1.6e-04 |
| 04A | *on hold* | | | | |
| 04B | closed form | Chebyshev N=96 | 1.16 s | 2.9e-05 | 3.3e-05 |
| 05A | Fourier N=8192 (4096: 2e-8) | Fourier N=1024 | 0.23 s | 4.4e-07 | 1.1e-04 |
| 05B | Fourier 2048², dt=2.5e-4 (1024²: 6.9e-6) | Fourier 1024², dt=1e-3 | 7.83 s | 9.7e-06 | 2.6e-04 |
| 05C | published data | — | | | |
| 06 | closed form | PN–PN-2 N=16, Newton | 0.19 s | 1.4e-07 | 4.7e-07 |
| 07 | published data (Re = 5000) | — | | | |
| 08 | closed form | Fourier vorticity N=32 | 0.13 s | 1.9e-15 | 7.7e-15 |
| 09 | *TODO* | | | | |
| 10 | *TODO* | | | | |
| 11–16 | published data | — | | | |

## 2. Harness notes

- `speedrun.run` sets `CUDA_MODULE_LOADING=EAGER` and initializes cuBLAS/cuSOLVER/cuFFT before the clock.
  Under torch's default lazy loading, first-use kernel loading added ~0.1 s to every small track.
- GPU check (`speedrun/devicecheck.py`): one extra untimed warmup + solve + query under a `TorchDispatchMode`
  that counts arithmetic ops by tensor device (meta tensors, data movement and views are ignored). A run with no
  torch arithmetic on the GPU (e.g. the scipy baseline of 01) is reported as CPU. Work outside torch (numpy,
  scipy, JAX, Julia, custom kernels) is invisible to it.
- Baselines using `torch.func` (06) run a throwaway N = 6 solve in `warmup` to initialize it. Imports of heavy
  libraries (scipy) belong at module level, outside the timed `solve`.

## 3. Problem-spec decisions

- **01.** Van der Pol, μ = 1000, y(0) = 2, y'(0) = 0, T = 3000 (period ≈ 1614). Radau rtol 1e-10 vs 1e-12:
  7e-12; BDF fails at tight tolerances; LSODA rtol 1e-6 / 1e-7 / 1e-8 gives RelL2 1.0e-4 / 8.2e-6 / 8.7e-7.
- **02.** `u_tt = u_xx − 0.5 u_t`, pulse σ = 0.02 at x = 0.3, T = 4 (four domain crossings). The pulse vanishes at
  the walls to ~1e-49, so the sine coefficients are closed-form Gaussian integrals and the reference is exact.
  A sine-mode method is exact here too (2.9 ms); the track is hard for mesh-free / neural methods (narrow
  features, long time, reflections), not for spectral ones.
- **03.** FV on r × 256 grids (r odd, so output centers are fine-cell centers). The observed order is 1.34
  (inclusion corners). Reference = Richardson extrapolation of 3840² and 5376² at that order; the 2304²/3840²
  extrapolation agrees to 1e-7. Baseline 3328² (r = 13): 3.8e-5.
- **04B.** Chebyshev N = 80 / 88 / 96 / 104: RelL2 5.8e-4 / 1.4e-4 / 2.9e-5 / 5.4e-6. Baseline N = 96.
- **05A.** ν = 1e-4, T = 0.5, domain [0, 2), N = 1024 output grid, continuous IC. The shock forms at
  t* = 1/(πμ) ≈ 0.53, just after T.
- **05B.** The 512² output grid is too coarse for the ν = 1e-3 shocks at the bar (512²: L∞ 1.6e-3). Reference
  2048², dt = 2.5e-4; baseline 1024², dt = 1e-3, injected.
- **05C** is now the benchmark's chaotic KS (published data). The earlier Gaussian-IC KS setup was dropped.
- **06, 08.** Reference/baseline solvers use the single-element Legendre PN–PN-2 method in
  `speedrun/numerics/sem.py` (GLL velocity, GLL(N−2) pressure, 3/2 dealiasing, Gordon–Hall maps, BDF3/EXT3 with
  incremental pressure projection, fast-diagonalization solves), validated on Kovasznay to 1e-14. The pressure
  FDM removes only the constant null mode (a relative threshold dropped physical modes at N = 384).
- **07** is now the benchmark's steady Re = 5000 cavity (regularized lid; published data). Its data use an
  unregularized lid row, so L∞ excludes points within 0.05 of the walls.
- **3, 4B.** Forcing scaled so max|u| ≈ 1, which makes the absolute L∞ bar meaningful.

## 4. Next steps (in order)

1. **Submission 1 (PR #1): SOAP for PINNs** (Wang et al.), with the PirateNet backbone and the training tricks of
   the paper (grad-norm loss weighting, causal weighting, exact periodicity), run for at most 1000 s per track
   with early stopping, on every active track.
2. **H100 re-timing** of record #0 baselines and submissions.
3. **Next neural submissions** (each with its own PR / submission number):
   - [ ] PINN + Gauss–Newton natural gradient (matrix-free CG);
   - [ ] multilevel explicit ansatz (dense multiresolution B-spline/Hermite grid, no hashing) + sparse
         Gauss–Newton with level-preconditioned CG, exact analytic derivatives (targets 03, 04B);
   - [ ] hybrid: classical smoother + learned coarse correction trained in-run from residuals only (03);
   - [ ] stiff time integration (01) and long-time wave propagation (02) for neural/hybrid time-steppers.
   See IDEAS.md for the reasoning.

### Deferred

- [ ] **04A Helmholtz: on hold** (resonant as specified; needs absorption).
- [ ] **09 CylinderFlow**, **10 BeltramiFlow**: set-up notes in their `TODO.md`.
- [ ] Classical baselines for the benchmark tracks (05C, 07, 11–16), if wanted.

## 5. Open questions

- Second precision tier (e.g. RelL2 ≤ 1e-6) once the default tier saturates?
- Separate "neural-field-only" leaderboard (no mesh solver inside) next to the open hybrid board?
- Multi-instance amortized track (train data-free on an instance sampler, score inference time)?
