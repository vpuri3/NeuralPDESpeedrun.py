# PLAN.md: status, decisions, and next steps

Rules, accuracy bars, and track definitions live in the README. This file tracks what is done, why the tracks
look the way they do, and what comes next. Research reasoning is in IDEAS.md.

## 1. Status (2026-10-30)

- Harness: default bar RelL2 ≤ 1e-4, L∞ ≤ 1e-3 (overridable per track); API `warmup(problem) -> solver`,
  `solve(problem, solver)`, `query(problem, solver)` (≤ 5 s; skipped if `solve` returns the output grid); GPU
  check by torch-op device accounting. Per-axis `Problem.periodic` flags.
- Track set (two-digit numbers): 01 VanDerPol, 02 DampedWave, 03 HighContrastDiffusion, 04A Helmholtz (on hold),
  04B AdvectionDiffusion, 05A Burgers1D, 05B Burgers2D, 05C KuramotoSivashinsky, 06 Kovasznay, 07 CavityRe1000
  (TODO), 08 DecayingEddies, 09 CylinderFlow (TODO), 10 BeltramiFlow (TODO). Removed: the old third-order ODE,
  the plain wave equation, the curved Kovasznay variant, the Re = 100 cavity, and the Stokes cavity.
- Times are on an **RTX 2080 Ti** (fp64 for the baselines, fp32 for the PINNs); H100 re-timing is pending.

| track | reference (self-check) | record #0 | time | RelL2 | L∞ | vanilla PINN, 1000 s (RelL2) |
|---|---|---|---|---|---|---|
| 01 | Radau IIA rtol 1e-12 (rtol 1e-10: 7e-12) | LSODA rtol 1e-8 (CPU) | 37 ms | 8.7e-07 | 5.4e-05 | 1.0e+00 |
| 02 | exact sine series, K=512 (K=256: 0) | sine modes N=256 | 2.9 ms | 8.0e-16 | 7.8e-16 | 8.5e-01 |
| 03 | FV 3840²/5376², Richardson p=1.34 (1e-7) | FV 3328² + MG-PCG | 8.36 s | 3.8e-05 | 1.6e-04 | 9.9e-01 |
| 04A | *on hold* | | | | | |
| 04B | closed form | Chebyshev N=96 | 1.16 s | 2.9e-05 | 3.3e-05 | 9.2e-01 |
| 05A | Fourier N=8192 (4096: 2e-8) | Fourier N=1024 | 0.23 s | 4.4e-07 | 1.1e-04 | 2.6e-03 |
| 05B | Fourier 2048², dt=2.5e-4 (1024²: 6.9e-6) | Fourier 1024², dt=1e-3 | 7.83 s | 9.7e-06 | 2.6e-04 | 2.5e-01 |
| 05C | Fourier N=1024, 64 steps/snapshot (32: 4.8e-9) | Fourier N=256, 16 steps | 6.24 s | 1.0e-07 | 4.1e-05 | 1.0e+00 |
| 06 | closed form | PN–PN-2 N=16, Newton | 0.19 s | 1.4e-07 | 4.7e-07 | 7.0e-04 |
| 07 | *TODO* (see §4) | | | | | |
| 08 | closed form | Fourier vorticity N=32 | 0.13 s | 1.9e-15 | 7.7e-15 | 8.0e-01 |
| 09 | *TODO* | | | | | |
| 10 | *TODO* | | | | | |

No vanilla PINN reaches the bar on any track: the best are 06 (7.0e-4) and 05A (2.6e-3); 01, 02, 03, 04B,
05C and 08 stay at RelL2 ≈ 0.8–1 (no usable solution).

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
- **05C.** T extended from 0.1 to 1 with ν = 0.01 unchanged (max|u| ≈ 34). At T = 1 the error is set by the time
  step, not by chaos: N = 1024 vs 2048 agree; 64 vs 32 steps per snapshot: 4.8e-9 RelL2, 1.9e-6 L∞.
  **TODO:** at T = 5 the dynamics are chaotic at this ν (resolutions differ by O(1) RelL2), so a longer
  horizon needs a different setup (larger ν, shorter T, or a statistical score).
- **06–08.** Reference/baseline solvers use the single-element Legendre PN–PN-2 method in
  `speedrun/numerics/sem.py` (GLL velocity, GLL(N−2) pressure, 3/2 dealiasing, Gordon–Hall maps, BDF3/EXT3 with
  incremental pressure projection, fast-diagonalization solves), validated on Kovasznay to 1e-14. The pressure
  FDM removes only the constant null mode (a relative threshold dropped physical modes at N = 384).
- **07 cavity bar.** The u = 1 lid is discontinuous at the top corners: RelL2 over all points converges ~N⁻¹
  (Gibbs along the lid), ~N⁻² at distance ≥ 0.05 from the walls. Planned bar: RelL2 ≤ 1e-3 (all points),
  L∞ ≤ 1e-3 (≥ 0.05 from the walls).
- **3, 4B.** Forcing scaled so max|u| ≈ 1, which makes the absolute L∞ bar meaningful.

## 4. Next steps (in order)

1. Maintainer submissions (such as the vanilla PINN, `submission_PR0001`) are committed directly to master; the
   draft-PR numbering applies to outside contributions. Reserve PR number #1 or renumber if it collides.
2. **H100 re-timing** of the vanilla PINN results together with the record #0 baselines.
3. **Look into PINN failures that may be bugs rather than method limits.** 03 ends at RelL2 ≈ 1 (output ≈ 0;
   the strong form ignores flux continuity at the inclusion edges and the residual is scaled by a = 1e3).
   Tracks with RelL2 ≈ 1 after the full budget deserve a second look (loss scaling, output scale).
4. **Next neural submissions** (each with its own submission number, committed directly to master):
   - [ ] vanilla PINN + L-BFGS polish;
   - [ ] PINN + SOAP / natural-gradient (Gauss–Newton with CG);
   - [ ] multilevel explicit ansatz (dense multiresolution B-spline/Hermite grid, no hashing) + sparse
         Gauss–Newton with level-preconditioned CG, exact analytic derivatives (targets 03, 04B, later 04A);
   - [ ] hybrid: classical smoother + learned coarse correction trained in-run from residuals only (03);
   - [ ] stiff time integration (01) and long-time wave propagation (02) for neural/hybrid time-steppers.
   See IDEAS.md for the reasoning.

### Deferred (fluids numerics: low priority)

- [ ] **07 CavityRe1000: TODO.** The reference is not finalized. N = 384 runs (BDF3/EXT3, dt = 2× smallest GLL
      spacing) take ~35 min on a 2080 Ti. From the N ≤ 256 study, N = 192 vs 256 differs inside the margin by
      1.0e-3 RelL2 at Re = 1000, so it needs a looser bar or a better reference method (singularity subtraction,
      or multi-element SEM with corner refinement).
- [ ] **05C at longer times** (chaos; see §3).
- [ ] **04A Helmholtz: on hold.** Original setup: `−Δu − (ω/c)² u = f` on [0, 1]², u = 0 on the boundary,
      ω = 40, `c = 1 − 0.4 exp(−((x−0.65)² + (y−0.6)²)/(2·0.12²))`, Gaussian source at (0.35, 0.4), σ = 0.05. It is
      resonant (lossless Dirichlet box): shifted-Laplacian GMRES does not converge in 3000 iterations at N = 256,
      and max|u| changes 7× between ω = 39.5 and 40. Needs absorption (sponge layer / PML / impedance BC).
- [ ] **09 CylinderFlow**: set-up notes in `tracks/track09_CylinderFlow/TODO.md`.
- [ ] **10 BeltramiFlow** (3D unsteady NS, Re = 1, closed form; arXiv:2402.10680 §4.2): set-up notes in
      `tracks/track10_BeltramiFlow/TODO.md`.

## 5. Open questions

- Second precision tier (e.g. RelL2 ≤ 1e-6) once the default tier saturates?
- Separate "neural-field-only" leaderboard (no mesh solver inside) next to the open hybrid board?
- Multi-instance amortized track (train data-free on an instance sampler, score inference time)?
