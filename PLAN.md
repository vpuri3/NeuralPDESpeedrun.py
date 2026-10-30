# PLAN.md: status, decisions, and next steps

Rules, accuracy bars, and track definitions live in the README. This file tracks what is done, why the tracks
look the way they do, and what comes next. Research reasoning is in IDEAS.md.

## 1. Status (2026-10-30)

- Everything runs under PyTorch 2.15 + CUDA. References regenerated on CUDA agree with the earlier
  CPU/numpy-shim ones to ≤ 8e-15.
- Default accuracy bar tightened from (RelL2 5e-3, L∞ 1e-2) to **(1e-4, 1e-3)**, now per track
  (`Problem.rel_l2_tol / linf_tol`). Baselines and references were re-sized so that every reference error is
  well below its track's bar (see §3).
- Tracks renamed `track<KK>[A|B]_<ProblemName>`. Submissions go in `tracks/<t>/submissions/submission_PR<NNNN>/`,
  with code shared across tracks in `speedrun/submissions/submission_PR<NNNN>/`.
- Submission API split into `solve(problem) -> state` and `query(state, problem)` (projection onto the
  evaluation points, timed separately, ≤ 5 s).
- Every record #0 below was checked by the harness GPU check (100% of tensor work on the GPU). Times are on an
  **RTX 2080 Ti** in fp64; H100 re-timing is pending.

| track | reference (self-check) | record #0 | time (2080 Ti) | RelL2 | L∞ |
|---|---|---|---|---|---|
| 01 | closed form | Chebyshev N=16 | 1.5 ms | 3.4e-13 | 6.2e-13 |
| 02A | d'Alembert | sine spectral N=32 | 1.2 ms | 7.4e-05 | 7.4e-05 |
| 02B | sine series K=32768 (K=8192: 3e-6) | sine spectral N=512 | 1.9 ms | 4.3e-05 | 1.2e-04 |
| 03 | FV 3840²/5376², Richardson p=1.34 (1e-7) | FV 3328² + MG-PCG | 8.39 s | 3.8e-05 | 1.6e-04 |
| 04A | *on hold* | | | | |
| 04B | closed form | Chebyshev N=96 | 1.16 s | 2.9e-05 | 3.3e-05 |
| 05A | Fourier N=8192 (4096: 2e-8) | Fourier N=1024 | 0.23 s | 4.4e-07 | 1.1e-04 |
| 05B | Fourier 2048², dt=2.5e-4 (1024²: 6.9e-6) | Fourier 1024², dt=1e-3 | 7.83 s | 9.7e-06 | 2.6e-04 |
| 05C | Fourier N=1024 (512: 2e-13) | Fourier N=256 | 0.46 s | 1.5e-12 | 2.1e-11 |
| 06A | closed form | PN–PN-2 N=16, Newton | 0.19 s | 1.4e-07 | 4.7e-07 |
| 06B | closed form | PN–PN-2 N=16, Newton | 0.22 s | 4.8e-07 | 2.9e-06 |
| 07A / 07B | *TODO* (see §4) | | | | |
| 08 | PN–PN-2 N=768 (512: 1.1e-4) | PN–PN-2 N=192, Uzawa | 0.22 s | 5.0e-04 | 4.0e-04 |
| 09 | closed form | Fourier vorticity N=32 | 0.14 s | 1.9e-15 | 7.7e-15 |

## 2. Harness notes

- `speedrun.run` sets `CUDA_MODULE_LOADING=EAGER` and initializes cuBLAS/cuSOLVER/cuFFT before the clock.
  Under torch's default lazy loading, first-use kernel loading added ~0.1 s to every small track
  (track 01: 95 ms → 1.5 ms).
- GPU check (`speedrun/devicecheck.py`): one extra untimed `solve` under a `TorchDispatchMode` that counts ops by
  tensor device (meta tensors from functorch shape inference are ignored). It caught the GLL node setup in
  `numerics/sem.py` running on the CPU; that is fixed. Work outside torch (numpy, scipy) is invisible to it.
- Baselines using `torch.func` (06A, 06B) run a throwaway N = 6 solve in `warmup` to initialize it.

## 3. Problem-spec decisions

- **02B.** `u_xx = u_tt/v² + b u_t`, v = 3, b = 2. `u_t(0, x) = 1 − 2x` is incompatible with the walls, so sine methods converge at
  first order: N = 64 / 128 / 256 / 512 give RelL2 3.8e-4 / 1.7e-4 / 8.6e-5 / 4.3e-5. Baseline N = 512.
- **03.** FV on r × 256 grids (r odd, so output centers are fine-cell centers). The observed order is
  1.34 (stable over the grid triples 5/9/15 and 9/15/21) because of the inclusion corners. The reference is the
  Richardson extrapolation of 3840² and 5376² at that order. The 2304²/3840² extrapolation agrees to 1e-7;
  un-extrapolated 5376² is still 2e-5 off. Baseline 3328² (r = 13): 3.8e-5.
- **04B.** Chebyshev N = 80 / 88 / 96 / 104: RelL2 5.8e-4 / 1.4e-4 / 2.9e-5 / 5.4e-6. Baseline N = 96.
- **05A.** ν = 1e-4, T = 0.5, domain [0, 2), N = 1024 output grid, continuous IC ("1 otherwise"). The shock
  forms at t* = 1/(πμ) ≈ 0.53, just after T.
- **05B.** The 512² output grid (paper) is too coarse for the ν = 1e-3 shocks at the new bar (512²: L∞ 1.6e-3).
  Reference 2048², dt = 2.5e-4 (vs 1024², dt = 5e-4: 6.9e-6 / 3.1e-5). Baseline 1024², dt = 1e-3, injected.
- **05A/05B/05C** are single instances (one solve per record).
- **06–09.** Kovasznay at Re = 40 (rectangle and a Gordon–Hall trapezoid with parabolic top), lid-driven cavity
  ([−1, 1]², u = 1 lid including the top corners, ν = 1e-2, fluid at rest at t = 0, T = 5), decaying eddies
  (ν = 5e-2, T = 5). Reference/baseline solvers use the single-element Legendre PN–PN-2 method in
  `speedrun/numerics/sem.py`: GLL velocity, GLL(N−2) pressure, 3/2 dealiasing, Gordon–Hall maps, BDF3/EXT3 with
  incremental pressure projection, fast-diagonalization solves. Validated on Kovasznay: 1e-14 on both domains;
  marching to steady state reproduces the steady solution to 4e-7. Steady tracks are solved directly (Newton,
  Uzawa) rather than by time-marching.
- **07/08 cavity bar.** The u = 1 lid is discontinuous at the top corners. RelL2 over all 129² points converges
  ~N⁻¹ (Gibbs along the lid) and ~N⁻² at distance ≥ 0.05 from the walls. Stokes vs N = 768: N = 128 / 256 / 512
  give RelL2 7.4e-4 / 2.6e-4 / 1.1e-4 over all points and 5.3e-4 / 1.2e-4 / 1.9e-5 inside the margin.
  The default 1e-4 is below what any affordable reference reaches. N = 1024 fails: the pressure mass matrix is
  not numerically positive definite in fp64. Hence the per-track bar RelL2 ≤ 1e-3 (all points), L∞ ≤ 1e-3
  (≥ 0.05 from the walls). Possible fix for a tighter bar: singularity subtraction (Botella & Peyret 1998).
- **07 time step.** CFL 0.25 / 0.5 / 1 / 2 (N = 64) agree to ≤ 5e-6, so dt = 2 × (smallest GLL spacing).
  Bug found on the way: the pressure FDM dropped eigenvalues below 1e-10·max, which at N = 384 (max ≈ 2e11)
  also dropped physical modes. Only the constant null mode is removed now.
- **3, 4B.** Forcing scaled so max|u| ≈ 1, which makes the absolute L∞ bar meaningful.

## 4. Next steps (in order)

Where things stood when work stopped (2026-10-30): the solve/query API, the vanilla PINN code
(`submission_PR0001`), and the accuracy-vs-time plots are committed. Every track's PINN `solve`/`query` passed
a 20 s smoke test (shapes correct, query ≤ 0.11 s). The 1000 s runs were **stopped before any finished**:
there are no PINN results yet, and no `result.json` files. 20 s smoke RelL2 (indicative only):
01 1.9e-4 · 02A 4.1e-2 · 02B 1.3 · 03 1.0 · 04B 0.88 · 05A 0.18 · 05B 0.90 · 05C 0.99 · 06A 0.46 · 06B 8.4e-2 ·
08 1.0 · 09 1.0.

1. **Open the draft PR for the vanilla PINN** and check that its GitHub number is #1, so it matches
   `submission_PR0001`. Otherwise rename the directories (`tracks/*/submissions/submission_PR0001/`,
   `speedrun/submissions/submission_PR0001/`) and their imports. Requires `gh` (not installed on the dev box) or
   the web UI.
2. **Run the vanilla PINN on every active track** (1000 s budget each; one run per GPU at a time so budgets are
   not shared; ≈ 51 min on 4 GPUs):
   ```bash
   d=$(ls -d tracks/track${t}_*)/submissions/submission_PR0001
   uv run python -m speedrun.run --track $t --submission $d --out $d/result.json
   uv run python -m speedrun.records --track $t --add $d/result.json --name "vanilla PINN" --kind neural --pr 1
   ```
   Then `uv run python -m speedrun.records --track all` (tracker plots, per-track READMEs), commit, and mark
   the PR ready for review.
3. **Sanity-check PINN failures that look like bugs, not method limits:** 08 and 09 sat at RelL2 ≈ 1.0 after
   20 s (output ≈ 0?). 03's strong form ignores the flux-continuity conditions at the inclusion edges (by
   design for a vanilla PINN, but document the outcome).
4. **Next neural submissions** (new draft PR each, so each gets its own number):
   - [ ] vanilla PINN + L-BFGS polish;
   - [ ] PINN + SOAP / natural-gradient (Gauss–Newton with CG);
   - [ ] multilevel explicit ansatz (dense multiresolution B-spline/Hermite grid, no hashing) + sparse
         Gauss–Newton with level-preconditioned CG, exact analytic derivatives (targets 03, 04B, later 04A);
   - [ ] hybrid: classical smoother + learned coarse correction trained in-run from residuals only (03);
   - [ ] curved / non-tensor geometry (06B, later 10) for mesh-free and hybrid methods.
   See IDEAS.md for the reasoning.
5. **H100 re-timing** of every record #0 and of the PINN results (official times).

### Deferred (fluids numerics: low priority)

- [ ] **07A / 07B cavity: TODO.** The reference is not finalized. N = 384 runs (BDF3/EXT3, dt = 2× smallest GLL
      spacing) take ~35 min each on a 2080 Ti; the first attempt was invalidated by the pressure-FDM null-mode bug
      (fixed since). From the N ≤ 256 study, N = 192 vs 256 differs inside the margin by 1.8e-4 RelL2 at Re = 100
      and 1.0e-3 at Re = 1000, so 07B needs a looser bar or a better reference method (singularity subtraction,
      or multi-element SEM with corner refinement).
- [ ] **04A Helmholtz: on hold.** Original setup: `−Δu − (ω/c)² u = f` on [0, 1]², u = 0 on the boundary,
      ω = 40, `c = 1 − 0.4 exp(−((x−0.65)² + (y−0.6)²)/(2·0.12²))`, Gaussian source at (0.35, 0.4), σ = 0.05. It is
      resonant (lossless Dirichlet box): shifted-Laplacian GMRES does not converge in 3000 iterations at N = 256,
      and max|u| changes 7× between ω = 39.5 and 40. Needs absorption (sponge layer / PML / impedance BC), with
      the reference as sine pseudo-spectral + complex-shifted-Laplacian-preconditioned GMRES.
- [ ] **10 CylinderFlow**: set-up notes in `tracks/track10_CylinderFlow/TODO.md`.
- [ ] Cavity references closer to the default bar (07, 08).

## 5. Open questions

- Second precision tier (e.g. RelL2 ≤ 1e-6) once the default tier saturates?
- Separate "neural-field-only" leaderboard (no mesh solver inside) next to the open hybrid board?
- Multi-instance amortized track (train data-free on an instance sampler, score inference time)?
