# submission_PR0004: shallow PINNs trained by Levenberg-Marquardt (PR #4)

After Shahab, Mukhlash & Susanto, *Do physics-informed neural networks (PINNs) need to be deep? Shallow PINNs
using the Levenberg-Marquardt algorithm* ([arXiv:2602.08515](https://arxiv.org/abs/2602.08515)), in JAX, float64,
on the 1D tracks. Code: [`pinn.py`](pinn.py); per-track residuals and settings in
`tracks/<track>/submissions/submission_PR0004/solve.py`.

## Method

Key hyperparameters are in one block at the top of [`pinn.py`](pinn.py) (`Config`); each track's `solve.py` sets
its own at its top (currently all tracks: 32×4, points redrawn every 50 iterations, everything else default).

- **Network**: tanh MLP, one output, Glorot-normal weights, zero biases, seed 0; one network on the whole
  space-time domain (no time windows). A periodic axis enters as (sin, cos)(2π(x − lo)/L), so periodicity is exact;
  any other axis is scaled to [−1, 1].
- **Residual**: the PDE residual at 10k uniform random interior points, plus one group of rows per initial or
  boundary condition (`Cond`: 500 random points on a face of the domain, or fixed given points), each group scaled
  by √(10k / its size), so ‖F‖²/10k = mean(PDE²) + Σ mean(condition²). Every 50 iterations all drawn points are
  redrawn, and the loss to beat is recomputed on the new points.
- **Derivatives**: all by JAX AD (Taylor-mode `jet` for input derivatives, reverse mode per point for J).
- **Levenberg-Marquardt**: (JᵀJ + μI) d = JᵀF by Cholesky, no preconditioning or scaling. Rejected trial: μ ← 10μ,
  retry. μ ← μ/10 only after 10 consecutive iterations whose first trial was accepted. No iteration cap: the
  harness's 1000 s budget ends the run (or the step tolerance, or no acceptable μ ≤ 1e16).
- **Inspection** (always on): every run writes `run/tracks/PR0004/<track>_<name>_<datetime>/` (per-iteration μ, loss,
  ‖d‖, gain ratio, trials; residual probes and weight snapshots); [`inspect_run.py`](inspect_run.py) adds the error
  against the reference, J's spectra, timings and one figure. Diagnostic mode (`PR0004_DIAG=1`) adds the singular
  values of J every 50 iterations, kept off the training clock.

## Choices and their reasons

| choice | value | why / status |
|---|---|---|
| network size | 32×4 (max) | 2 hidden layers beat 1 by 1–4 orders of magnitude; 4 beat 2 at width 32; 64-wide is too slow per iteration (140 ms at 64×4) — user cap 32×4 |
| time | one network over [0, T] | the paper's setting (user: no time windows); likely why KS fails |
| conditions | residual rows, weight 1 | user: nothing built in except periodicity; weight 1 untuned. Built-in BCs were 10× more accurate on 1D_Steady_AdvectionDiffusion but not needed for the bar |
| group scaling | √(n_colloc / m) | each group counts as a mean, independent of how many points it has |
| periodicity | (sin, cos) inputs | exact; the only constraint built into the network |
| interior points | 10k uniform random | the paper's count |
| condition points | 500 per face, redrawn with the interior | faces can be overfitted just like the interior; KS's initial samples are given data and stay fixed |
| resampling | every 50 iterations | fixed points overfit (Burgers: held-out PDE residual 600× the training one; DampedWave fixed: loss 4e-12 at 78% error); every-iteration redraws made small nets unstable on DampedWave. 50 is a first guess, not tuned |
| loss to beat after a redraw | recomputed on the new points | so "accept" always compares two losses on the same points; the loss curve therefore jumps at redraws |
| damping form | μI (Levenberg) | user: no preconditioning; Marquardt's μ·diag(JᵀJ) would be a diagonal scaling |
| μ₀ | 0.01 | the paper (MATLAB `fsolve`) |
| μ update | ×10 on rejection; ÷10 after 10 straight first-trial acceptances | the textbook ÷10 after every acceptance made the next first trial fail every time: 2.00 trials per iteration, half the Cholesky solves wasted. Now ~1.09 |
| linear solve | Cholesky of JᵀJ + μI (parameter space) | rows (≈11k) > parameters (≈3.3k), so the 3.3k system is the small one. Forming JᵀJ squares J's conditioning; a QR / SVD solve of the stacked [J; √μI] system would avoid that at higher cost (not done) |
| stopping | budget, step tolerance 1e-12 (relative), μ > 1e16 | no residual-based tolerance yet: the bar is typically met within 1–20 s and the rest of the 1000 s is spent past it — the main open item |
| precision | float64, highest matmul precision | user: everything FP64 |
| derivatives | `jet` (inputs), reverse mode per point (J) | all AD (user); reverse per point keeps memory at rows × parameters |
| checkpoints | every ≤ 50 iterations and ≤ ~1 s | user: not every iteration; ~1 s keeps the budget overshoot under a second for big nets |

## Recorded results (current code: 32×4 on every track, points redrawn every 50 iterations)

Scored on one H100 under the 1000 s budget, 2026-11-05. "Bar first met" is measured after the run from weight
snapshots (the reference is never loaded during a run): the method itself has no stopping rule yet, so every
passing run still uses the whole budget and is scored at ~1000 s.

| track | RelL2 | RelL∞ | bar | bar first met | iterations | ms / iter |
|---|---|---|---|---|---|---|
| 1D_Steady_AdvectionDiffusion | 2.7e-11 | 3.5e-11 | ✅ | 0.96 s (it 113) | 129k | 7.7 |
| 2D_Steady_AdvectionDiffusion | 3.2e-8 | 1.3e-7 | ✅ | 1.06 s (it 113) | 113k | 8.9 |
| 2D_Steady_Kovasznay | 7.8e-9 | 2.4e-8 | ✅ | 1.43 s (it 63) | 47k | 21.5 |
| 1D_Burgers | 8.8e-8 | 3.0e-6 | ✅ | 1.92 s (it 213) | 114k | 8.8 |
| 1D_KdV | 8.1e-6 | 9.2e-6 | ✅ | 5.65 s (it 663) | 119k | 8.4 |
| 1D_AllenCahn | 8.4e-6 | 5.1e-4 | ✅ | 13.0 s (it 1563) | 120k | 8.3 |
| 0D_VanDerPol | 0.94 | 1.66 | ❌ | — (LM stopped itself at 174 s) | 22k | 7.7 |
| 1D_DampedWave | 0.99 | 0.78 | ❌ | — | 111k | 9.0 |
| 1D_KuramotoSivashinsky | 0.98 | 1.11 | ❌ | — | 117k | 8.6 |
| 2D_Steady_LidDrivenCavity | 0.85 | 0.91 | ❌ | — | 46k | 21.8 |
| 2D_GreyScott | 0.64 | 0.88 | ❌ | — | 66k | 15.1 |
| 2D_GinzburgLandau | 1.00 | 1.00 | ❌ | — | 67k | 14.9 |
| 2D_KolmogorovFlow | 1.59 | 2.23 | ❌ | — | 47k | 21.2 |
| 2D_RayleighTaylor | 0.87 | 1.00 | ❌ | — | 35k | 28.9 |

Passing tracks reach the bar within 1–13 s; with a residual-based stopping rule they would be records on most of
them (current records: 37.5 s, 251 s, 17.8 s, 200 s, 15.2 s, 109 s). The failing time-dependent tracks fit the
current points but not the solution between them (held-out PDE residual orders of magnitude above the training
one; DampedWave and KS diagnostics in `run/`), and one network over the whole time interval is the likely limit.
Per-run inspection figures (loss, μ, gain ratio, residual groups, conditioning, solve accuracy) are written to
`run/tracks/PR0004/<track>_<name>_<datetime>/inspect.png` (not in git).

## Experiments (earlier code; run directories deleted 2026-11-05, figures below kept)

1. **Hard vs soft conditions** (1D_Steady_AdvectionDiffusion, 32×2, 4k iterations): u = x(1 − x)·ũ (the paper)
   9.87 s, RelL2 1.9e-9; u = ũ with the walls as residual rows 10.03 s, RelL2 2.2e-8. Kept soft.
2. **Sweep, 20k cap**: widths 8, 16, 32 × 1, 2 hidden layers on all six tracks, fixed points.
3. **No cap** (1000 s budget only): 32×2, 64×2, 32×4, 64×4 on Burgers, KdV, DampedWave, KS, fixed points.
   DampedWave and KS reach tiny losses with large errors (the networks fit the 10k fixed points, not the solution),
   so those two were stopped and rerun with **points redrawn every iteration** (the 64×2 fixed-point runs were
   killed mid-run and are not reported).

Notes on the data below: loss is the logged sum of squared residuals / 10k at each checkpoint (for resampled runs,
on that iteration's points). μ is sampled at the checkpoints (every ≤ 50 iterations), so its min/max are of those
samples; at steady state μ alternates between neighbouring powers of 10 (one rejected trial, then an accepted one).
Sweep 2 ran before the time-based chunking and the resampling option were added; the fixed-point path reproduces
it bit for bit. Times are solve times on one H100 80GB.

### 1D_Steady_AdvectionDiffusion

![1D_Steady_AdvectionDiffusion](figures/1D_Steady_AdvectionDiffusion.png)

| setup | net | params | iterations | time [s] | ms / iter | loss: final (min) | μ: final [min, max] | stopped by | RelL2 | RelL∞ | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k cap, fixed points | 8×1 | 25 | 20000 | 4.9 | 0.24 | 8.8e-14 (8.8e-14) | 1e-11 [1e-11, 1e+00] | 20k-iteration cap | 1.1e-06% | 1.4e-06% | ✅ |
| 20k cap, fixed points | 16×1 | 49 | 20000 | 5.2 | 0.26 | 1.8e-12 (1.8e-12) | 1e-05 [1e-06, 1e+00] | 20k-iteration cap | 5.21e-06% | 6.38e-06% | ✅ |
| 20k cap, fixed points | 32×1 | 97 | 20000 | 7.5 | 0.37 | 6.1e-14 (6.1e-14) | 1e-07 [1e-07, 1e+00] | 20k-iteration cap | 7.29e-07% | 7.84e-07% | ✅ |
| 20k cap, fixed points | 8×2 | 97 | 20000 | 8.2 | 0.41 | 6.3e-12 (6.3e-12) | 1e-05 [1e-05, 1e+01] | 20k-iteration cap | 5.4e-06% | 5.98e-06% | ✅ |
| 20k cap, fixed points | 16×2 | 321 | 20000 | 17.4 | 0.87 | 6.2e-14 (6.2e-14) | 1e-06 [1e-06, 1e+01] | 20k-iteration cap | 1.78e-07% | 2.36e-07% | ✅ |
| 20k cap, fixed points | 32×2 | 1153 | 20000 | 50.1 | 2.51 | 2.8e-15 (2.8e-15) | 1e-06 [1e-06, 1e+01] | 20k-iteration cap | 5.2e-08% | 5.62e-08% | ✅ |

### 1D_Burgers

![1D_Burgers](figures/1D_Burgers.png)

| setup | net | params | iterations | time [s] | ms / iter | loss: final (min) | μ: final [min, max] | stopped by | RelL2 | RelL∞ | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k cap, fixed points | 8×1 | 33 | 4385 | 1.3 | 0.29 | 1.2e-01 (1.2e-01) | 1e+05 [1e-06, 1e+05] | step tolerance | 50.5% | 97.9% | ❌ |
| 20k cap, fixed points | 16×1 | 65 | 20000 | 7.3 | 0.36 | 4.3e-02 (4.3e-02) | 1e-01 [1e-05, 1e+01] | 20k-iteration cap | 21.8% | 117% | ❌ |
| 20k cap, fixed points | 32×1 | 129 | 20000 | 10.5 | 0.52 | 1.1e-02 (1.1e-02) | 1e-05 [1e-05, 1e+01] | 20k-iteration cap | 9.54% | 39.6% | ❌ |
| 20k cap, fixed points | 8×2 | 105 | 20000 | 9.3 | 0.46 | 5.5e-03 (5.5e-03) | 1e-06 [1e-06, 1e+01] | 20k-iteration cap | 7.19% | 59.6% | ❌ |
| 20k cap, fixed points | 16×2 | 337 | 20000 | 19.0 | 0.95 | 1.5e-06 (1.5e-06) | 1e-06 [1e-06, 1e+01] | 20k-iteration cap | 0.166% | 2.24% | ❌ |
| 20k cap, fixed points | 32×2 | 1185 | 20000 | 55.5 | 2.77 | 2.0e-10 (2.0e-10) | 1e-05 [1e-05, 1e+01] | 20k-iteration cap | 0.0835% | 1.07% | ❌ |
| no cap, fixed points | 32×2 | 1185 | 358063 | 1000.0 | 2.79 | 1.2e-12 (1.2e-12) | 1e-07 [1e-08, 1e+02] | budget | 0.00581% | 0.187% | ✅ |
| no cap, fixed points | 64×2 | 4417 | 58813 | 1000.6 | 17.01 | 2.0e-13 (2.0e-13) | 1e-06 [1e-06, 1e+02] | budget | 0.00793% | 0.254% | ✅ |
| no cap, fixed points | 32×4 | 3297 | 92813 | 1000.4 | 10.78 | 2.7e-15 (2.7e-15) | 1e-07 [1e-07, 1e+02] | budget | 0.000435% | 0.00608% | ✅ |
| no cap, fixed points | 64×4 | 12737 | 6814 | 1000.3 | 146.81 | 1.1e-12 (1.1e-12) | 1e-05 [1e-05, 1e+03] | budget | 0.00282% | 0.0406% | ✅ |

### 1D_DampedWave

![1D_DampedWave](figures/1D_DampedWave.png)

| setup | net | params | iterations | time [s] | ms / iter | loss: final (min) | μ: final [min, max] | stopped by | RelL2 | RelL∞ | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k cap, fixed points | 8×1 | 33 | 4694 | 1.3 | 0.28 | 2.3e-03 (2.3e-03) | 1e+16 [1e-09, 1e+16] | mu > mu_max | 86.6% | 77.9% | ❌ |
| 20k cap, fixed points | 16×1 | 65 | 20000 | 7.8 | 0.39 | 1.2e-03 (1.2e-03) | 1e-06 [1e-06, 1e+00] | 20k-iteration cap | 81.4% | 67.5% | ❌ |
| 20k cap, fixed points | 32×1 | 129 | 20000 | 10.1 | 0.50 | 8.0e-04 (8.0e-04) | 1e-04 [1e-04, 1e+00] | 20k-iteration cap | 70% | 54.3% | ❌ |
| 20k cap, fixed points | 8×2 | 105 | 20000 | 9.8 | 0.49 | 1.2e-03 (1.2e-03) | 1e-02 [1e-03, 1e+01] | 20k-iteration cap | 82.5% | 69.9% | ❌ |
| 20k cap, fixed points | 16×2 | 337 | 20000 | 18.4 | 0.92 | 3.1e-04 (3.1e-04) | 1e-02 [1e-03, 1e+01] | 20k-iteration cap | 50.6% | 46.7% | ❌ |
| 20k cap, fixed points | 32×2 | 1185 | 20000 | 54.8 | 2.74 | 1.1e-05 (1.1e-05) | 1e-02 [1e-02, 1e+01] | 20k-iteration cap | 34.2% | 34.7% | ❌ |
| no cap, fixed points | 32×4 | 3297 | 90963 | 1000.4 | 11.00 | 3.9e-12 (3.9e-12) | 1e-04 [1e-04, 1e+01] | budget | 77.9% | 77.6% | ❌ |
| no cap, fixed points | 64×4 | 12737 | 6766 | 1000.4 | 147.86 | 3.3e-08 (3.3e-08) | 1e-01 [1e-01, 1e+01] | budget | 91.3% | 77.9% | ❌ |
| no cap, resampled every iteration | 32×2 | 1185 | 356963 | 1000.1 | 2.80 | 1.1e-05 (1.1e-07) | 1e-04 [1e-05, 1e+01] | budget | 50.6% | 112% | ❌ |
| no cap, resampled every iteration | 64×2 | 4417 | 56913 | 1000.8 | 17.59 | 8.4e-08 (8.3e-08) | 1e-02 [1e-03, 1e+01] | budget | 1.07% | 0.95% | ❌ |
| no cap, resampled every iteration | 32×4 | 3297 | 91063 | 1000.3 | 10.99 | 4.6e-07 (5.0e-08) | 1e-03 [1e-03, 1e+03] | budget | 0.393% | 0.334% | ✅ |
| no cap, resampled every iteration | 64×4 | 12737 | 6800 | 1000.5 | 147.14 | 9.1e-09 (6.6e-09) | 1e-06 [1e-06, 1e+01] | budget | 0.135% | 0.133% | ✅ |

### 1D_KuramotoSivashinsky

![1D_KuramotoSivashinsky](figures/1D_KuramotoSivashinsky.png)

| setup | net | params | iterations | time [s] | ms / iter | loss: final (min) | μ: final [min, max] | stopped by | RelL2 | RelL∞ | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k cap, fixed points | 8×1 | 41 | 9177 | 2.6 | 0.28 | 3.7e-01 (3.7e-01) | 1e+16 [1e-09, 1e+16] | mu > mu_max | 98% | 110% | ❌ |
| 20k cap, fixed points | 16×1 | 81 | 9245 | 3.8 | 0.41 | 3.6e-01 (3.6e-01) | 1e+16 [1e-10, 1e+16] | mu > mu_max | 98.8% | 111% | ❌ |
| 20k cap, fixed points | 32×1 | 161 | 20000 | 11.0 | 0.55 | 3.6e-01 (3.6e-01) | 1e-01 [1e-06, 1e+00] | 20k-iteration cap | 99.3% | 112% | ❌ |
| 20k cap, fixed points | 8×2 | 113 | 20000 | 9.4 | 0.47 | 3.5e-01 (3.5e-01) | 1e-02 [1e-04, 1e+01] | 20k-iteration cap | 99.4% | 113% | ❌ |
| 20k cap, fixed points | 16×2 | 353 | 20000 | 19.4 | 0.97 | 6.5e-02 (6.5e-02) | 1e+02 [1e-01, 1e+03] | 20k-iteration cap | 110% | 120% | ❌ |
| 20k cap, fixed points | 32×2 | 1217 | 20000 | 61.7 | 3.08 | 3.1e-05 (3.1e-05) | 1e-02 [1e-02, 1e+02] | 20k-iteration cap | 92.7% | 113% | ❌ |
| no cap, fixed points | 32×4 | 3329 | 94413 | 1000.2 | 10.59 | 3.7e-06 (3.7e-06) | 1e+01 [1e+01, 1e+03] | budget | 129% | 908% | ❌ |
| no cap, fixed points | 64×4 | 12801 | 7053 | 1000.2 | 141.81 | 1.1e-02 (1.1e-02) | 1e+03 [1e+01, 1e+04] | budget | 107% | 397% | ❌ |
| no cap, resampled every iteration | 32×2 | 1217 | 312763 | 1000.0 | 3.20 | 5.1e-05 (4.1e-05) | 1e-01 [1e-01, 1e+02] | budget | 91.7% | 112% | ❌ |
| no cap, resampled every iteration | 64×2 | 4481 | 57113 | 1000.4 | 17.52 | 1.9e-05 (1.7e-05) | 1e-01 [1e-02, 1e+02] | budget | 92.1% | 111% | ❌ |
| no cap, resampled every iteration | 32×4 | 3329 | 92963 | 1000.4 | 10.76 | 1.6e-05 (1.5e-05) | 1e-01 [1e-02, 1e+02] | budget | 93.2% | 111% | ❌ |
| no cap, resampled every iteration | 64×4 | 12801 | 7135 | 1000.3 | 140.20 | 2.9e-05 (2.1e-05) | 1e-01 [1e-02, 1e+04] | budget | 92.4% | 112% | ❌ |

### 1D_AllenCahn

![1D_AllenCahn](figures/1D_AllenCahn.png)

| setup | net | params | iterations | time [s] | ms / iter | loss: final (min) | μ: final [min, max] | stopped by | RelL2 | RelL∞ | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k cap, fixed points | 8×1 | 41 | 20000 | 5.5 | 0.28 | 8.9e-04 (8.9e-04) | 1e-10 [1e-10, 1e+01] | 20k-iteration cap | 70.6% | 191% | ❌ |
| 20k cap, fixed points | 16×1 | 81 | 20000 | 7.6 | 0.38 | 6.9e-04 (6.9e-04) | 1e-02 [1e-05, 1e+01] | 20k-iteration cap | 58.4% | 176% | ❌ |
| 20k cap, fixed points | 32×1 | 161 | 17869 | 9.7 | 0.54 | 6.2e-04 (6.2e-04) | 1e+13 [1e-09, 1e+13] | step tolerance | 50.7% | 175% | ❌ |
| 20k cap, fixed points | 8×2 | 113 | 20000 | 8.9 | 0.45 | 4.1e-05 (4.1e-05) | 1e-05 [1e-06, 1e+01] | 20k-iteration cap | 14.8% | 88.2% | ❌ |
| 20k cap, fixed points | 16×2 | 353 | 20000 | 18.3 | 0.92 | 7.4e-07 (7.4e-07) | 1e-07 [1e-08, 1e+01] | 20k-iteration cap | 1.66% | 19.5% | ❌ |
| 20k cap, fixed points | 32×2 | 1217 | 20000 | 59.6 | 2.98 | 5.4e-11 (5.4e-11) | 1e-06 [1e-06, 1e+01] | 20k-iteration cap | 0.00403% | 0.259% | ✅ |

### 1D_KdV

![1D_KdV](figures/1D_KdV.png)

| setup | net | params | iterations | time [s] | ms / iter | loss: final (min) | μ: final [min, max] | stopped by | RelL2 | RelL∞ | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k cap, fixed points | 8×1 | 41 | 2043 | 0.6 | 0.29 | 9.4e-02 (9.4e-02) | 1e+13 [1e-07, 1e+13] | step tolerance | 70.9% | 109% | ❌ |
| 20k cap, fixed points | 16×1 | 81 | 20000 | 8.0 | 0.40 | 7.0e-02 (7.0e-02) | 1e+00 [1e-04, 1e+02] | 20k-iteration cap | 75.7% | 110% | ❌ |
| 20k cap, fixed points | 32×1 | 161 | 20000 | 11.0 | 0.55 | 3.6e-02 (3.6e-02) | 1e-01 [1e-05, 1e+01] | 20k-iteration cap | 66.7% | 104% | ❌ |
| 20k cap, fixed points | 8×2 | 113 | 20000 | 9.2 | 0.46 | 2.9e-02 (2.9e-02) | 1e-02 [1e-07, 1e+01] | 20k-iteration cap | 84.7% | 115% | ❌ |
| 20k cap, fixed points | 16×2 | 353 | 20000 | 19.0 | 0.95 | 7.1e-04 (7.1e-04) | 1e-04 [1e-06, 1e+01] | 20k-iteration cap | 5.58% | 5.75% | ❌ |
| 20k cap, fixed points | 32×2 | 1217 | 20000 | 60.7 | 3.04 | 9.6e-08 (9.6e-08) | 1e-03 [1e-03, 1e+01] | 20k-iteration cap | 0.0335% | 0.0526% | ✅ |
| no cap, fixed points | 32×2 | 1217 | 329963 | 1000.1 | 3.03 | 1.1e-08 (1.1e-08) | 1e-05 [1e-05, 1e+01] | budget | 0.00813% | 0.0124% | ✅ |
| no cap, fixed points | 64×2 | 4481 | 58613 | 1000.8 | 17.07 | 3.0e-11 (3.0e-11) | 1e-05 [1e-05, 1e+01] | budget | 0.000665% | 0.00109% | ✅ |
| no cap, fixed points | 32×4 | 3329 | 95563 | 1000.0 | 10.46 | 7.4e-12 (7.4e-12) | 1e-05 [1e-05, 1e+02] | budget | 0.000315% | 0.000412% | ✅ |
| no cap, fixed points | 64×4 | 12801 | 7147 | 1000.7 | 140.01 | 6.4e-10 (6.4e-10) | 1e-03 [1e-03, 1e+03] | budget | 0.00261% | 0.00373% | ✅ |

