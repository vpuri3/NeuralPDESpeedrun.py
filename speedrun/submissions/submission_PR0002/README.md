# submission_PR0002: PINNs trained by Gauss-Newton / dual natural gradient (PR #2)

Physics-informed training by damped Gauss-Newton, after Jnini & Vella, *Dual Natural Gradient Descent for Scalable
Training of Physics-Informed Neural Networks* (TMLR 2025, [arXiv:2505.21404](https://arxiv.org/abs/2505.21404)),
in JAX, float64. The time-windowed hard-IC scheme and the 1D settings follow the authors' code
([JAX-NG](https://github.com/HicrestLaboratory/JAX-NG): `optimizers/windowed_gn.py`, `problems/kdv.py`,
`problems/ks1d.py`); the Levenberg-Marquardt damping and geodesic acceleration follow the paper's supplementary
material (`DNGD/main_DNGDgeo.py`).

- **Optimizer** (`gn.py`): one Gauss-Newton step per iteration on the stacked residual r (all terms, vector
  residuals flattened) with Jacobian J (m rows x n parameters; reverse mode per point, vmapped). The normal
  matrix (J^T J, n x n, "primal"; or J J^T, m x m, the paper's residual-space "dual" step, picked when n > m) is
  symmetrically scaled to unit diagonal (Jacobi, as JAX-NG), damped, and factored by Cholesky. Damping: fixed
  1e-8 (the 1D tracks, as JAX-NG), or the paper's Levenberg-Marquardt rule lam = min(loss, 0.1) (2D tracks; without
  it, GN from a random start takes huge steps and stalls, e.g. Kovasznay stays at RelL2 ~50%). Step length: grid
  line search over 0.5^k, k = 0..9, on the same points (the best trial is taken, as JAX-NG). Geodesic
  acceleration (nested `jvp`, same factorization) is implemented but off everywhere.
- **Network** (`net.py`): sine-activated MLP on (2 tau / T - 1, per spatial axis (sin, cos)(2 pi k x / L),
  k = 1..n_modes, if periodic, else the coordinate scaled to [-1, 1]); Glorot-normal init. Derivatives by
  Taylor-mode AD (`jax.experimental.jet`, `net.partials`).
- **PINN** (`pinn.py`): outputs U = (state fields, free fields such as pressure).
  - *Time-dependent tracks*: [0, t_end] is split into equal windows, each with its own network (warm-started
    from the previous one). On window w the state fields are U = s_w(x) + phi_w(tau, x) - phi_w(0, x), s_w the
    initial state plus the jumps of all earlier windows, so the initial condition and the continuity across
    windows are exact; second order in time (02): u and u_t are both continued exactly. Initial states given
    only as samples (15-18) are fitted by a separate network (100 GN steps at the start of `solve`, timed).
  - *Steady tracks*: one network; walls either exact through a factor (07, 09: u = x(1-x)y(1-y) phi) or soft
    (10, 11: weighted residual terms on the walls).
  - Loss: PDE residual at uniformly resampled collocation points (09, 11: denser at the walls), plus boundary
    terms (10, 11, 17, 19: weight 10). Zero Dirichlet walls in 02 are exact through an odd periodic extension.
- **Execution**: everything is compiled ahead of time in `setup` (`jit(...).lower(...).compile()`; problem data
  enter as numpy constants, since closed-over JAX arrays become hidden inputs of AOT-compiled functions). `solve`
  runs a fixed number of GN steps per window, checkpoints after every step, and returns after the last window
  (no reference used). `query` is one compiled pass over the output grid; the window start states are
  accumulated on the grid, so its cost does not grow with the number of windows.

Per-track settings (`tracks/<track>/submissions/submission_PR0002/solve.py`):

| track | windows | MLP | Fourier modes | points / step | GN steps | damping |
|---|---|---|---|---|---|---|
| 02 DampedWave | 16 | 50 x 4 | 30 (odd ext.) | 20k | 100 / window | 1e-8 |
| 03 Burgers | 5 | 40 x 4 | 10 | 20k | 100 / window | 1e-8 |
| 04 KuramotoSivashinsky | 20 (JAX-NG) | 50 x 4 (JAX-NG) | 10 | 20k | 200 / window | 1e-8 |
| 05 AllenCahn | 10 | 50 x 4 | 10 | 20k | 100 / window | 1e-8 |
| 06 KdV | 5 (JAX-NG) | 30 x 4 (JAX-NG) | 10 | 20k | 100 / window | 1e-8 |
| 07 HighContrastDiffusion | steady | 50 x 4 | - | 8k | 6000 | LM |
| 09 AdvectionDiffusion | steady | 50 x 4 | - | 8k (clustered) | 6000 | LM |
| 10 Kovasznay | steady | 50 x 4 | - | 4k + 4 x 500 wall | 300 | LM |
| 11 LidDrivenCavity | steady | 50 x 4 | - | 6k (clustered) + 4 x 750 wall | 6000 | LM |
| 12 Burgers 2D | 10 | 50 x 4 | 10 | 8k | 300 / window | LM |
| 13 DecayingEddies | 5 | 50 x 4 | 8 | 8k | 100 / window | LM |
| 15 GreyScott | 10 | 50 x 4 | 10 | 8k | 300 / window | LM |
| 16 GinzburgLandau | 10 | 50 x 4 | 10 | 8k | 300 / window | LM |
| 17 RayleighTaylor | 10 | 50 x 4 | 8 | 8k + 2k wall | 200 / window | LM |
| 18 KolmogorovFlow | 10 | 50 x 4 | 8 | 8k | 300 / window | LM |
| 19 BeltramiFlow | 2 | 50 x 4 | - | 6k + 3 x 1k wall | 100 / window | 1e-8 |

Differences from JAX-NG: 100-300 GN steps per window instead of 2000, and the window networks are kept as flat
vectors so that one compiled step serves every window.

Not covered: 01 VanDerPol (an adaptive-window variant stalls at the first relaxation jump; shelved), 08 and 14
(no problem defined yet).
