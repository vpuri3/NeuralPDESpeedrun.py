# submission_PR0001: PirateNet + SOAP (PR #1)

Physics-informed training with the second-order SOAP optimizer, following Wang et al., *Gradient Alignment in
Physics-informed Neural Networks: A Second-Order Optimization Perspective*, in JAX, as a port of their reference
code (github.com/PredictiveIntelligenceLab/jaxpi, branch `pirate`) and its `pirate_soap` / `soap` configs:

- **Network** (`pinn.py`, port of `jaxpi.archs.PirateNet`): per-point network on raw coordinates; period
  embeddings (cos, sin)(2π x / L) of periodic coordinates; trainable random Fourier features (width 256);
  weight-factorized dense layers (W = g · V, log g ~ N(1, 0.1)); 3 gated residual blocks (4 for 11) of width 256
  (384 for 18) whose skip weights α start at 0.
- **Optimizer**: SOAP from [`soap_jax`](https://github.com/haydn-jones/SOAP_JAX) (pinned commit `a7c92f1`), the
  implementation jaxpi uses, with jaxpi's arguments: b1 = 0.9, b2 = 0.999, precondition_frequency = 2,
  weight_decay = 0. Learning rate 1e-3, linear warmup over 5000 iterations (2000 for 17, 18), then decay 0.9
  every `decay_steps` (optax, as `jaxpi.models._create_optimizer`).
- **Loss weighting**: grad-norm weights (mean grad norm / grad norm of each loss term) with momentum 0.9, every
  1000 iterations from iteration 0, as `jaxpi.models.PINN.compute_weights`. Evolution problems use causal
  weights (residuals sorted in time, chunk i weighted by exp(−tol Σ_{j<i} L_j)).
- **Physics-informed initialization** (the IVPs whose jaxpi configs use it, and the other IVPs with a known
  IC): the last layer is the least-squares fit of the initial state, tiled over 20 times, on the features of
  the initialized network. It runs in the first timed step.
- **Per-track settings**: the jaxpi config of the same problem where one exists (Fourier scale, batch, lr decay,
  causal chunks and tolerance, initial weights); see each track's `solve.py`.
- **Execution**: `solve` is one open-ended training loop; every 50 iterations (one jitted `lax.fori_loop`,
  collocation points resampled every iteration) it checkpoints the parameters for the harness. Losses are read
  one chunk late, so the GPU never waits for the host. All functions are compiled ahead of time in `setup`.
  Matmuls run in true float32 (`jax_default_matmul_precision = highest`), not JAX's TF32 default: TF32
  caps the accuracy of the residual's derivatives (03 Burgers: L∞ 5e-3 with TF32, 6e-4 with float32 in 120 s).
- **Stopping**: `solve` returns once the loss stops improving by 2% for max(150 s, 25% of the elapsed time), or
  drops below 1e-10; otherwise the harness interrupts it at the track's budget (1000 s) and queries the last
  checkpoint. No reference is used.
- **Query**: one batched, compiled forward pass of the network on the output grid.

Differences from the jaxpi runs, forced by the budget: one time window (jaxpi time-marches 04, 15–18 over 4–25
windows of 2·10⁴–2·10⁵ iterations each), no Reynolds-number curriculum for 11, grad-norm weighting everywhere
(jaxpi uses NTK weights for Allen-Cahn), and SOAP for 06 (jaxpi's KdV config uses Adam).

Speed vs. the earlier PyTorch port (05 Allen-Cahn, 60 s, one H100): PyTorch 27.9 it/s at batch 4096,
JAX ≈ 63 it/s at batch 8192 (≈ 4.5× the collocation points per second).
