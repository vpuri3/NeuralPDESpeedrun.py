# track01_ThirdOrderODE / submission_PR0001: vanilla PINN

- **Method:** standard tanh MLP (1 -> 4 x 64 -> 1), PDE residual by autograd, loss = mean-squared strong-form
  residual + the three boundary conditions (all weights 1), 1024 collocation points resampled uniformly every step.
  Adam, lr 1e-3 decaying geometrically to 1e-5 over the budget, fp32.
- **Stopping:** fixed wall-clock budget of 1000 s (no oracle).
- **What changed vs. the previous record:** first neural submission; the honest starting point.
