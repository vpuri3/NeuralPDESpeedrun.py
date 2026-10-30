# track03_HighContrastDiffusion / submission_PR0001: vanilla PINN

- **Method:** standard tanh MLP ((x, y) -> 4 x 64 -> u), PDE residual by autograd, loss = mean-squared strong-form
  residual + the Dirichlet loss (all weights 1), 4096 collocation points resampled uniformly every step.
  Adam, lr 1e-3 decaying geometrically to 1e-5 over the budget, fp32.
- **Stopping:** fixed wall-clock budget of 1000 s (no oracle).
- **What changed vs. the previous record:** first neural submission; the honest starting point.
