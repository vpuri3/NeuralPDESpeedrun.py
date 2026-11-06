# 1D_Steady_AdvectionDiffusion / submission_PR0005: shallow PINN + LM with QR steps

Method: [speedrun/submissions/submission_PR0005/README.md](../../../../speedrun/submissions/submission_PR0005/README.md).
The residual, conditions and hyperparameters are at the top of `solve.py` (the same as submission 4: 32×4, points
redrawn every 50 iterations).

What changed vs. the previous record (submission 2, Gauss-Newton on deeper sine MLPs) and vs. submission 4: the
Levenberg-Marquardt step is computed by a QR factorization of the stacked system [J; √μI] instead of the normal
equations, so JᵀJ is never formed and the linear solve's condition number is √κ instead of κ.
