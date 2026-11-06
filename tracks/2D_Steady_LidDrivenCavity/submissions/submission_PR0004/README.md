# 2D_Steady_LidDrivenCavity / submission_PR0004: Shallow PINN + Levenberg-Marquardt

Method and every explicit choice: [speedrun/submissions/submission_PR0004/README.md](../../../../speedrun/submissions/submission_PR0004/README.md)
(after Shahab et al., arXiv:2602.08515). The residual, the conditions and the hyperparameters for this track are at
the top of `solve.py`.

What it is: one shallow tanh MLP (32×4) on the whole domain (no time windows), trained by Levenberg-Marquardt on
the stacked residual (μI damping, no preconditioning, float64), with initial and boundary conditions as residual
rows (only periodicity is built in) and all residual points redrawn every 50 iterations.
