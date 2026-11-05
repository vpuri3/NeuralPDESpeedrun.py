# 1D_KuramotoSivashinsky / submission_PR0004: Shallow PINN + Levenberg-Marquardt

Method: see [speedrun/submissions/submission_PR0004/README.md](../../../../speedrun/submissions/submission_PR0004/README.md)
(Shahab et al., arXiv:2602.08515). The residual, conditions and the network chosen for this track are in
`solve.py`; every architecture tried on this track, with loss and μ histories, is in the
[1D_KuramotoSivashinsky section](../../../../speedrun/submissions/submission_PR0004/README.md#1d_kuramotosivashinsky) there. What changed vs. the previous
record: one shallow tanh MLP on the whole space-time domain trained by textbook Levenberg-Marquardt (μI damping,
no preconditioning), float64, instead of time-windowed Gauss-Newton on deeper sine MLPs.
