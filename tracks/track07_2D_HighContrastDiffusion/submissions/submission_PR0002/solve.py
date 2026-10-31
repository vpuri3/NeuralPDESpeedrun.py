"""Submission 2: PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). High-contrast diffusion:
u = x (1 - x) y (1 - y) phi (exact zero walls), residual Lap u + f / a (piecewise-constant a), float64."""

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    a, f = problem.fns["a"], problem.fns["forcing"]

    def residual(U, z):
        lap = partials(U, z, 0, 2)[2] + partials(U, z, 1, 2)[2]
        return lap + f(z[0], z[1]) / a(z[0], z[1])

    def hard(z, U):
        return z[0] * (1 - z[0]) * z[1] * (1 - z[1]) * U

    cfg = Config(gn=GNConfig(lm_max=0.1), width=50, depth=4, n_colloc=8000, steps=6000)
    return PINN(problem, residual, 1, cfg, hard=hard).compile()
