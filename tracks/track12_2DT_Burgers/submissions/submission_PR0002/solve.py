"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). 2D Burgers
(periodic): outputs (u, v), hard IC, float64."""

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    nu = problem.params["nu"]

    def residual(U, z):
        W_t = partials(U, z, 0, 1)[1]
        W, W_x, W_xx = partials(U, z, 1, 2)
        _, W_y, W_yy = partials(U, z, 2, 2)
        return W_t + W[0] * W_x + W[1] * W_y - nu * (W_xx + W_yy)

    cfg = Config(gn=GNConfig(lm_max=0.1), n_windows=10, width=50, depth=4, n_modes=10, n_colloc=8000, steps=300)
    return PINN(problem, residual, 2, cfg, ic=(problem.fns["u0"], problem.fns["v0"])).compile()
