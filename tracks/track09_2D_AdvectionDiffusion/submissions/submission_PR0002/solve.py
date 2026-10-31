"""Submission 2: PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Advection-diffusion
(eps 1e-3): u = x (1 - x) y (1 - y) phi (exact zero walls), collocation clustered at the walls, float64."""

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    eps, f = problem.params["eps"], problem.fns["forcing"]
    bx, by = problem.params["b"]

    def residual(U, z):
        _, u_x, u_xx = partials(U, z, 0, 2)
        _, u_y, u_yy = partials(U, z, 1, 2)
        return -eps * (u_xx + u_yy) + bx * u_x + by * u_y - f(z[0], z[1])

    def hard(z, U):
        return z[0] * (1 - z[0]) * z[1] * (1 - z[1]) * U

    cfg = Config(gn=GNConfig(lm_max=0.1), width=50, depth=4, n_colloc=8000, steps=6000, cluster_axes=("x", "y"))
    return PINN(problem, residual, 1, cfg, hard=hard).compile()
