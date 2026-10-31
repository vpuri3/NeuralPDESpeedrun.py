"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Burgers:
5 windows of a sine MLP 40 x 4 on (t, 10 Fourier modes of x), hard IC, float64."""
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import PINN, Config, query, solve  # noqa: F401


def setup(problem):
    nu = problem.params["nu"]

    def residual(U, z):
        u, u_x, u_xx = partials(U, z, 1, 2)
        return partials(U, z, 0, 1)[1] + u * u_x - nu * u_xx

    cfg = Config(n_windows=5, width=40, depth=4, n_modes=10, n_colloc=20000, steps=100)
    return PINN(problem, residual, 1, cfg, ic=(problem.fns["u0"],)).compile()
