"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002).
Allen-Cahn: 10 windows of a sine MLP 50 x 4 on (t, 10 Fourier modes of x), hard IC, float64."""
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import PINN, Config, query, solve  # noqa: F401


def setup(problem):
    eps, kappa = problem.params["eps"], problem.params["kappa"]

    def residual(U, z):
        u, _, u_xx = partials(U, z, 1, 2)
        return partials(U, z, 0, 1)[1] - eps * u_xx + kappa * (u ** 3 - u)

    cfg = Config(n_windows=10, width=50, depth=4, n_modes=10, n_colloc=20000, steps=100)
    return PINN(problem, residual, 1, cfg, ic=(problem.fns["u0"],)).compile()
