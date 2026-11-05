"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Raissi's
1D Burgers: u = 0 on both walls by an odd periodic extension in x, hard IC, 10 windows of a sine MLP 50 x 4 on
(t, 30 Fourier modes of x), float64."""
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import PINN, Config, query, solve  # noqa: F401


def setup(problem):
    nu = problem.params["nu"]

    def residual(U, z):
        u, u_x, u_xx = partials(U, z, 1, 2)
        return partials(U, z, 0, 1)[1] + u * u_x - nu * u_xx

    cfg = Config(n_windows=10, width=50, depth=4, n_modes=30, n_colloc=20000, steps=100, odd_axes=("x",))
    return PINN(problem, residual, 1, cfg, ic=(problem.fns["u0"],)).compile()
