"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). KdV with
JAX-NG's kdv_windowed_gn setup: 5 windows, sine MLP 30 x 4 on (t, 10 Fourier modes of x), 20k collocation
points per step, hard IC, float64."""
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import PINN, Config, query, solve  # noqa: F401


def setup(problem):
    d2 = problem.params["delta"] ** 2

    def residual(U, z):
        u, u_x, _, u_xxx = partials(U, z, 1, 3)
        return partials(U, z, 0, 1)[1] + u * u_x + d2 * u_xxx

    cfg = Config(n_windows=5, width=30, depth=4, n_modes=10, n_colloc=20000, steps=100)
    return PINN(problem, residual, 1, cfg, ic=(problem.fns["u0"],)).compile()
