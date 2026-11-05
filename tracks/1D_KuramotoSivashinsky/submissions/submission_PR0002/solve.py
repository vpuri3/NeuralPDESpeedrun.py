"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002).
Kuramoto-Sivashinsky with JAX-NG's ks1d setup (20 windows, sine MLP 50 x 4 on (t, 10 Fourier modes of x)), 200 GN
steps per window, hard IC from the trigonometric interpolant of the given samples (as numpy constants), float64."""
import math

from speedrun.submissions.submission_PR0002.net import partials, periodic_interp
from speedrun.submissions.submission_PR0002.pinn import PINN, Config, query, solve  # noqa: F401


def setup(problem):
    a, b, c = (problem.params[k] for k in ("a", "b", "c"))
    u0 = periodic_interp(problem.fns["ic"]["u"], 0.0, 2 * math.pi)

    def residual(U, z):
        u, u_x, u_xx, _, u_xxxx = partials(U, z, 1, 4)
        return partials(U, z, 0, 1)[1] + a * u * u_x + b * u_xx + c * u_xxxx

    cfg = Config(n_windows=20, width=50, depth=4, n_modes=10, n_colloc=20000, steps=200)
    return PINN(problem, residual, 1, cfg, ic=(u0,)).compile()
