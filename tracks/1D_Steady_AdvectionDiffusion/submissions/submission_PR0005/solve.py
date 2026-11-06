"""Submission 5: shallow PINN + Levenberg-Marquardt with QR steps (speedrun/submissions/submission_PR0005; Shahab et al.,
arXiv:2602.08515). Steady advection-diffusion -eps u'' + b u' = f on [0, 1], u = u~(x); the walls u(0) = u(1) = 0
are residual rows."""
import numpy as np

from speedrun.submissions.submission_PR0005.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    eps, b, f, g = problem.params["eps"], problem.params["b"], problem.fns["forcing"], problem.fns["bc"]

    def residual(u, z):
        _, u_x, u_xx = derivs(u, z, 0, 2)
        return -eps * u_xx + b * u_x - f(z[0])

    walls = Cond("walls u(0) = u(1) = 0", lambda u, z: u(z) - g(z[0]), points=np.array([[0.0], [1.0]]))
    return LMPINN(problem, residual, CONFIG, [walls]).compile()
