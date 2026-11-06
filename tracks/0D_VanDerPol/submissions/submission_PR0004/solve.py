"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Stiff Van der Pol y'' - mu (1 - y^2) y' + y = 0 on [0, 3000]: one
network on the whole interval, y = y~(t); y(0) and y'(0) are residual rows."""
import jax.numpy as jnp
import numpy as np

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    mu, y0, dy0 = problem.params["mu"], problem.fns["y0"], problem.fns["dy0"]

    def residual(u, z):
        y, y_t, y_tt = derivs(u, z, 0, 2)
        return y_tt - mu * (1 - y ** 2) * y_t + y

    ic = Cond("IC y(0) = 2, y'(0) = 0", lambda u, z: jnp.stack([u(z) - y0, derivs(u, z, 0, 1)[1] - dy0]),
              points=np.array([[0.0]]))
    return LMPINN(problem, residual, CONFIG, [ic]).compile()
