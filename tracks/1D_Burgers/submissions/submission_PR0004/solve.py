"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Raissi's Burgers u_t + u u_x = nu u_xx on [0, 1] x [-1, 1]: one network on the whole space-time
domain, u = u~(t, x); the initial state and the walls u(t, +-1) = 0 are residual rows."""
from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    nu, fn = problem.params["nu"], problem.fns

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, u_x, u_xx = derivs(u, z, 1, 2)
        return u_t + u_ * u_x - nu * u_xx

    conds = [Cond("IC u(0, x)", lambda u, z: u(z) - fn["u0"](z[1]), face=("t", 0.0)),
             Cond("wall x = -1", lambda u, z: u(z) - fn["bc_left"](z[0]), face=("x", -1.0)),
             Cond("wall x = 1", lambda u, z: u(z) - fn["bc_right"](z[0]), face=("x", 1.0))]
    return LMPINN(problem, residual, CONFIG, conds).compile()
