"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Damped wave u_tt = c^2 u_xx - gamma u_t on [0, 4] x [0, 1]: one network on the whole space-time
domain, u = u~(t, x); u(0, x), u_t(0, x) and the walls u(t, 0) = u(t, 1) = 0 are residual rows."""
from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    c2, gamma, fn = problem.params["c"] ** 2, problem.params["gamma"], problem.fns
    lo, hi = problem.domain["x"]

    def residual(u, z):
        _, u_t, u_tt = derivs(u, z, 0, 2)
        return u_tt - c2 * derivs(u, z, 1, 2)[2] + gamma * u_t

    conds = [Cond("IC u(0, x)", lambda u, z: u(z) - fn["u0"](z[1]), face=("t", 0.0)),
             Cond("IC u_t(0, x)", lambda u, z: derivs(u, z, 0, 1)[1] - fn["ut0"](z[1]), face=("t", 0.0)),
             Cond("wall x = 0", lambda u, z: u(z) - fn["bc_left"](z[0]), face=("x", lo)),
             Cond("wall x = 1", lambda u, z: u(z) - fn["bc_right"](z[0]), face=("x", hi))]
    return LMPINN(problem, residual, CONFIG, conds).compile()
