"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). KdV u_t + u u_x + delta^2 u_xxx = 0, x periodic (exact, through (sin, cos)(pi x) inputs): one
network on the whole space-time domain, u = u~(t, x); the initial state is residual rows."""
from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    d2, u0 = problem.params["delta"] ** 2, problem.fns["u0"]

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, u_x, _, u_xxx = derivs(u, z, 1, 3)
        return u_t + u_ * u_x + d2 * u_xxx

    ic = Cond("IC u(0, x)", lambda u, z: u(z) - u0(z[1]), face=("t", 0.0))
    return LMPINN(problem, residual, CONFIG, [ic]).compile()
