"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Allen-Cahn u_t - eps u_xx + kappa (u^3 - u) = 0, x periodic (exact, through (sin, cos)(pi x)
inputs): one network on the whole space-time domain, u = u~(t, x); the initial state is residual rows."""
from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    eps, kappa, u0 = problem.params["eps"], problem.params["kappa"], problem.fns["u0"]

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, _, u_xx = derivs(u, z, 1, 2)
        return u_t - eps * u_xx + kappa * (u_ ** 3 - u_)

    ic = Cond("IC u(0, x)", lambda u, z: u(z) - u0(z[1]), face=("t", 0.0))
    return LMPINN(problem, residual, CONFIG, [ic]).compile()
