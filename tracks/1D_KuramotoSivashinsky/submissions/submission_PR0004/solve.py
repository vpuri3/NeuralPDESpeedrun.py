"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Chaotic Kuramoto-Sivashinsky u_t + a u u_x + b u_xx + c u_xxxx = 0, x periodic (exact, through
(sin, cos)(x) inputs): one network on the whole space-time domain, u = u~(t, x); the initial state is residual rows
at its given samples (fixed: they are the data)."""
import numpy as np

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    a, b, c = (problem.params[k] for k in "abc")
    xs = problem.fns["ic"]["x"].detach().cpu().numpy()
    u0 = np.asarray(problem.fns["ic"]["u"].detach().cpu().numpy(), np.float64)

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, u_x, u_xx, _, u_xxxx = derivs(u, z, 1, 4)
        return u_t + a * u_ * u_x + b * u_xx + c * u_xxxx

    data = np.stack([np.zeros_like(xs), xs, u0], -1)      # rows (t = 0, x, u0): the target rides along
    ic = Cond("IC u(0, x) (given samples)", lambda u, z: u(z[:2]) - z[2], points=data)
    return LMPINN(problem, residual, CONFIG, [ic]).compile()
