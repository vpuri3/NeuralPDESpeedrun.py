"""Submission 4: shallow PINN trained by Levenberg-Marquardt (speedrun/submissions/submission_PR0004), after
Shahab et al. (arXiv:2602.08515). Chaotic Kuramoto-Sivashinsky u_t + a u u_x + b u_xx + c u_xxxx = 0, x periodic
(exact, through (sin, cos)(x) inputs): one tanh MLP on the whole space-time domain, u = u~(t, x); the initial
state is residual rows at its given samples. 10k collocation points redrawn every LM iteration, float64."""
import numpy as np

from speedrun.submissions.submission_PR0004.pinn import Config, LMPINN, arch, derivs, face, query, solve  # noqa: F401


def setup(problem):
    a, b, c = (problem.params[k] for k in "abc")
    xs = problem.fns["ic"]["x"].detach().cpu().numpy()
    u0 = np.asarray(problem.fns["ic"]["u"].detach().cpu().numpy(), np.float64)

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, u_x, u_xx, _, u_xxxx = derivs(u, z, 1, 4)
        return u_t + a * u_ * u_x + b * u_xx + c * u_xxxx

    ic = np.stack([np.zeros_like(xs), xs], -1)
    data = np.concatenate([ic, u0[:, None]], -1)          # rows (t, x, u0): the target rides along with the point
    bcs = [(data, lambda u, z: u(z[:2]) - z[2], 1.0)]
    width, layers = arch(32, 2)
    return LMPINN(problem, residual, Config(width=width, layers=layers, resample=True, max_iters=None), bcs=bcs).compile()
