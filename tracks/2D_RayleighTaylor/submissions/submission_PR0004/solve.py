"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Boussinesq Rayleigh-Taylor (Ra = 1e6), x periodic (exact, through (sin, cos)
inputs): one network on the whole space-time domain, outputs (u, v, T, p); u = v = T = 0 on the walls y = 0, 2 and
the initial (u, v, T) are residual rows (the initial state is only given as samples on a non-uniform grid: a fixed
random subset of N_IC of them)."""
import jax.numpy as jnp
import numpy as np

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)
N_IC = 2000                 # initial-state samples used (fixed: they are the data)


def setup(problem):
    p, ic = problem.params, problem.fns["ic"]
    a1, a2, a4 = p["a1"], p["a2"], p["a4"]
    y0, y1 = problem.domain["y"]

    def residual(u, z):
        W_t = derivs(u, z, 0, 1)[1]
        W, W_x, W_xx = derivs(u, z, 1, 2)
        _, W_y, W_yy = derivs(u, z, 2, 2)
        a, b = W[0], W[1]
        adv = W_t + a * W_x + b * W_y
        lap = W_xx + W_yy
        return jnp.stack([adv[0] + W_x[3] - a1 * lap[0], adv[1] + W_y[3] - a1 * lap[1] - a2 * W[2],
                          W_x[0] + W_y[1], adv[2] - a4 * lap[2]])

    X, Y = np.meshgrid(ic["x"].cpu().numpy(), ic["y"].cpu().numpy(), indexing="ij")
    vals = [ic[k].cpu().numpy().ravel() for k in ("u", "v", "T")]
    pick = np.random.default_rng(0).choice(X.size, N_IC, replace=False)
    data = np.stack([np.zeros(N_IC), X.ravel()[pick], Y.ravel()[pick]] + [v[pick] for v in vals], -1)
    wall = lambda u, z: u(z)[:3]
    conds = [Cond("walls y = 0: u = v = T = 0", wall, face=("y", y0)),
             Cond("walls y = 2: u = v = T = 0", wall, face=("y", y1)),
             Cond("IC (u, v, T) (given samples)", lambda u, z: u(z[:3])[:3] - z[3:6], points=data)]
    return LMPINN(problem, residual, CONFIG, conds, n_out=4).compile()
