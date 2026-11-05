"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002).
Ginzburg-Landau (periodic): outputs (u, v), hard IC from a network fitted to the given samples, float64."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def REACT(W, lap, p):
    u, v = W[0], W[1]
    r2 = u ** 2 + v ** 2
    return jnp.stack([p["eps"] * lap[0] + p["k"] * (u - r2 * u + 1.5 * r2 * v),
                      p["eps"] * lap[1] + p["k"] * (v - r2 * v - 1.5 * r2 * u)])


def setup(problem):
    pr, ic = problem.params, problem.fns["ic"]
    data = {"axes": [ic["x"], ic["y"]], "values": jnp.stack([jnp.asarray(ic[k].cpu().numpy()) for k in ("u", "v")])}

    def residual(U, z):
        W_t = partials(U, z, 0, 1)[1]
        W, _, W_xx = partials(U, z, 1, 2)
        W_yy = partials(U, z, 2, 2)[2]
        return W_t - REACT(W, W_xx + W_yy, pr)

    cfg = Config(gn=GNConfig(lm_max=0.1), n_windows=10, width=50, depth=4, n_modes=10, n_colloc=8000, steps=300, ic_steps=100, ic_width=64)
    return PINN(problem, residual, 2, cfg, ic_data=data).compile()
