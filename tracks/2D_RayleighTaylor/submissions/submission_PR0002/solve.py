"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002).
Rayleigh-Taylor (Boussinesq, x periodic, walls at y = 0, 2): outputs (u, v, T, p), hard IC on (u, v, T) from a
network fitted to the given samples, soft wall conditions u = v = T = 0, float64."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    p, ic = problem.params, problem.fns["ic"]
    a1, a2, a4 = p["a1"], p["a2"], p["a4"]
    data = {"axes": [ic["x"], ic["y"]],
            "values": jnp.stack([jnp.asarray(ic[k].cpu().numpy()) for k in ("u", "v", "T")])}

    def residual(U, z):
        W_t = partials(U, z, 0, 1)[1]
        W, W_x, W_xx = partials(U, z, 1, 2)
        _, W_y, W_yy = partials(U, z, 2, 2)
        u, v = W[0], W[1]
        adv = W_t + u * W_x + v * W_y
        lap = W_xx + W_yy
        return jnp.stack([adv[0] + W_x[3] - a1 * lap[0], adv[1] + W_y[3] - a1 * lap[1] - a2 * W[2],
                          W_x[0] + W_y[1], adv[2] - a4 * lap[2]])

    walls = BC("y", lambda U, z: U(z)[:3], weight=10.0)
    cfg = Config(gn=GNConfig(lm_max=0.1), n_windows=10, width=50, depth=4, n_modes=8, n_colloc=8000, n_bc=2000, steps=200, ic_steps=100,
                 ic_width=64)
    return PINN(problem, residual, 4, cfg, ic_data=data, bcs=[walls]).compile()
