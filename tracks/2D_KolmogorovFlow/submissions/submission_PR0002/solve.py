"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Kolmogorov
flow (forced periodic NS, Re 1e4): outputs (u, v, p), hard IC on (u, v) from a network fitted to the given
samples, residuals: momentum and continuity, float64."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def ns2(U, z, nu, t_axis=True, force=(0.0, 0.0)):
    """Incompressible NS residuals (x, y momentum, continuity); U = (u, v, ..., p) with p last."""
    W, W_x, W_xx = partials(U, z, 1 if t_axis else 0, 2)
    _, W_y, W_yy = partials(U, z, 2 if t_axis else 1, 2)
    W_t = partials(U, z, 0, 1)[1] if t_axis else jnp.zeros_like(W)
    u, v = W[0], W[1]
    mx = W_t[0] + u * W_x[0] + v * W_y[0] + W_x[-1] - nu * (W_xx[0] + W_yy[0]) - force[0]
    my = W_t[1] + u * W_x[1] + v * W_y[1] + W_y[-1] - nu * (W_xx[1] + W_yy[1]) - force[1]
    return jnp.stack([mx, my, W_x[0] + W_y[1]])

def setup(problem):
    nu, ic = problem.params["nu"], problem.fns["ic"]
    data = {"axes": [ic["x"], ic["y"]], "values": jnp.stack([jnp.asarray(ic[k].cpu().numpy()) for k in ("u", "v")])}

    def residual(U, z):
        return ns2(U, z, nu, force=(2.0 * jnp.sin(4 * jnp.pi * z[2]), 0.0))

    cfg = Config(gn=GNConfig(lm_max=0.1), n_windows=10, width=50, depth=4, n_modes=8, n_colloc=8000, steps=300, ic_steps=100, ic_width=64)
    return PINN(problem, residual, 3, cfg, ic_data=data).compile()
