"""Submission 2: PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Lid-driven cavity
(steady NS, Re 5000): outputs (u, v, p), soft Dirichlet velocity on the four walls, float64."""
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
    nu, f = problem.params["nu"], problem.fns

    def wall(U, z):
        return U(z)[:2] - jnp.stack([f["u_bc"](z[0], z[1]), f["v_bc"](z[0], z[1])])

    cfg = Config(gn=GNConfig(lm_max=0.1), width=50, depth=4, n_colloc=6000, n_bc=1500, steps=6000, cluster_axes=("x", "y"))
    return PINN(problem, lambda U, z: ns2(U, z, nu, t_axis=False), 3, cfg,
                bcs=[BC(a, wall, weight=10.0) for a in ("x", "y")]).compile()
