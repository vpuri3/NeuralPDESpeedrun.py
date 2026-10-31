"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Beltrami
flow (3D NS): outputs (u, v, w, p), hard IC on (u, v, w), soft Dirichlet velocity on the six faces, float64."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    nu, f = problem.params["nu"], problem.fns
    bc = (f["u_bc"], f["v_bc"], f["w_bc"])

    def residual(U, z):
        W_t = partials(U, z, 0, 1)[1]
        W, W_x, W_xx = partials(U, z, 1, 2)
        _, W_y, W_yy = partials(U, z, 2, 2)
        _, W_z, W_zz = partials(U, z, 3, 2)
        m = W_t[:3] + W[0] * W_x[:3] + W[1] * W_y[:3] + W[2] * W_z[:3] - nu * (W_xx + W_yy + W_zz)[:3]
        m = m + jnp.stack([W_x[3], W_y[3], W_z[3]])
        return jnp.concatenate([m, (W_x[0] + W_y[1] + W_z[2])[None]])

    def wall(U, z):
        return U(z)[:3] - jnp.stack([g(z[0], z[1], z[2], z[3]) for g in bc])

    cfg = Config(n_windows=2, width=50, depth=4, n_colloc=6000, n_bc=1000, steps=100)
    return PINN(problem, residual, 4, cfg, ic=(f["u0"], f["v0"], f["w0"]),
                bcs=[BC(a, wall, weight=10.0) for a in ("x", "y", "z")]).compile()
