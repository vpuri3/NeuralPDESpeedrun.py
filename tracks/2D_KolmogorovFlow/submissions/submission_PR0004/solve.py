"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Forced 2D Navier-Stokes (Kolmogorov flow, Re = 1e4), x, y periodic (exact,
through (sin, cos) inputs): one network on the whole space-time domain, outputs (u, v, p); the initial velocity is
residual rows at redrawn points, against the trigonometric interpolant of the given samples (pinn.trig_interp)."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve, trig_interp  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def ns2(u, z, nu, ix, iy, it=None, force=(0.0, 0.0)):
    """Incompressible NS residuals (x, y momentum, continuity); u(z) = (u, v, ..., p), p last."""
    W, W_x, W_xx = derivs(u, z, ix, 2)
    _, W_y, W_yy = derivs(u, z, iy, 2)
    W_t = derivs(u, z, it, 1)[1] if it is not None else jnp.zeros_like(W)
    a, b = W[0], W[1]
    mx = W_t[0] + a * W_x[0] + b * W_y[0] + W_x[-1] - nu * (W_xx[0] + W_yy[0]) - force[0]
    my = W_t[1] + a * W_x[1] + b * W_y[1] + W_y[-1] - nu * (W_xx[1] + W_yy[1]) - force[1]
    return jnp.stack([mx, my, W_x[0] + W_y[1]])


def setup(problem):
    nu, ic = problem.params["nu"], problem.fns["ic"]
    grid, dom = [ic["x"].cpu().numpy(), ic["y"].cpu().numpy()], [problem.domain["x"], problem.domain["y"]]
    u0, v0 = (trig_interp(ic[k].cpu().numpy(), grid, dom) for k in ("u", "v"))

    def residual(u, z):
        return ns2(u, z, nu, 1, 2, it=0, force=(2.0 * jnp.sin(4 * jnp.pi * z[2]), 0.0))

    ic = Cond("IC (u, v)(0, x, y)", lambda u, z: u(z)[:2] - jnp.stack([u0(z[1], z[2]), v0(z[1], z[2])]),
              face=("t", 0.0))
    return LMPINN(problem, residual, CONFIG, [ic], n_out=3).compile()
