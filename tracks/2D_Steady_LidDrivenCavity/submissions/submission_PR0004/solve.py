"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Steady Navier-Stokes lid-driven cavity, Re = 5000, outputs (u, v, p). Residual points: stratified
on the 256 x 256 output grid (problem.axes): the PDE (momentum and continuity) at one uniform random point inside
each of the 255^2 grid cells, the lid profile and no-slip walls at one uniform random point on each of the 4 x 255
wall segments; all redrawn every 50 iterations. (Fixed grid nodes let the network satisfy the PDE at the nodes with
u = 0 inside and hide the jump to the lid between the last two node rows: RelL2 98%.)"""
import jax
import jax.numpy as jnp
import numpy as np

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config); points: the output grid, fixed
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
    nu, f = problem.params["nu"], problem.fns
    x, y = (problem.axes[k].detach().cpu().numpy() for k in ("x", "y"))
    (x0, x1), (y0, y1) = problem.domain["x"], problem.domain["y"]
    CX, CY = np.meshgrid(x[:-1], y[:-1], indexing="ij")             # lower-left corners of the 255^2 cells
    HX, HY = np.meshgrid(np.diff(x), np.diff(y), indexing="ij")
    cell = np.stack([CX.ravel(), CY.ravel()], -1)
    size = np.stack([HX.ravel(), HY.ravel()], -1)

    def interior(key):                                              # one uniform point per cell
        return jnp.asarray(cell) + jnp.asarray(size) * jax.random.uniform(key, cell.shape, jnp.float64)

    def segments(axis, value):                                      # one uniform point per wall segment
        e = (y if axis == "x" else x)
        lo_, h = jnp.asarray(e[:-1]), jnp.asarray(np.diff(e))

        def sample(key):
            t = lo_ + h * jax.random.uniform(key, lo_.shape, jnp.float64)
            c = jnp.full_like(t, value)
            return jnp.stack([c, t] if axis == "x" else [t, c], -1)
        return sample

    wall = lambda u, z: u(z)[:2] - jnp.stack([f["u_bc"](z[0], z[1]), f["v_bc"](z[0], z[1])])
    conds = [Cond(f"wall {a} = {v:g}" + (" (lid)" if (a, v) == ("y", y1) else ""), wall, sample=segments(a, v))
             for a, v in (("x", x0), ("x", x1), ("y", y0), ("y", y1))]
    return LMPINN(problem, lambda u, z: ns2(u, z, nu, 0, 1), CONFIG, conds, n_out=3, interior=interior).compile()
