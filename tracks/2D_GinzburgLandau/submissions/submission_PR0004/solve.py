"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). complex Ginzburg-Landau (A = u + iv), x, y periodic (exact, through (sin, cos)
inputs): one network on the whole space-time domain, outputs (u, v); the initial state is residual rows at
redrawn points, against the trigonometric interpolant of the given samples (pinn.trig_interp)."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve, trig_interp  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def react(W, lap, p):
    u, v = W[0], W[1]
    r2 = u ** 2 + v ** 2
    return jnp.stack([p["eps"] * lap[0] + p["k"] * (u - r2 * u + 1.5 * r2 * v),
                      p["eps"] * lap[1] + p["k"] * (v - r2 * v - 1.5 * r2 * u)])


def setup(problem):
    pr, ic = problem.params, problem.fns["ic"]
    grid, dom = [ic["x"].cpu().numpy(), ic["y"].cpu().numpy()], [problem.domain["x"], problem.domain["y"]]
    u0, v0 = (trig_interp(ic[k].cpu().numpy(), grid, dom) for k in ("u", "v"))

    def residual(u, z):
        W_t = derivs(u, z, 0, 1)[1]
        W, _, W_xx = derivs(u, z, 1, 2)
        W_yy = derivs(u, z, 2, 2)[2]
        return W_t - react(W, W_xx + W_yy, pr)

    ic = Cond("IC (u, v)(0, x, y)", lambda u, z: u(z) - jnp.stack([u0(z[1], z[2]), v0(z[1], z[2])]),
              face=("t", 0.0))
    return LMPINN(problem, residual, CONFIG, [ic], n_out=2).compile()
