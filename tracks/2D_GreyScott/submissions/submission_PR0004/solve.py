"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Grey-Scott reaction-diffusion, x, y periodic (exact, through (sin, cos)
inputs): one network on the whole space-time domain, outputs (u, v); the initial state is residual rows at
redrawn points, against the trigonometric interpolant of the given samples (pinn.trig_interp)."""
import jax.numpy as jnp

from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve, trig_interp  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def react(W, lap, p):
    u, v = W[0], W[1]
    return jnp.stack([p["eps1"] * lap[0] + p["b1"] * (1 - u) - p["c1"] * u * v ** 2,
                      p["eps2"] * lap[1] - p["b2"] * v + p["c2"] * u * v ** 2])


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
