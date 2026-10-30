"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). 2D Burgers: residuals with causal
weights + IC; exact periodicity in x and y; physics-informed initialization from the IC."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, step, to_jax  # noqa: F401

def setup(problem):
    nu, f = problem.params["nu"], problem.fns
    t0 = problem.domain["t"][0]
    gx, gy = to_jax(problem.axes["x"])[::4], to_jax(problem.axes["y"])[::4]
    XI, YI = (a.ravel() for a in jnp.meshgrid(gx, gy, indexing="ij"))

    def r(m, params, t, x, y):
        Y, J, H = jet(m, params, t, x, y)
        u, v = Y
        ru = J[0, 0] + u * J[0, 1] + v * J[0, 2] - nu * (H[0, 1, 1] + H[0, 2, 2])
        rv = J[1, 0] + u * J[1, 1] + v * J[1, 2] - nu * (H[1, 1, 1] + H[1, 2, 2])
        return ru**2 + rv**2

    def losses(m, params, key):
        kr, ki = jax.random.split(key)
        t, x, y = m.sample(kr, m.cfg.batch)
        r2 = jax.vmap(r, (None, None, 0, 0, 0))(m, params, t, x, y)
        _, xi, yi = m.sample(ki, 2048, t=t0)
        Yi = jax.vmap(m.apply, (None, None, 0, 0))(params, t0, xi, yi)
        ics = jnp.mean((Yi[:, 0] - f["u0"](xi, yi)) ** 2 + (Yi[:, 1] - f["v0"](xi, yi)) ** 2)
        return {"ics": ics, "res": m.causal(r2, t)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000, num_chunks=32, causal_tol=1.0, pi_init=True)
    ic = lambda m: m.ic_tiled([XI, YI], jnp.stack([f["u0"](XI, YI), f["v0"](XI, YI)], 1))  # noqa: E731
    return PINN(problem, ("t", "x", "y"), 2, losses, cfg, ic_data=ic).compile()
