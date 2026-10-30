"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Decaying eddies: unsteady NS
residuals with causal weights + IC; exact periodicity in x and y; outputs (u, v, p); physics-informed
initialization of (u, v) from the IC."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve, to_jax  # noqa: F401
def ns_unsteady(m, params, t, x, y, nu):
    """Unsteady incompressible NS residuals (ru, rv, rc) of the outputs (u, v, p, ...) at one point (t, x, y)."""
    Y, J, H = jet(m, params, t, x, y)
    u, v = Y[0], Y[1]
    ru = J[0, 0] + u * J[0, 1] + v * J[0, 2] + J[2, 1] - nu * (H[0, 1, 1] + H[0, 2, 2])
    rv = J[1, 0] + u * J[1, 1] + v * J[1, 2] + J[2, 2] - nu * (H[1, 1, 1] + H[1, 2, 2])
    return ru, rv, J[0, 1] + J[1, 2], (Y, J, H)

def setup(problem):
    nu, f = problem.params["nu"], problem.fns
    t0 = problem.domain["t"][0]
    g = to_jax(problem.axes["x"])
    XI, YI = (a.ravel() for a in jnp.meshgrid(g, g, indexing="ij"))

    def losses(m, params, key):
        kr, ki = jax.random.split(key)
        t, x, y = m.sample(kr, m.cfg.batch)
        ru, rv, rc, _ = jax.vmap(ns_unsteady, (None, None, 0, 0, 0, None))(m, params, t, x, y, nu)
        _, xi, yi = m.sample(ki, 2048, t=t0)
        Yi = jax.vmap(m.apply, (None, None, 0, 0))(params, t0, xi, yi)
        ics = jnp.mean((Yi[:, 0] - f["u0"](xi, yi)) ** 2 + (Yi[:, 1] - f["v0"](xi, yi)) ** 2)
        return {"ics": ics, "ru": m.causal(ru**2, t), "rv": m.causal(rv**2, t), "rc": m.causal(rc**2, t)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000, num_chunks=16, causal_tol=1.0, pi_init=True)
    ic = lambda m: m.ic_tiled([XI, YI], jnp.stack([f["u0"](XI, YI), f["v0"](XI, YI)], 1))  # noqa: E731
    return PINN(problem, ("t", "x", "y"), 3, losses, cfg, ic_data=ic).compile()
