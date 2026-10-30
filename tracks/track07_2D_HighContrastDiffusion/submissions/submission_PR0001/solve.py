"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). High-contrast diffusion:
-Lap u = f / a inside each constant-coefficient region, u = 0 on the boundary (the flux conditions at the
inclusion edges are not imposed explicitly)."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve, to_jax  # noqa: F401

def setup(problem):
    f = problem.fns

    def losses(m, params, key):
        u = m.u(0)
        kr, kb = jax.random.split(key)

        def r(params, x, y):
            return -(d(u, 1, 1)(params, x, y) + d(u, 2, 2)(params, x, y)) - f["forcing"](x, y) / f["a"](x, y)

        x, y = m.sample(kr, m.cfg.batch)
        xb, yb = m.faces(kb, 2048, ["x", "y"])
        return {"res": jnp.mean(jax.vmap(r, (None, 0, 0))(params, x, y) ** 2),
                "bcs": jnp.mean(jax.vmap(u, (None, 0, 0))(params, xb, yb) ** 2)}

    cfg = Config(embed_scale=2.0, batch=8192, decay_steps=2000)
    return PINN(problem, ("x", "y"), 1, losses, cfg).compile()
