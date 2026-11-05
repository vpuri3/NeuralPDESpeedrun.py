"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Steady 1D advection-diffusion
-eps u'' + b u' = f with an outflow boundary layer: residual + homogeneous Dirichlet walls."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve  # noqa: F401


def setup(problem):
    eps, b, f = problem.params["eps"], problem.params["b"], problem.fns

    def losses(m, params, key):
        u = m.u(0)
        kr, kb = jax.random.split(key)

        def r(params, x):
            return -eps * d(u, 1, 1)(params, x) + b * d(u, 1)(params, x) - f["forcing"](x)

        (x,) = m.sample(kr, m.cfg.batch)
        (xb,) = m.faces(kb, 256, ["x"])
        return {"res": jnp.mean(jax.vmap(r, (None, 0))(params, x) ** 2),
                "bcs": jnp.mean(jax.vmap(u, (None, 0))(params, xb) ** 2)}

    cfg = Config(embed_scale=10.0, batch=4096, decay_steps=2000)
    return PINN(problem, ("x",), 1, losses, cfg).compile()
