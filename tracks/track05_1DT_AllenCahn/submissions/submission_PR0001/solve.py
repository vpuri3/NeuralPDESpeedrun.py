"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Allen-Cahn with jaxpi's
allen_cahn config: Fourier scale 2, batch 8192, lr decay every 5000 iterations, 32 causal chunks (tol 1),
physics-informed initialization from the IC; exact periodicity in x."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, query, step, to_jax  # noqa: F401


def setup(problem):
    eps, kappa, u0 = problem.params["eps"], problem.params["kappa"], problem.fns["u0"]
    t0 = problem.domain["t"][0]
    x_ic = to_jax(problem.axes["x"])

    def losses(m, params, key):
        u = m.u(0)

        def r(params, t, x):
            v = u(params, t, x)
            return d(u, 1)(params, t, x) - eps * d(u, 2, 2)(params, t, x) + kappa * (v**3 - v)

        t, x = m.sample(key, m.cfg.batch)
        r2 = jax.vmap(r, (None, 0, 0))(params, t, x) ** 2
        ui = jax.vmap(u, (None, None, 0))(params, t0, x_ic)
        return {"ics": jnp.mean((ui - u0(x_ic)) ** 2), "res": m.causal(r2, t)}

    cfg = Config(embed_scale=2.0, batch=8192, decay_steps=5000, num_chunks=32, causal_tol=1.0, pi_init=True)
    return PINN(problem, ("t", "x"), 1, losses, cfg, ic_data=lambda m: m.ic_tiled([x_ic], u0(x_ic)[:, None])).compile()
