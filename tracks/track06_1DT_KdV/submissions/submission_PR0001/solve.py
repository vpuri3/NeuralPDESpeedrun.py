"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). KdV with jaxpi's kdv config
(SOAP instead of the config's Adam): Fourier scale 1, batch 4096, 16 causal chunks (tol 0.1),
physics-informed initialization from the IC; exact periodicity in x."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve, to_jax  # noqa: F401

def setup(problem):
    delta, u0 = problem.params["delta"], problem.fns["u0"]
    t0 = problem.domain["t"][0]
    x_ic = to_jax(problem.axes["x"])

    def losses(m, params, key):
        u = m.u(0)

        def r(params, t, x):
            return (d(u, 1)(params, t, x) + u(params, t, x) * d(u, 2)(params, t, x)
                    + delta**2 * d(u, 2, 2, 2)(params, t, x))

        t, x = m.sample(key, m.cfg.batch)
        r2 = jax.vmap(r, (None, 0, 0))(params, t, x) ** 2
        ui = jax.vmap(u, (None, None, 0))(params, t0, x_ic)
        return {"ics": jnp.mean((ui - u0(x_ic)) ** 2), "res": m.causal(r2, t)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000, num_chunks=16, causal_tol=0.1, pi_init=True)
    return PINN(problem, ("t", "x"), 1, losses, cfg, ic_data=lambda m: m.ic_tiled([x_ic], u0(x_ic)[:, None])).compile()
