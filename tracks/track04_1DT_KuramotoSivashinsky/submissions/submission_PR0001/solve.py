"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Kuramoto-Sivashinsky with
jaxpi's ks_chaotic config (one time window instead of 10): Fourier scale 1, batch 4096, initial weights ics 1000,
16 causal chunks (tol 0.1), physics-informed initialization from the sampled initial state."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, step, to_jax  # noqa: F401

def setup(problem):
    a, b, c = problem.params["a"], problem.params["b"], problem.params["c"]
    t0 = problem.domain["t"][0]
    ic = problem.fns["ic"]
    x_ic, u_ic = to_jax(ic["x"]), to_jax(ic["u"])

    def losses(m, params, key):
        u = m.u(0)

        def r(params, t, x):
            return (d(u, 1)(params, t, x) + a * u(params, t, x) * d(u, 2)(params, t, x)
                    + b * d(u, 2, 2)(params, t, x) + c * d(u, 2, 2, 2, 2)(params, t, x))

        t, x = m.sample(key, m.cfg.batch)
        r2 = jax.vmap(r, (None, 0, 0))(params, t, x) ** 2
        ui = jax.vmap(u, (None, None, 0))(params, t0, x_ic)
        return {"ics": jnp.mean((ui - u_ic) ** 2), "res": m.causal(r2, t)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000, num_chunks=16, causal_tol=0.1, pi_init=True,
                 init_weights={"ics": 1000.0})
    return PINN(problem, ("t", "x"), 1, losses, cfg, ic_data=lambda m: m.ic_tiled([x_ic], u_ic[:, None])).compile()
