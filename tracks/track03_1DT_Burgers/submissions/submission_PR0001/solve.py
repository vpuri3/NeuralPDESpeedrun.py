"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Raissi's 1D Burgers with
jaxpi's burgers config (written for this problem): Fourier scale 1, batch 8192, lr decay every 1000 iterations,
32 causal chunks (tol 1), physics-informed initialization from the IC; ics / bcs (u = 0 on x = +-1) / res
losses."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve, to_jax  # noqa: F401

def setup(problem):
    nu, u0 = problem.params["nu"], problem.fns["u0"]
    t0 = problem.domain["t"][0]
    x_ic = to_jax(problem.axes["x"])

    def losses(m, params, key):
        u = m.u(0)
        kr, kb = jax.random.split(key)

        def r(params, t, x):
            return d(u, 1)(params, t, x) + u(params, t, x) * d(u, 2)(params, t, x) - nu * d(u, 2, 2)(params, t, x)

        t, x = m.sample(kr, m.cfg.batch)
        r2 = jax.vmap(r, (None, 0, 0))(params, t, x) ** 2
        ui = jax.vmap(u, (None, None, 0))(params, t0, x_ic)
        tb, xb = m.faces(kb, 1024, ["x"])
        ub = jax.vmap(u, (None, 0, 0))(params, tb, xb)
        return {"ics": jnp.mean((ui - u0(x_ic)) ** 2), "bcs": jnp.mean(ub**2), "res": m.causal(r2, t)}

    cfg = Config(embed_scale=1.0, batch=8192, decay_steps=1000, num_chunks=32, causal_tol=1.0, pi_init=True)
    return PINN(problem, ("t", "x"), 1, losses, cfg, ic_data=lambda m: m.ic_tiled([x_ic], u0(x_ic)[:, None])).compile()
