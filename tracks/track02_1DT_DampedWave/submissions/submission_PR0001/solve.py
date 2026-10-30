"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Damped wave with jaxpi's wave
config: Fourier scale 10, batch 8192, 32 causal chunks (tol 1e-2), separate u0 / u_t0 / bcs / res losses,
physics-informed initialization from the IC."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, step, to_jax  # noqa: F401

def setup(problem):
    c, gamma, f = problem.params["c"], problem.params["gamma"], problem.fns
    t0 = problem.domain["t"][0]
    x_ic = to_jax(problem.axes["x"])

    def losses(m, params, key):
        u = m.u(0)
        kr, kb = jax.random.split(key)

        def r(params, t, x):
            return d(u, 1, 1)(params, t, x) - c**2 * d(u, 2, 2)(params, t, x) + gamma * d(u, 1)(params, t, x)

        t, x = m.sample(kr, m.cfg.batch)
        r2 = jax.vmap(r, (None, 0, 0))(params, t, x) ** 2
        ui = jax.vmap(u, (None, None, 0))(params, t0, x_ic)
        uti = jax.vmap(d(u, 1), (None, None, 0))(params, t0, x_ic)
        tb, xb = m.faces(kb, 1024, ["x"])
        ub = jax.vmap(u, (None, 0, 0))(params, tb, xb)
        return {"u0": jnp.mean((ui - f["u0"](x_ic)) ** 2), "u_t0": jnp.mean((uti - f["ut0"](x_ic)) ** 2),
                "bcs": jnp.mean(ub**2), "res": m.causal(r2, t)}

    cfg = Config(embed_scale=10.0, batch=8192, decay_steps=2000, num_chunks=32, causal_tol=1e-2, pi_init=True)
    return PINN(problem, ("t", "x"), 1, losses, cfg,
                ic_data=lambda m: m.ic_tiled([x_ic], f["u0"](x_ic)[:, None])).compile()
