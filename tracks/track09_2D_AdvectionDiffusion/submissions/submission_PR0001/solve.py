"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Steady advection-diffusion by a
cellular flow (Pe = Re Pr): residual + wall temperatures at x = 0, 1 + insulated walls y = 0, 2."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve  # noqa: F401


def setup(problem):
    pe, f = problem.params["Pe"], problem.fns

    def losses(m, params, key):
        T = m.u(0)
        kr, kd, kn = jax.random.split(key, 3)

        def r(params, x, y):
            u, v = f["velocity"](x, y)
            lap = d(T, 1, 1)(params, x, y) + d(T, 2, 2)(params, x, y)
            return lap / pe - u * d(T, 1)(params, x, y) - v * d(T, 2)(params, x, y)

        x, y = m.sample(kr, m.cfg.batch)
        xd, yd = m.faces(kd, 2048, ["x"])
        xn, yn = m.faces(kn, 2048, ["y"])
        return {"res": jnp.mean(jax.vmap(r, (None, 0, 0))(params, x, y) ** 2),
                "bcs": jnp.mean((jax.vmap(T, (None, 0, 0))(params, xd, yd) - f["bc"](xd, yd)) ** 2),
                "ins": jnp.mean(jax.vmap(d(T, 2), (None, 0, 0))(params, xn, yn) ** 2)}

    cfg = Config(embed_scale=10.0, batch=8192, decay_steps=2000)
    return PINN(problem, ("x", "y"), 1, losses, cfg).compile()
