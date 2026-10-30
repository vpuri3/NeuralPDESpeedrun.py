"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Grey-Scott with jaxpi's
grey_scott config (one time window instead of 10): Fourier scale 1, batch 4096, 32 causal chunks (tol 1),
physics-informed initialization from the sampled initial state; exact periodicity in x and y."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, step, to_jax  # noqa: F401
def ic_grid(problem, names):
    """The sampled initial state on its grid: flattened coordinates (x, y) and values (n, len(names))."""
    ic = problem.fns["ic"]
    X, Y = jnp.meshgrid(to_jax(ic["x"]), to_jax(ic["y"]), indexing="ij")
    return X.ravel(), Y.ravel(), jnp.stack([to_jax(ic[k]).ravel() for k in names], 1)

def setup(problem):
    p = problem.params
    t0 = problem.domain["t"][0]
    XI, YI, UI = ic_grid(problem, ("u", "v"))

    def r(m, params, t, x, y):
        Y, J, H = jet(m, params, t, x, y)
        u, v = Y
        ru = J[0, 0] - p["eps1"] * (H[0, 1, 1] + H[0, 2, 2]) - p["b1"] * (1 - u) + p["c1"] * u * v**2
        rv = J[1, 0] - p["eps2"] * (H[1, 1, 1] + H[1, 2, 2]) + p["b2"] * v - p["c2"] * u * v**2
        return ru**2, rv**2

    def losses(m, params, key):
        kr, ki = jax.random.split(key)
        t, x, y = m.sample(kr, m.cfg.batch)
        ru2, rv2 = jax.vmap(r, (None, None, 0, 0, 0))(m, params, t, x, y)
        i = jax.random.randint(ki, (4096,), 0, XI.shape[0])
        Yi = jax.vmap(m.apply, (None, None, 0, 0))(params, t0, XI[i], YI[i])
        return {"u_ic": jnp.mean((Yi[:, 0] - UI[i, 0]) ** 2), "v_ic": jnp.mean((Yi[:, 1] - UI[i, 1]) ** 2),
                "ru": m.causal(ru2, t), "rv": m.causal(rv2, t)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000, num_chunks=32, causal_tol=1.0, pi_init=True)
    return PINN(problem, ("t", "x", "y"), 2, losses, cfg, ic_data=lambda m: m.ic_tiled([XI, YI], UI)).compile()
