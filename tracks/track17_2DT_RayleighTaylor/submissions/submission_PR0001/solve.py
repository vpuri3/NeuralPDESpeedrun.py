"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Rayleigh-Taylor with jaxpi's
rayleigh_taylor config (one time window instead of 4): Fourier scale 1, batch 8192, lr warmup 2000, 16 causal
chunks (tol 1); Boussinesq residuals + sampled initial state + walls; exact periodicity in x; outputs
(u, v, p, T)."""
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
def ic_grid(problem, names):
    """The sampled initial state on its grid: flattened coordinates (x, y) and values (n, len(names))."""
    ic = problem.fns["ic"]
    X, Y = jnp.meshgrid(to_jax(ic["x"]), to_jax(ic["y"]), indexing="ij")
    return X.ravel(), Y.ravel(), jnp.stack([to_jax(ic[k]).ravel() for k in names], 1)

def setup(problem):
    p = problem.params
    t0 = problem.domain["t"][0]
    XI, YI, UI = ic_grid(problem, ("u", "v", "T"))

    def r(m, params, t, x, y):
        ru, rv, rc, (Y, J, H) = ns_unsteady(m, params, t, x, y, p["a1"])
        T = Y[3]
        rT = J[3, 0] + Y[0] * J[3, 1] + Y[1] * J[3, 2] - p["a4"] * (H[3, 1, 1] + H[3, 2, 2])
        return ru, rv - p["a2"] * T, rc, rT

    def losses(m, params, key):
        kr, ki, kb = jax.random.split(key, 3)
        t, x, y = m.sample(kr, m.cfg.batch)
        ru, rv, rc, rT = jax.vmap(r, (None, None, 0, 0, 0))(m, params, t, x, y)
        i = jax.random.randint(ki, (4096,), 0, XI.shape[0])
        Yi = jax.vmap(m.apply, (None, None, 0, 0))(params, t0, XI[i], YI[i])
        tb, xb, yb = m.faces(kb, 2048, ["y"])
        Yb = jax.vmap(m.apply, (None, 0, 0, 0))(params, tb, xb, yb)
        return {"ics": jnp.mean(jnp.sum((Yi[:, (0, 1, 3)] - UI[i]) ** 2, axis=1)),
                "bcs": jnp.mean(jnp.sum(Yb[:, (0, 1, 3)] ** 2, axis=1)),
                "ru": m.causal(ru**2, t), "rv": m.causal(rv**2, t), "rc": m.causal(rc**2, t),
                "rT": m.causal(rT**2, t)}

    cfg = Config(embed_scale=1.0, batch=8192, decay_steps=2000, warmup_steps=2000, num_chunks=16, causal_tol=1.0)
    return PINN(problem, ("t", "x", "y"), 4, losses, cfg, fields=(0, 1, 3)).compile()
