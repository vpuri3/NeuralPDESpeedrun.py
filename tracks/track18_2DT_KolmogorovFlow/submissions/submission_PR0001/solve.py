"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Kolmogorov flow with jaxpi's
kolmogorov_flow config (one time window instead of 25): width 384, Fourier scale 2, batch 8192, lr warmup 2000,
16 causal chunks (tol 1); forced NS residuals + sampled initial state; exact periodicity in x and y; outputs
(u, v, p)."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, step, to_jax  # noqa: F401
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
    nu, force = problem.params["nu"], problem.fns["forcing"]
    t0 = problem.domain["t"][0]
    XI, YI, UI = ic_grid(problem, ("u", "v"))

    def r(m, params, t, x, y):
        ru, rv, rc, _ = ns_unsteady(m, params, t, x, y, nu)
        fx, fy = force(x, y)
        return ru - fx, rv - fy, rc

    def losses(m, params, key):
        kr, ki = jax.random.split(key)
        t, x, y = m.sample(kr, m.cfg.batch)
        ru, rv, rc = jax.vmap(r, (None, None, 0, 0, 0))(m, params, t, x, y)
        i = jax.random.randint(ki, (4096,), 0, XI.shape[0])
        Yi = jax.vmap(m.apply, (None, None, 0, 0))(params, t0, XI[i], YI[i])
        return {"u_ic": jnp.mean((Yi[:, 0] - UI[i, 0]) ** 2), "v_ic": jnp.mean((Yi[:, 1] - UI[i, 1]) ** 2),
                "ru": m.causal(ru**2, t), "rv": m.causal(rv**2, t), "rc": m.causal(rc**2, t)}

    cfg = Config(hidden_dim=384, embed_scale=2.0, batch=8192, decay_steps=2000, warmup_steps=2000, num_chunks=16,
                 causal_tol=1.0)
    return PINN(problem, ("t", "x", "y"), 3, losses, cfg).compile()
