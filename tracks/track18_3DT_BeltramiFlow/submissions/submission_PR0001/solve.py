"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Beltrami flow: 3D unsteady NS
residuals (momentum, continuity) with causal weights + initial and Dirichlet velocity; outputs (u, v, w, p)."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve, to_jax  # noqa: F401

def setup(problem):
    nu, f = problem.params["nu"], problem.fns
    t0 = problem.domain["t"][0]

    def r(m, params, t, x, y, z):
        Y, J, H = jet(m, params, t, x, y, z)
        res = 0.0
        for k in range(3):
            adv = sum(Y[j] * J[k, 1 + j] for j in range(3))
            lap = sum(H[k, 1 + j, 1 + j] for j in range(3))
            res = res + (J[k, 0] + adv + J[3, 1 + k] - nu * lap) ** 2
        return res, (J[0, 1] + J[1, 2] + J[2, 3]) ** 2

    def losses(m, params, key):
        kr, ki, kb = jax.random.split(key, 3)
        t, x, y, z = m.sample(kr, m.cfg.batch)
        rm, rc = jax.vmap(r, (None, None, 0, 0, 0, 0))(m, params, t, x, y, z)
        _, xi, yi, zi = m.sample(ki, 2048, t=t0)
        Yi = jax.vmap(m.apply, (None, None, 0, 0, 0))(params, t0, xi, yi, zi)
        ics = sum(jnp.mean((Yi[:, k] - f[n](xi, yi, zi)) ** 2) for k, n in enumerate(("u0", "v0", "w0")))
        tb, xb, yb, zb = m.faces(kb, 4096, ["x", "y", "z"])
        Yb = jax.vmap(m.apply, (None, 0, 0, 0, 0))(params, tb, xb, yb, zb)
        bcs = sum(jnp.mean((Yb[:, k] - f[n](tb, xb, yb, zb)) ** 2) for k, n in enumerate(("u_bc", "v_bc", "w_bc")))
        return {"ics": ics, "bcs": bcs, "res": m.causal(rm, t), "div": m.causal(rc, t)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000, num_chunks=16, causal_tol=1.0)
    return PINN(problem, ("t", "x", "y", "z"), 4, losses, cfg, fields=(0, 1, 2)).compile()
