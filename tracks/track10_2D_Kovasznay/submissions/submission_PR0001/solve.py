"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Kovasznay flow: steady NS
residuals (momentum, continuity) + Dirichlet velocity on the whole boundary; outputs (u, v, p)."""
import jax
import jax.numpy as jnp

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, step, to_jax  # noqa: F401
def ns_steady(m, params, x, y, nu):
    """Steady incompressible NS residuals (ru, rv, rc) of the outputs (u, v, p) at one point."""
    Y, J, H = jet(m, params, x, y)
    (u, v), (gu, gv, gp) = Y[:2], J[:3]
    ru = u * gu[0] + v * gu[1] + gp[0] - nu * (H[0, 0, 0] + H[0, 1, 1])
    rv = u * gv[0] + v * gv[1] + gp[1] - nu * (H[1, 0, 0] + H[1, 1, 1])
    return ru, rv, gu[0] + gv[1]

def setup(problem):
    nu, f = problem.params["nu"], problem.fns

    def losses(m, params, key):
        kr, kb = jax.random.split(key)
        x, y = m.sample(kr, m.cfg.batch)
        ru, rv, rc = jax.vmap(ns_steady, (None, None, 0, 0, None))(m, params, x, y, nu)
        xb, yb = m.faces(kb, 2048, ["x", "y"])
        Yb = jax.vmap(m.apply, (None, 0, 0))(params, xb, yb)
        return {"ru": jnp.mean(ru**2), "rv": jnp.mean(rv**2), "rc": jnp.mean(rc**2),
                "u_bc": jnp.mean((Yb[:, 0] - f["u_bc"](xb, yb)) ** 2),
                "v_bc": jnp.mean((Yb[:, 1] - f["v_bc"](xb, yb)) ** 2)}

    cfg = Config(embed_scale=1.0, batch=4096, decay_steps=2000)
    return PINN(problem, ("x", "y"), 3, losses, cfg).compile()
