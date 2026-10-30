"""Submission 1: PirateNet + SOAP in JAX (speedrun/submissions/submission_PR0001). Van der Pol: residual with
causal weights + y(0), y'(0). t enters raw, as in jaxpi, so the Fourier scale is set for t in [0, 3000]."""
import jax

from speedrun.submissions.submission_PR0001.pinn import PINN, Config, d, jet, query, solve, to_jax  # noqa: F401

def setup(problem):
    mu, y0, dy0 = problem.params["mu"], problem.fns["y0"], problem.fns["dy0"]
    t0, t1 = problem.domain["t"]

    def losses(m, params, key):
        y = m.u(0)

        def r(params, t):
            v, vt = y(params, t), d(y, 1)(params, t)
            return d(y, 1, 1)(params, t) - mu * (1 - v**2) * vt + v

        (t,) = m.sample(key, m.cfg.batch)
        r2 = jax.vmap(r, (None, 0))(params, t) ** 2
        ics = (y(params, t0) - y0) ** 2 + (d(y, 1)(params, t0) - dy0) ** 2
        return {"ics": ics, "res": m.causal(r2, t)}

    cfg = Config(embed_scale=2.0 / (t1 - t0), batch=4096, num_chunks=32)
    return PINN(problem, ("t",), 1, losses, cfg).compile()
