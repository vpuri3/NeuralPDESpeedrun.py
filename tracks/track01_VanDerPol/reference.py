"""Reference: implicit Runge-Kutta Radau IIA (order 5, adaptive; scipy) with rtol 1e-12, atol 1e-14 and the
analytic Jacobian. Self-convergence is checked against rtol 1e-10."""
import numpy as np
import torch

from speedrun import plotting

from .problem import make_problem


def radau(p, rtol):
    from scipy.integrate import solve_ivp
    mu = p.params["mu"]
    t = p.axes["t"].cpu().numpy()
    f = lambda s, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]  # noqa: E731
    jac = lambda s, y: [[0.0, 1.0], [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]]  # noqa: E731
    sol = solve_ivp(f, (t[0], t[-1]), [p.fns["y0"], p.fns["dy0"]], method="Radau", t_eval=t, rtol=rtol,
                    atol=rtol * 1e-2, jac=jac)
    assert sol.status == 0, sol.message
    return torch.from_numpy(np.ascontiguousarray(sol.y[0]))


def compute(device="cpu"):
    p = make_problem(device=device)
    u, c = radau(p, 1e-12), radau(p, 1e-10)
    meta = {"method": "Radau IIA (scipy), rtol 1e-12, atol 1e-14, analytic Jacobian",
            "check_rtol1e-10_linf": float((c - u).abs().max()),
            "check_rtol1e-10_rel_l2": float(torch.linalg.norm(c - u) / torch.linalg.norm(u))}
    return u[None], meta, {}


def plot(problem, u, extras, figdir):
    plotting.line(problem.axes["t"], {"y(t)": u[0]}, figdir / "reference.png",
                  title="Track 01: Van der Pol, mu = 1000", xlabel="t", ylabel="y")
