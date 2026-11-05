"""Reference: the exact solution u = (1 - exp(-x/eps)) / (1 - exp(-1/eps)) - x, cross-checked against Chebyshev
collocation (the baseline's solver) at N = 128."""
import math

from .baseline import chebyshev_collocation
from .problem import exact, make_problem


def compute(device="cpu"):
    p = make_problem(device=device)
    u = exact(p.axes["x"])[None]
    chk = float((chebyshev_collocation(p, 128) - u).abs().max())
    return u, {"method": "exact solution", "check_chebyshev_N128_linf": chk}, {}


def plot(problem, u, extras, figdir):
    import matplotlib.pyplot as plt
    import numpy as np
    eps = problem.params["eps"]
    x, u = np.asarray(problem.axes["x"].cpu()), np.asarray(u[0].cpu())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for a, hi in zip(ax, (1.0, 10 * eps)):
        m = x <= hi + 1e-12
        a.plot(x[m], u[m], lw=1.8, color="#e8604c")
        if hi < 1:
            a.plot(x[m], u[m], "o", ms=2.5, color="k", label="output points")
            a.axvline(eps * math.log(2), ls=":", color="k", lw=1)
            a.legend()
        a.set(xlabel="x", ylabel="u", title="full domain" if hi == 1 else f"boundary layer, x <= {hi:g}")
        a.grid(alpha=0.3)
    fig.suptitle(f"Track 07: -eps u'' - u' = 1, u(0) = u(1) = 0, eps = {eps:g}")
    fig.savefig(figdir / "reference.png", dpi=150)
    plt.close(fig)
