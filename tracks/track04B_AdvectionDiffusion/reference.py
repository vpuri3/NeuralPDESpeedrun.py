"""Reference: closed-form solution u = g(x) g(y) (manufactured by construction of f),
cross-checked against the Chebyshev collocation baseline at N = 96."""
from speedrun import plotting

from .problem import g, make_problem


def compute(device="cpu"):
    p = make_problem(device=device)
    X, Y = p.grid()
    u = (g(X) * g(Y))[None]
    from .baseline import chebyshev_collocation
    chk = float((chebyshev_collocation(p, 96) - u).abs().max())
    return u, {"method": "closed form u = g(x) g(y)", "check_chebyshev_N96_linf": chk}, {}


def plot(problem, u, extras, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    plotting.fields2d(x, y, {"u": u[0]}, figdir / "reference.png", suptitle="Track 04B: advection-diffusion, eps = 1e-3",
                      cmap="viridis")
    n = u.shape[-1]
    plotting.line(x, {"u(x, y=0.5)": u[0, :, n // 2], "u(x, y=x)": u[0].diagonal()}, figdir / "reference_slices.png",
                  title="Track 04B: slices (boundary layer at x = 1)")
