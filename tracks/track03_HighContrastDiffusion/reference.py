"""Reference: cell-centered finite volumes (harmonic face averaging, inclusion edges aligned with
faces), multigrid-preconditioned CG to rtol 1e-11, on grids r x 256 (r = 9, 15, 21; output cell centers
coincide with fine cell centers, injection), Richardson-extrapolated.

The inclusion corners make u singular (~ rho^(2/3)), so FV converges at order p ~ 4/3, not 2. p is measured
from the three grids and the 15/21 pair is extrapolated with it; the 9/15 extrapolation at the same p is the
self-consistency check (an upper estimate of the reference error)."""
import torch

from speedrun import plotting
from speedrun.numerics.fv_diffusion import solve_diffusion

from .problem import N_EVAL, make_problem


def run(p, N):
    u, hist = solve_diffusion(p.fns["a"], p.fns["forcing"], N, dtype=p.dtype, device=p.device, rtol=1e-11)
    r = N // N_EVAL
    return u[r // 2::r, r // 2::r], len(hist)


def _order(u9, u15, u21, a=9, b=15, c=21):
    q = float(torch.linalg.norm(u9 - u15) / torch.linalg.norm(u15 - u21))
    lo, hi = 0.25, 4.0                                   # (a^-p - b^-p) / (b^-p - c^-p) = q, increasing in p
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (a**-mid - b**-mid) / (b**-mid - c**-mid) < q else (lo, mid)
    return 0.5 * (lo + hi)


def _extrap(uc, uf, rc, rf, p):
    return (rf**p * uf - rc**p * uc) / (rf**p - rc**p)


def compute(device="cpu"):
    p = make_problem(device=device)
    (u9, i9), (u15, i15), (u21, i21) = (run(p, r * N_EVAL) for r in (9, 15, 21))
    order = _order(u9, u15, u21)
    u = _extrap(u15, u21, 15, 21, order)
    c = _extrap(u9, u15, 9, 15, order)
    rel = lambda a: float(torch.linalg.norm(a - u) / torch.linalg.norm(u))  # noqa: E731
    meta = {"method": f"cell-centered FV + MG-PCG on {15 * N_EVAL}^2 and {21 * N_EVAL}^2 ({i15}, {i21} PCG its), "
                      f"Richardson-extrapolated at the observed order p = {order:.3f}",
            "observed_order": order,
            "check_extrap9_15_vs_ref_rel_l2": rel(c), "check_extrap9_15_vs_ref_linf": float((c - u).abs().max()),
            "check_N5376_vs_ref_rel_l2": rel(u21), "check_N5376_vs_ref_linf": float((u21 - u).abs().max()),
            "check_N2304_vs_ref_rel_l2": rel(u9), "check_N2304_vs_ref_linf": float((u9 - u).abs().max())}
    return u[None], meta, {}


def plot(problem, u, extras, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    X, Y = problem.grid()
    plotting.fields2d(x, y, {"a(x, y)  [log10]": torch.log(problem.fns["a"](X, Y)) / 2.302585, "u": u[0]},
                      figdir / "reference.png", suptitle="Track 03: high-contrast diffusion", cmap="viridis")
