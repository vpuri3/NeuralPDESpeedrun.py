"""Reference: cell-centered finite volumes (harmonic face averaging, inclusion edges aligned with
faces) on a 2304 x 2304 grid (9x the output grid), multigrid-preconditioned CG to rtol 1e-11.
Output cell centers coincide with fine cell centers (injection). Self-convergence is checked
against 768 x 768; the FV scheme is second order, so the 2304 grid error is ~1/9 of that difference."""
import torch

from speedrun import plotting
from speedrun.numerics.fv_diffusion import solve_diffusion

from .problem import N_EVAL, make_problem


def run(p, N):
    u, hist = solve_diffusion(p.fns["a"], p.fns["forcing"], N, dtype=p.dtype, device=p.device, rtol=1e-11)
    r = N // N_EVAL
    return u[r // 2::r, r // 2::r], len(hist)


def compute(device="cpu"):
    p = make_problem(device=device)
    u, it_ref = run(p, 9 * N_EVAL)
    c, it_chk = run(p, 3 * N_EVAL)
    meta = {"method": f"cell-centered FV + MG-PCG, N={9 * N_EVAL} ({it_ref} PCG its)",
            "check_N768_vs_N2304_linf": float((c - u).abs().max()),
            "check_N768_vs_N2304_rel_l2": float(torch.linalg.norm(c - u) / torch.linalg.norm(u))}
    return u[None], meta, {}


def plot(problem, u, extras, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    X, Y = problem.grid()
    plotting.fields2d(x, y, {"a(x, y)  [log10]": torch.log(problem.fns["a"](X, Y)) / 2.302585, "u": u[0]},
                      figdir / "reference.png", suptitle="Track 3: high-contrast diffusion", cmap="viridis")
