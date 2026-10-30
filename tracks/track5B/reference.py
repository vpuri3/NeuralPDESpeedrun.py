"""Reference: Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 at 1024 x 1024 (2x the prescribed grid
in each direction), dt = 5e-4, restricted to the prescribed 512 x 512 grid by injection.
Snapshots are stored every 0.01 for the animation; every 5th one (t = 0, 0.05, ..., 0.5) is scored.
Self-convergence: compared with the 512 x 512, dt = 1e-3 solution (the baseline)."""
import torch

from speedrun import plotting
from speedrun.numerics.periodic import burgers_2d

from .problem import NX, X0, make_problem

N_REF = 1024
DT_REF = 5e-4


def run(p, N, dt, every=0.01):
    g = X0 + torch.arange(N, dtype=p.dtype, device=p.device) / N
    X, Y = torch.meshgrid(g, g, indexing="ij")
    T = float(p.axes["t"][-1])
    nfr = int(round(T / every)) + 1
    t_out = torch.linspace(0.0, T, nfr, dtype=p.dtype, device=p.device)
    sub = int(round(every / dt))
    cb = lambda i, t: print(f"    t = {t:.3f}", flush=True)  # noqa: E731
    return t_out, burgers_2d(p.fns["u0"](X, Y), p.fns["v0"](X, Y), 1.0, N, p.params["nu"], t_out, sub,
                             dtype=p.dtype, device=p.device, callback=cb, stride=N // NX)


def compute(device="cpu"):
    p = make_problem(device=device)
    t_fr, frames = run(p, N_REF, DT_REF)                  # (51, 2, 512, 512)
    stride = (t_fr.numel() - 1) // (p.axes["t"].numel() - 1)
    u = frames[::stride].permute(1, 0, 2, 3).contiguous()  # (2, 11, 512, 512)
    meta = {"method": f"Fourier pseudo-spectral + ETDRK4, {N_REF}^2, dt={DT_REF}"}
    try:
        _, coarse = run(p, NX, 1e-3, every=0.05)
        c = coarse.permute(1, 0, 2, 3)
        meta["check_N512_dt1e-3_vs_ref_linf"] = float((c - u).abs().max())
        meta["check_N512_dt1e-3_vs_ref_rel_l2"] = float(torch.linalg.norm(c - u) / torch.linalg.norm(u))
    except Exception as e:  # pragma: no cover
        meta["check_error"] = repr(e)
    return u, meta, {"t_frames": t_fr, "frames_u": frames[:, 0, ::2, ::2]}


def plot(problem, u, extras, figdir):
    x, y, t = problem.axes["x"], problem.axes["y"], problem.axes["t"]
    for k in (0, 5, 10):
        plotting.fields2d(x, y, {"u": u[0, k], "v": u[1, k]}, figdir / f"reference_t{float(t[k]):.2f}.png",
                          suptitle=f"Track 5B: 2D Burgers, t = {float(t[k]):.2f}", cmap="viridis")
    if "frames_u" in extras:
        plotting.gif2d(x[::2], y[::2], extras["frames_u"], extras["t_frames"], figdir / "reference.gif",
                       title="Track 5B: u", cmap="RdBu_r")
    else:
        plotting.gif2d(x, y, u[0], t, figdir / "reference.gif", title="Track 5B: u")
