"""Reference: Fourier pseudo-spectral (2/3 dealiasing) + ETDRK4 at 2048 x 2048 (4x the prescribed grid
in each direction), dt = 2.5e-4, restricted to the prescribed 512 x 512 grid by injection.
Snapshots are stored every 0.01 for the animation; every 5th one (t = 0, 0.05, ..., 0.5) is scored.
Self-convergence: compared with the 1024 x 1024, dt = 5e-4 solution."""
import torch

from speedrun.numerics.periodic import burgers_2d

from .problem import NX, X0, make_problem

N_REF = 2048
DT_REF = 2.5e-4


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
        _, coarse = run(p, N_REF // 2, 2 * DT_REF, every=0.05)
        c = coarse.permute(1, 0, 2, 3)
        meta["check_N1024_dt5e-4_vs_ref_linf"] = float((c - u).abs().max())
        meta["check_N1024_dt5e-4_vs_ref_rel_l2"] = float(torch.linalg.norm(c - u) / torch.linalg.norm(u))
    except Exception as e:  # pragma: no cover
        meta["check_error"] = repr(e)
    return u, meta, {"t_frames": t_fr, "frames_u": frames[:, 0, ::2, ::2]}


def plot(problem, u, extras, figdir):
    from tracks.data_common import plot_2d_time
    plot_2d_time(problem, u, figdir, "Track 05B: 2D Burgers")
