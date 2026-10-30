"""Helpers for tracks whose reference is published data (speedrun/external.py).

Each such track stores the small problem data a solver may use (evaluation axes, initial state on its grid,
parameters) in tracks/<track>/data/problem.npz, written by its reference.py. The reference itself goes to
tracks/<track>/reference/ref.npz like every other track.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from speedrun import plotting
from speedrun.problem import array_device, xp_of
from speedrun.tracks import TRACKS_DIR


def problem_npz(track: str) -> Path:
    return TRACKS_DIR / track / "data" / "problem.npz"


def save_problem_data(track: str, **arrays) -> None:
    p = problem_npz(track)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})


def load_problem_data(track: str, device="cpu", dtype=torch.float64) -> dict:
    p = problem_npz(track)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `uv run python -m speedrun.make_reference --track {track}`")
    with np.load(p) as d:
        return {k: torch.as_tensor(d[k], dtype=dtype, device=device) for k in d.files}


def periodic_interp(values: torch.Tensor, lo, length, endpoint=True):
    """Trigonometric interpolant of samples on a uniform periodic grid.

    values: (n,) or (nx, ny) samples starting at `lo`; with endpoint=True the last point duplicates the first
    and is dropped. lo, length: scalars or per-axis tuples.
    Returns f(*coords) evaluating the interpolant at arbitrary points: torch or JAX arrays of any shape, same
    namespace, device and dtype out. Real arithmetic only: Re(sum_k c_k e^{i k x}) with c = a + i b."""
    v = values if not endpoint else (values[:-1] if values.ndim == 1 else values[:-1, :-1])
    v = v.detach().cpu().numpy().astype(np.float64)
    c = np.fft.fftn(v) / v.size
    ks = [np.fft.fftfreq(n, d=1.0 / n) for n in v.shape]
    lo = lo if isinstance(lo, (tuple, list)) else (lo,)
    length = length if isinstance(length, (tuple, list)) else (length,)
    cache = {}

    def consts(xp, dev, dtype):
        k = (xp.__name__, str(dev), str(dtype))
        if k not in cache:
            put = lambda a: xp.asarray(np.ascontiguousarray(a), dtype=dtype, device=dev)  # noqa: E731
            cache[k] = [put(kk) for kk in ks], put(c.real), put(c.imag)
        return cache[k]

    def f(*coords, chunk=4096):
        xp = xp_of(*coords)
        dev, dtype = array_device(coords[0]), coords[0].dtype
        (kx, *ky), a, b = consts(xp, dev, dtype)
        shape = coords[0].shape
        flat = [xp.reshape(x, (-1,)) for x in coords]
        out = []
        for s in range(0, flat[0].shape[0], chunk):
            th = 2 * math.pi * kx[None, :] * ((flat[0][s:s + chunk, None] - lo[0]) / length[0])
            cx, sx = xp.cos(th), xp.sin(th)
            if len(flat) == 1:
                out.append(cx @ a - sx @ b)
            else:
                ph = 2 * math.pi * ky[0][None, :] * ((flat[1][s:s + chunk, None] - lo[1]) / length[1])
                re, im = cx @ a - sx @ b, cx @ b + sx @ a           # (e^{i kx x} c)_l
                out.append(xp.sum(re * xp.cos(ph) - im * xp.sin(ph), axis=1))
        return xp.reshape(xp.concat(out), shape)

    return f


def plot_1d(problem, u, figdir, title):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title=title)
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title=title)


def periodic_vorticity(u, v, Lx, Ly):
    """w = v_x - u_y by FFT on a uniform periodic grid; u, v: (..., nx, ny) torch tensors."""
    nx, ny = u.shape[-2], u.shape[-1]
    kx = 2 * math.pi * torch.fft.fftfreq(nx, d=Lx / nx, dtype=torch.float64)[:, None]
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=Ly / ny, dtype=torch.float64)[None, :]
    vx = torch.fft.ifft2(1j * kx * torch.fft.fft2(v.to(torch.float64))).real
    uy = torch.fft.ifft2(1j * ky * torch.fft.fft2(u.to(torch.float64))).real
    return vx - uy


def plot_2d_time(problem, u, figdir, title, names=None, cmaps=None, vorticity=False):
    """Snapshot rows (one row per field) + side-by-side animation. With vorticity=True (periodic flows), plot
    w = v_x - u_y instead of the velocity components."""
    t, x, y = problem.axes["t"], problem.axes["x"], problem.axes["y"]
    if vorticity:
        (x0, x1), (y0, y1) = problem.domain["x"], problem.domain["y"]
        F = {"vorticity": periodic_vorticity(u[0], u[1], x1 - x0, y1 - y0)}
    else:
        F = {n: u[i] for i, n in enumerate(names or problem.fields)}
    plotting.snapshot_rows(x, y, F, t, figdir / "reference.png", title=title, cmaps=cmaps)
    plotting.gif_fields2d(x, y, F, t, figdir / "reference.gif", title=title, cmaps=cmaps)
