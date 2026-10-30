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
    Returns f(*coords) evaluating the interpolant at arbitrary points (torch, any shape, same device)."""
    v = values if not endpoint else (values[:-1] if values.ndim == 1 else values[:-1, :-1])
    c = torch.fft.fftn(v.to(torch.complex128)) / v.numel()
    ks = [torch.fft.fftfreq(n, d=1.0 / n, device=v.device, dtype=torch.float64) for n in v.shape]
    lo = lo if isinstance(lo, (tuple, list)) else (lo,)
    length = length if isinstance(length, (tuple, list)) else (length,)

    def f(*coords, chunk=4096):
        shape = coords[0].shape
        flat = [x.reshape(-1).to(torch.float64) for x in coords]
        out = torch.empty(flat[0].numel(), dtype=torch.float64, device=flat[0].device)
        for s in range(0, flat[0].numel(), chunk):
            sl = slice(s, s + chunk)
            if len(flat) == 1:
                ph = torch.exp(2j * math.pi * ks[0][None, :] * ((flat[0][sl, None] - lo[0]) / length[0]))
                out[sl] = (ph * c[None, :]).sum(1).real
            else:
                px = torch.exp(2j * math.pi * ks[0][None, :] * ((flat[0][sl, None] - lo[0]) / length[0]))
                py = torch.exp(2j * math.pi * ks[1][None, :] * ((flat[1][sl, None] - lo[1]) / length[1]))
                out[sl] = torch.einsum("nk,kl,nl->n", px, c, py).real
        return out.reshape(shape).to(coords[0].dtype)

    return f


def plot_1d(problem, u, figdir, title):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title=title)
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title=title)


def plot_2d_time(problem, u, figdir, title, names=None, cmaps=None):
    names = names or problem.fields
    t, x, y = problem.axes["t"], problem.axes["x"], problem.axes["y"]
    k = len(t) - 1
    plotting.fields2d(x, y, {n: u[i, k] for i, n in enumerate(names)}, figdir / "reference.png",
                      suptitle=f"{title}, t = {float(t[k]):.3g}", cmaps=cmaps)
    plotting.gif_fields2d(x, y, {n: u[i] for i, n in enumerate(names)}, t, figdir / "reference.gif",
                          title=title, cmaps=cmaps)
