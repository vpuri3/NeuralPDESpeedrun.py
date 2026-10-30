"""Plot helpers for reference solutions: filled-contour PNGs (200 dpi) and GIF animations."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.titlesize": 13,
    "axes.linewidth": 0.8,
})
GIF_DPI = 110
DIVERGING = {"RdBu_r", "seismic", "coolwarm", "bwr"}


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _norm(F, cmap):
    if cmap in DIVERGING:
        m = float(np.abs(F).max()) or 1.0
        return -m, m
    return float(F.min()), float(F.max())


LEVELS = 64


def _contour(ax, x, y, F, cmap, lo, hi):
    """Filled contours of F[iy, ix] on fixed levels (identical color scale across panels / frames)."""
    if hi <= lo:
        hi = lo + 1e-12
    levels = np.linspace(lo, hi, LEVELS + 1)
    cs = ax.contourf(x, y, np.clip(F, lo, hi), levels=levels, cmap=cmap, extend="neither")
    cs.set_rasterized(True)
    return cs


def _cbar(fig, ax, im):
    """Colorbar exactly as tall as its (aspect-locked) axes."""
    cb = fig.colorbar(im, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0]))
    cb.locator = matplotlib.ticker.MaxNLocator(6)
    cb.update_ticks()


def _cmap_for(name, F, cmaps):
    if cmaps and name in cmaps:
        return cmaps[name]
    return "RdBu_r" if F.min() < 0 < F.max() and abs(F.min()) > 0.05 * abs(F.max()) else "viridis"


def line(x, ys: dict, path: Path, title="", xlabel="x", ylabel="u"):
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for k, y in ys.items():
        ax.plot(_np(x), _np(y), lw=1.6, label=k)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if len(ys) > 1:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)


def fields2d(x, y, fields: dict, path: Path, suptitle="", cmap=None, streamplot=False, cmaps=None):
    """Heatmaps of several 2D fields F[i, j] = f(x_i, y_j)."""
    n = len(fields)
    xs, ys = _np(x), _np(y)
    aspect = (ys.max() - ys.min()) / (xs.max() - xs.min())
    w = 4.2 if aspect <= 1.2 else 2.6
    fig, axs = plt.subplots(1, n, figsize=((w + 1.0) * n, w * min(aspect, 2.2) + 0.7), squeeze=False,
                            layout="constrained")
    for ax, (k, F) in zip(axs[0], fields.items()):
        F = _np(F)
        cm = cmap or _cmap_for(k, F, cmaps)
        lo, hi = _norm(F, cm)
        im = _contour(ax, xs, ys, F.T, cm, lo, hi)
        ax.set(title=k, xlabel="x", ylabel="y", aspect="equal")
        _cbar(fig, ax, im)
    if streamplot and "u" in fields and "v" in fields:
        ax = axs[0][-1]
        ax.streamplot(xs, ys, _np(fields["u"]).T, _np(fields["v"]).T, color="k", density=1.2, linewidth=0.5)
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(path)
    plt.close(fig)


def spacetime(x, t, U, path: Path, title="", xlabel="x", cmap=None):
    """U[it, ix] as a space-time heatmap (x horizontal, t vertical)."""
    U = _np(U)
    cm = cmap or _cmap_for("u", U, None)
    lo, hi = _norm(U, cm)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    im = _contour(ax, _np(x), _np(t), U, cm, lo, hi)
    ax.set(title=title, xlabel=xlabel, ylabel="t")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.locator = matplotlib.ticker.MaxNLocator(6)
    cb.update_ticks()
    fig.savefig(path)
    plt.close(fig)


def snapshots1d(x, t, U, path: Path, n=6, title=""):
    U, t = _np(U), _np(t)
    idx = np.linspace(0, len(t) - 1, n).round().astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    cm = plt.get_cmap("viridis")
    for c, i in enumerate(idx):
        ax.plot(_np(x), U[i], color=cm(c / max(n - 1, 1)), lw=1.6, label=f"t={t[i]:.3g}")
    ax.set(title=title, xlabel="x", ylabel="u")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def gif1d(x, t, U, path: Path, title="", frames=60, fps=15):
    U, t, x = _np(U), _np(t), _np(x)
    idx = np.linspace(0, len(t) - 1, min(frames, len(t))).round().astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    (ln,) = ax.plot(x, U[0], lw=1.8, color="tab:blue")
    lo, hi = U.min(), U.max()
    pad = 0.05 * (hi - lo + 1e-12)
    ax.set(xlim=(x.min(), x.max()), ylim=(lo - pad, hi + pad), xlabel="x", ylabel="u")
    ax.grid(alpha=0.3)
    ttl = ax.set_title("")

    def upd(k):
        ln.set_ydata(U[idx[k]])
        ttl.set_text(f"{title}   t = {t[idx[k]]:.3f}")
        return ln, ttl

    FuncAnimation(fig, upd, frames=len(idx), blit=False).save(path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)


def gif_fields2d(x, y, fields: dict, times, path: Path, title="", fps=10, frames=50, cmaps=None):
    """Side-by-side animation of fields[name][k, i, j] = f(x_i, y_j, t_k), fixed color scales."""
    xs, ys, times = _np(x), _np(y), _np(times)
    F = {k: _np(v) for k, v in fields.items()}
    idx = np.linspace(0, len(times) - 1, min(frames, len(times))).round().astype(int)
    n = len(F)
    aspect = (ys.max() - ys.min()) / (xs.max() - xs.min())
    w = 4.0 if aspect <= 1.2 else 2.6
    fig, axs = plt.subplots(1, n, figsize=((w + 1.0) * n, w * min(aspect, 2.2) + 0.8), squeeze=False,
                            layout="constrained")
    ims, styles = [], []
    for ax, (k, A) in zip(axs[0], F.items()):
        cm = _cmap_for(k, A, cmaps)
        lo, hi = _norm(A, cm)
        im = _contour(ax, xs, ys, A[0].T, cm, lo, hi)
        ax.set(title=k, xlabel="x", ylabel="y", aspect="equal")
        _cbar(fig, ax, im)
        ims.append(im)
        styles.append((cm, lo, hi))
    sup = fig.suptitle("")

    def upd(j):
        for n, (ax, A, (cm, lo, hi)) in enumerate(zip(axs[0], F.values(), styles)):
            ims[n].remove()
            ims[n] = _contour(ax, xs, ys, A[idx[j]].T, cm, lo, hi)
        sup.set_text(f"{title}   t = {times[idx[j]]:.3f}")
        return ims

    FuncAnimation(fig, upd, frames=len(idx), blit=False).save(path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)


def gif2d(x, y, frames, times, path: Path, title="", fps=8, cmap="RdBu_r"):
    """Single-field animation; frames[k, i, j] = f(x_i, y_j, t_k)."""
    gif_fields2d(x, y, {"": frames}, times, path, title=title, fps=fps, cmaps={"": cmap})
