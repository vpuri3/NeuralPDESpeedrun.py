"""Plot helpers for reference solutions."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def line(x, ys: dict, path: Path, title="", xlabel="x", ylabel="u"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, y in ys.items():
        ax.plot(_np(x), _np(y), lw=2, label=k)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fields2d(x, y, fields: dict, path: Path, suptitle="", cmap="RdBu_r", streamplot=False):
    """Heatmaps of several 2D fields F[i, j] = f(x_i, y_j)."""
    n = len(fields)
    fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 4.0), squeeze=False)
    xs, ys = _np(x), _np(y)
    for ax, (k, F) in zip(axs[0], fields.items()):
        F = _np(F)
        vmax = np.abs(F).max()
        im = ax.pcolormesh(xs, ys, F.T, shading="auto", cmap=cmap,
                           vmin=-vmax if cmap == "RdBu_r" else None, vmax=vmax if cmap == "RdBu_r" else None)
        ax.set(title=k, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(im, ax=ax, shrink=0.85)
    if streamplot and "u" in fields and "v" in fields:
        ax = axs[0][-1]
        ax.streamplot(xs, ys, _np(fields["u"]).T, _np(fields["v"]).T, color="k", density=1.2, linewidth=0.5)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spacetime(x, t, U, path: Path, title="", xlabel="x"):
    """U[it, ix] as a space-time heatmap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(_np(x), _np(t), _np(U), shading="auto", cmap="viridis")
    ax.set(title=title, xlabel=xlabel, ylabel="t")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def snapshots1d(x, t, U, path: Path, n=6, title=""):
    U, t = _np(U), _np(t)
    idx = np.linspace(0, len(t) - 1, n).round().astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = plt.get_cmap("viridis")
    for c, i in enumerate(idx):
        ax.plot(_np(x), U[i], color=cm(c / max(n - 1, 1)), lw=2, label=f"t={t[i]:.3g}")
    ax.set(title=title, xlabel="x", ylabel="u")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gif1d(x, t, U, path: Path, title="", frames=80, fps=20):
    U, t, x = _np(U), _np(t), _np(x)
    idx = np.linspace(0, len(t) - 1, min(frames, len(t))).round().astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    (ln,) = ax.plot(x, U[0], lw=2)
    lo, hi = U.min(), U.max()
    pad = 0.05 * (hi - lo + 1e-12)
    ax.set(xlim=(x.min(), x.max()), ylim=(lo - pad, hi + pad), xlabel="x", ylabel="u")
    ax.grid(alpha=0.3)
    ttl = ax.set_title("")

    def upd(k):
        ln.set_ydata(U[idx[k]])
        ttl.set_text(f"{title}   t = {t[idx[k]]:.4f}")
        return ln, ttl

    anim = FuncAnimation(fig, upd, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def gif2d(x, y, frames, times, path: Path, title="", fps=8, cmap="RdBu_r"):
    """frames[k, i, j] = f(x_i, y_j, t_k)."""
    F, times, xs, ys = _np(frames), _np(times), _np(x), _np(y)
    vmax = np.abs(F).max()
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.pcolormesh(xs, ys, F[0].T, shading="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set(xlabel="x", ylabel="y", aspect="equal")
    ttl = ax.set_title("")

    def upd(k):
        im.set_array(F[k].T.ravel())
        ttl.set_text(f"{title}   t = {times[k]:.3f}")
        return im, ttl

    anim = FuncAnimation(fig, upd, frames=len(times), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
