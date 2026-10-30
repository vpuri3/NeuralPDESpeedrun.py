"""Plot helpers for reference solutions: high-contrast filled-contour figures (200 dpi PNG) and GIF animations.

Styles follow Fig. 3 of Wang et al. (SOAP for PINNs):
  - space-time plots of 1D evolutions: t horizontal, x vertical, `jet`, colorbar below          (Fig. 3d)
  - 2D evolutions: one row of snapshots per field, no ticks, shared colorbar; diverging fields in a dark-centered
    blue/orange map (`ICEFIRE`), non-negative fields in `twilight_shifted`                       (Fig. 3a, 3e)
  - steady flows: speed in `jet` with white streamlines, plus centerline profiles                (Fig. 3c)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.titlesize": 13,
    "axes.linewidth": 0.8, "mathtext.fontset": "cm",
})
GIF_DPI = 110
LEVELS = 64

# dark-centered diverging map (light blue -> blue -> black -> red -> light orange), after seaborn's "icefire"
ICEFIRE = LinearSegmentedColormap.from_list("icefire", [
    (0.00, "#bde7f0"), (0.15, "#4fa8dc"), (0.32, "#3b56a7"), (0.45, "#241d3f"), (0.50, "#1b1a1f"),
    (0.55, "#3f1b2b"), (0.68, "#a8323a"), (0.85, "#ef7d38"), (1.00, "#fbe3b6")])
DIVERGING = {"RdBu_r", "seismic", "coolwarm", "bwr", "icefire"}


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _signed(F):
    F = np.asarray(F)
    return F.min() < 0 < F.max() and min(-F.min(), F.max()) > 0.05 * np.abs(F).max()


def _norm(F, cmap):
    if getattr(cmap, "name", cmap) in DIVERGING:
        m = float(np.abs(F).max()) or 1.0
        return -m, m
    return float(F.min()), float(F.max())


def _contour(ax, x, y, F, cmap, lo, hi):
    """Filled contours of F[iy, ix] on fixed levels (identical color scale across panels / frames)."""
    if hi <= lo:
        hi = lo + 1e-12
    cs = ax.contourf(x, y, np.clip(F, lo, hi), levels=np.linspace(lo, hi, LEVELS + 1), cmap=cmap)
    cs.set_rasterized(True)
    return cs


def _ticks(cb):
    cb.locator = matplotlib.ticker.MaxNLocator(5)
    cb.update_ticks()


def _cbar(fig, ax, im):
    """Colorbar exactly as tall as its (aspect-locked) axes."""
    _ticks(fig.colorbar(im, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0])))


def evolution_cmap(F):
    return ICEFIRE if _signed(F) else "twilight_shifted"


def _cmap_for(name, F, cmaps, default="jet"):
    if cmaps and name in cmaps:
        return cmaps[name]
    return default


# ----------------------------------------------------------------------------- 1D
def line(x, ys: dict, path: Path, title="", xlabel="x", ylabel="u"):
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    for k, y in ys.items():
        ax.plot(_np(x), _np(y), lw=1.6, label=k, color="#e8604c" if len(ys) == 1 else None)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    if len(ys) > 1:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)


def spacetime(x, t, U, path: Path, title="", xlabel="x", cmap="jet"):
    """U[it, ix] as filled contours with t horizontal and x vertical, colorbar below (Fig. 3d style)."""
    U, x, t = _np(U), _np(x), _np(t)
    lo, hi = _norm(U, cmap)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    im = _contour(ax, t, x, U.T, cmap, lo, hi)
    ax.set(title=title, xlabel="t", ylabel=xlabel)
    _ticks(fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.02, aspect=50))
    fig.savefig(path)
    plt.close(fig)


def snapshots1d(x, t, U, path: Path, n=6, title=""):
    U, t = _np(U), _np(t)
    idx = np.linspace(0, len(t) - 1, n).round().astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
    cm = plt.get_cmap("jet")
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
    fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
    (ln,) = ax.plot(x, U[0], lw=1.8, color="#e8604c")
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


# ----------------------------------------------------------------------------- 2D
def fields2d(x, y, fields: dict, path: Path, suptitle="", cmap=None, streamplot=False, cmaps=None, ticks=True):
    """Filled contours of several 2D fields F[i, j] = f(x_i, y_j), side by side (jet by default)."""
    n = len(fields)
    xs, ys = _np(x), _np(y)
    aspect = (ys.max() - ys.min()) / (xs.max() - xs.min())
    w = 4.0 if aspect <= 1.2 else 2.4
    fig, axs = plt.subplots(1, n, figsize=((w + 1.0) * n, w * min(aspect, 2.2) + 0.7), squeeze=False,
                            layout="constrained")
    for ax, (k, F) in zip(axs[0], fields.items()):
        F = _np(F)
        cm = cmap or _cmap_for(k, F, cmaps)
        lo, hi = _norm(F, cm)
        im = _contour(ax, xs, ys, F.T, cm, lo, hi)
        ax.set(title=k, aspect="equal")
        if ticks:
            ax.set(xlabel="x", ylabel="y")
        else:
            ax.set(xticks=[], yticks=[])
        _cbar(fig, ax, im)
    if streamplot and "u" in fields and "v" in fields:
        ax = axs[0][-1]
        ax.streamplot(xs, ys, _np(fields["u"]).T, _np(fields["v"]).T, color="w", density=1.2, linewidth=0.6)
    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(path)
    plt.close(fig)


def snapshot_rows(x, y, fields: dict, times, path: Path, n=4, title="", cmaps=None):
    """One row per field, n snapshots per row at evenly spaced times, no ticks, one colorbar per row
    (Fig. 3a / 3e style). fields[name][k, i, j] = f(x_i, y_j, t_k)."""
    xs, ys, times = _np(x), _np(y), _np(times)
    F = {k: _np(v) for k, v in fields.items()}
    idx = np.linspace(0, len(times) - 1, n + 1).round().astype(int)[1:]
    aspect = (ys.max() - ys.min()) / (xs.max() - xs.min())
    w = 2.6 if aspect <= 1.2 else 1.6
    fig, axs = plt.subplots(len(F), n, figsize=(w * n + 1.2, w * aspect * len(F) + 0.6), squeeze=False,
                            layout="constrained")
    for r, (name, A) in enumerate(F.items()):
        cm = _cmap_for(name, A, cmaps, default=evolution_cmap(A))
        lo, hi = _norm(A, cm)
        for c, k in enumerate(idx):
            ax = axs[r][c]
            im = _contour(ax, xs, ys, A[k].T, cm, lo, hi)
            ax.set(xticks=[], yticks=[], aspect="equal")
            if r == 0:
                ax.set_title(f"$t = {times[k]:.3g}$")
            if c == 0:
                ax.set_ylabel(name, fontsize=13)
        _ticks(fig.colorbar(im, ax=list(axs[r]), fraction=0.025, pad=0.01))
    if title:
        fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def gif_fields2d(x, y, fields: dict, times, path: Path, title="", fps=10, frames=50, cmaps=None):
    """Side-by-side animation of fields[name][k, i, j] = f(x_i, y_j, t_k), fixed color scales, no ticks."""
    xs, ys, times = _np(x), _np(y), _np(times)
    F = {k: _np(v) for k, v in fields.items()}
    idx = np.linspace(0, len(times) - 1, min(frames, len(times))).round().astype(int)
    n = len(F)
    aspect = (ys.max() - ys.min()) / (xs.max() - xs.min())
    w = 3.4 if aspect <= 1.2 else 2.2
    fig, axs = plt.subplots(1, n, figsize=((w + 0.8) * n, w * min(aspect, 2.2) + 0.7), squeeze=False,
                            layout="constrained")
    ims, styles = [], []
    for ax, (k, A) in zip(axs[0], F.items()):
        cm = _cmap_for(k, A, cmaps, default=evolution_cmap(A))
        lo, hi = _norm(A, cm)
        im = _contour(ax, xs, ys, A[0].T, cm, lo, hi)
        ax.set(title=k, xticks=[], yticks=[], aspect="equal")
        _cbar(fig, ax, im)
        ims.append(im)
        styles.append((cm, lo, hi))
    sup = fig.suptitle("")

    def upd(j):
        for m, (ax, A, (cm, lo, hi)) in enumerate(zip(axs[0], F.values(), styles)):
            ims[m].remove()
            ims[m] = _contour(ax, xs, ys, A[idx[j]].T, cm, lo, hi)
        sup.set_text(f"{title}   t = {times[idx[j]]:.3f}")
        return ims

    FuncAnimation(fig, upd, frames=len(idx), blit=False).save(path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)


def gif2d(x, y, frames, times, path: Path, title="", fps=8, cmap=None):
    """Single-field animation; frames[k, i, j] = f(x_i, y_j, t_k)."""
    gif_fields2d(x, y, {"": frames}, times, path, title=title, fps=fps, cmaps={"": cmap} if cmap else None)


def cavity(x, y, u, v, path: Path, title=""):
    """Steady 2D flow: speed in jet with white streamlines, plus centerline profiles u(0.5, y), v(x, 0.5)
    (Fig. 3c style). u[i, j] = u(x_i, y_j)."""
    xs, ys, u, v = _np(x), _np(y), _np(u), _np(v)
    speed = np.sqrt(u**2 + v**2)
    fig = plt.figure(figsize=(12.5, 3.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
    ax = fig.add_subplot(gs[0])
    im = _contour(ax, xs, ys, speed.T, "jet", 0.0, float(speed.max()))
    ax.streamplot(xs, ys, u.T, v.T, color="w", density=1.6, linewidth=0.6, arrowsize=0.6)
    ax.set(title="|u|", xticks=[], yticks=[], aspect="equal", xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]))
    _cbar(fig, ax, im)
    i, j = np.abs(xs - 0.5 * (xs[0] + xs[-1])).argmin(), np.abs(ys - 0.5 * (ys[0] + ys[-1])).argmin()
    for a, (xx, yy, lab) in zip([fig.add_subplot(gs[1]), fig.add_subplot(gs[2])],
                                [(ys, u[i, :], f"$u({xs[i]:.2g}, y)$"), (xs, v[:, j], f"$v(x, {ys[j]:.2g})$")]):
        a.plot(xx, yy, color="#e8604c", lw=1.8)
        a.set(title=lab, xlabel="y" if "u(" in lab else "x")
        a.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)
