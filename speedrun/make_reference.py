"""Generate (or re-plot) the reference solution of one or all tracks.

    uv run python -m speedrun.make_reference --track track05A_Burgers1D
    uv run python -m speedrun.make_reference --track all
    uv run python -m speedrun.make_reference --track track05A_Burgers1D --plot-only

Each track's `reference.py` exposes

    compute(device) -> (u: Tensor[problem.out_shape], meta: dict, extras: dict)
    plot(problem, u: Tensor, extras: dict, figdir: Path) -> None

Reference solutions are produced by high-resolution classical solvers written in
PyTorch (or by closed-form solutions where they exist) and are saved to
tracks/<track>/reference/ref.npz (+ meta.json).
"""
from __future__ import annotations

import argparse
import importlib
import time

import numpy as np
import torch

from .tracks import TRACKS, load_problem, load_reference, resolve, save_reference, track_dir


def make(track: str, device: str = "cpu", plot_only: bool = False):
    track = resolve(track)
    mod = importlib.import_module(f"tracks.{track}.reference")
    if plot_only:
        u, extras = load_reference(track), {}
        problem = load_problem(track, device="cpu")
    else:
        t0 = time.perf_counter()
        u, meta, extras = mod.compute(device=device)
        meta["wall_time_s"] = time.perf_counter() - t0
        meta["torch"] = torch.__version__
        problem = load_problem(track, device="cpu")     # after compute: data tracks write their problem data
        u = u.detach().to("cpu", torch.float64) if hasattr(u, "detach") else torch.as_tensor(np.asarray(u))
        assert tuple(u.shape) == problem.out_shape, (tuple(u.shape), problem.out_shape)
        save_reference(track, u, meta)
        print(f"[{track}] reference saved: shape={tuple(u.shape)} max|u|={float(u.abs().max()):.4f} "
              f"({meta['wall_time_s']:.1f}s) :: {meta.get('method', '')}")
        for k, v in meta.items():
            if k.startswith("check"):
                print(f"    {k}: {v}")
    figdir = track_dir(track) / "figures"
    figdir.mkdir(exist_ok=True)
    mod.plot(problem, u, extras, figdir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, help="track id or 'all'")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    for t in (TRACKS if a.track == "all" else [a.track]):
        if not (track_dir(t) / "reference.py").exists():
            print(f"[{t}] not implemented yet, skipped")
            continue
        make(t, a.device, a.plot_only)


if __name__ == "__main__":
    main()
