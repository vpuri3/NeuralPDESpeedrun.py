"""Track registry and reference-solution I/O."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "tracks"

# Ordered list of tracks. Folder name == track id.
TRACKS = [
    "track1",   # 3rd-order ODE (NeuralPDE.jl example)
    "track2A",  # 1D wave equation, Dirichlet
    "track2B",  # 1D damped wave equation
    "track3",   # 2D variable-coefficient diffusion, high contrast
    "track4A",  # 2D Helmholtz, heterogeneous medium, moderate frequency
    "track4B",  # 2D advection-diffusion, high Peclet
    "track5A",  # 1D viscous Burgers (SNF-ROM sec. 6.3)
    "track5B",  # 2D viscous Burgers (SNF-ROM sec. 6.4)
    "track5C",  # 1D Kuramoto-Sivashinsky (SNF-ROM sec. 6.5)
    "track6",   # Kovasznay flow (steady Navier-Stokes)
    "track7A",  # Lid-driven cavity, Re = 100
    "track7B",  # Lid-driven cavity, Re = 1000
    "track8",   # Steady Stokes, lid-driven cavity
]


def resolve(track: str) -> str:
    t = track if track.startswith("track") else f"track{track}"
    if t not in TRACKS:
        raise KeyError(f"unknown track {track!r}; choose from {TRACKS}")
    return t


def track_dir(track: str) -> Path:
    return TRACKS_DIR / resolve(track)


def load_problem(track: str, device: str = "cpu", dtype=torch.float64):
    mod = importlib.import_module(f"tracks.{resolve(track)}.problem")
    return mod.make_problem(device=device, dtype=dtype)


def reference_path(track: str) -> Path:
    return track_dir(track) / "reference" / "ref.npz"


def save_reference(track: str, u, meta: dict) -> Path:
    u = u.numpy() if hasattr(u, "numpy") else np.asarray(u)
    u = np.asarray(u, dtype=np.float64)
    p = reference_path(track)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, u=u)
    meta = dict(meta)
    meta["shape"] = list(u.shape)
    meta["sha256_float32"] = hashlib.sha256(np.ascontiguousarray(u.astype(np.float32)).tobytes()).hexdigest()
    meta["norm_l2"] = float(np.linalg.norm(u))
    meta["max_abs"] = float(np.abs(u).max())
    (p.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return p


def load_reference(track: str) -> torch.Tensor:
    p = reference_path(track)
    if not p.exists():
        raise FileNotFoundError(
            f"no reference for {track} at {p}. Generate it with\n"
            f"  uv run python -m speedrun.make_reference --track {resolve(track)}")
    return torch.from_numpy(np.load(p)["u"]).to(torch.float64)
