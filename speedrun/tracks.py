"""Track registry and reference-solution I/O."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "tracks"

# Ordered list of tracks. Folder name == track id == "track<KK>[A|B]_<ProblemName>" (two-digit number).
TRACKS = [
    "track01_ThirdOrderODE",           # 3rd-order ODE (NeuralPDE.jl example)
    "track02A_Wave",                   # 1D wave equation, Dirichlet
    "track02B_DampedWave",             # 1D damped wave equation
    "track03_HighContrastDiffusion",   # 2D variable-coefficient diffusion, high contrast
    "track04A_Helmholtz",              # 2D Helmholtz, heterogeneous medium, moderate frequency
    "track04B_AdvectionDiffusion",     # 2D advection-diffusion, high Peclet
    "track05A_Burgers1D",              # 1D viscous Burgers (SNF-ROM sec. 6.3)
    "track05B_Burgers2D",              # 2D viscous Burgers (SNF-ROM sec. 6.4)
    "track05C_KuramotoSivashinsky",    # 1D Kuramoto-Sivashinsky (SNF-ROM sec. 6.5)
    "track06A_Kovasznay",              # Kovasznay flow, Re = 40, rectangle (steady Navier-Stokes)
    "track06B_KovasznayCurved",        # Kovasznay flow, Re = 40, curved trapezoid
    "track07A_CavityRe100",            # Lid-driven cavity, Re = 100, impulsive start, t in [0, 5]
    "track07B_CavityRe1000",           # Lid-driven cavity, Re = 1000, impulsive start, t in [0, 5]
    "track08_StokesCavity",            # Steady Stokes, lid-driven cavity
    "track09_DecayingEddies",          # Decaying eddies (Walsh 1992), periodic
    "track10_CylinderFlow",            # Flow past a cylinder (TODO: not set up yet)
]


def resolve(track: str) -> str:
    """Full track id from the full id, a short id ("track07A", "track7A") or a bare key ("07A", "7A")."""
    if track in TRACKS:
        return track
    m = re.fullmatch(r"(?:track)?0*(\d+)([A-Z]?)(?:_\w+)?", track)
    key = f"track{int(m.group(1)):02d}{m.group(2)}" if m else track
    hits = [k for k in TRACKS if k.split("_", 1)[0] == key]
    if len(hits) != 1:
        raise KeyError(f"unknown track {track!r}; choose from {TRACKS}")
    return hits[0]


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
