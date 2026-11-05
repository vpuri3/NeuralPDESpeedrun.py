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

# Ordered list of tracks, by dimension: the ODE, then 1D steady, 1D + time, 2D steady, 2D + time, ... Folder
# name == track id == "<dim>D[_Steady]_<ProblemName>": the spatial dimension (0D for ODEs), "_Steady" for steady
# problems (time-dependent otherwise), and the problem name. Ids carry no numbers, so adding or removing a track
# never renames the others.
TRACKS = [
    "0D_VanDerPol",                   # stiff ODE: Van der Pol oscillator, mu = 1000
    "1D_Steady_AdvectionDiffusion",   # steady advection-diffusion, outflow boundary layer (Chebfun)
    "1D_DampedWave",                  # damped wave, narrow pulse, many reflections
    "1D_Burgers",                     # viscous Burgers, Raissi et al. benchmark
    "1D_KuramotoSivashinsky",         # Kuramoto-Sivashinsky, chaotic, periodic (published data)
    "1D_AllenCahn",                   # Allen-Cahn, periodic (published data)
    "1D_KdV",                         # Korteweg-de Vries, periodic (published data)
    "2D_Steady_Helmholtz",            # Helmholtz scattering by a square in a disk (TODO)
    "2D_Steady_AdvectionDiffusion",   # steady advection-diffusion, cellular flow (Wolfram Demonstration)
    "2D_Steady_Kovasznay",            # Kovasznay flow, Re = 40 (steady Navier-Stokes)
    "2D_Steady_LidDrivenCavity",      # lid-driven cavity, Re = 5000, steady (published data)
    "2D_CylinderFlow",                # flow past a cylinder (TODO)
    "2D_GreyScott",                   # Grey-Scott reaction-diffusion, periodic (published data)
    "2D_GinzburgLandau",              # complex Ginzburg-Landau, periodic (published data)
    "2D_RayleighTaylor",              # Rayleigh-Taylor instability, Ra = 1e6 (published data)
    "2D_KolmogorovFlow",              # Kolmogorov flow, Re = 1e4, periodic (published data)
]

# Reference solutions are not in git: they are assets of this GitHub release, one <track id>.npz per track,
# downloaded on first use into tracks/<track>/reference/ and checked against the sha256 in its meta.json.
REFS_RELEASE = "https://github.com/vpuri3/NeuralPDESpeedrun.py/releases/download/refs-v1"


def resolve(track: str) -> str:
    """Full track id from the id itself (case-insensitive) or, when unique, the problem name ("Burgers")."""
    hits = [k for k in TRACKS if k.lower() == track.lower()] or [k for k in TRACKS if name(k).lower() == track.lower()]
    if len(hits) != 1:
        raise KeyError(f"unknown or ambiguous track {track!r}; choose from {TRACKS}")
    return hits[0]


def key(track: str) -> str:
    """The id of a track, as used in figure markers and labels ("Burgers" -> "1D_Burgers")."""
    return resolve(track)


def dim_label(track: str) -> str:
    """Human label of the dimension: "0D_*" -> "ODE", "1D_Steady_*" -> "1D, steady", "2D_*" -> "2D + time"."""
    parts = resolve(track).split("_")
    if parts[0] == "0D":
        return "ODE"
    return f"{parts[0]}, steady" if parts[1] == "Steady" else f"{parts[0]} + time"


def name(track: str) -> str:
    """Problem name of a track: "1D_Burgers" -> "Burgers", "2D_Steady_Kovasznay" -> "Kovasznay"."""
    return track.rsplit("_", 1)[1]


def track_dir(track: str) -> Path:
    return TRACKS_DIR / resolve(track)


def load_problem(track: str, device: str = "cpu", dtype=torch.float64):
    mod = importlib.import_module(f"tracks.{resolve(track)}.problem")
    return mod.make_problem(device=device, dtype=dtype)


def reference_path(track: str) -> Path:
    return track_dir(track) / "reference" / "ref.npz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_reference(track: str, u, meta: dict) -> Path:
    u = u.numpy() if hasattr(u, "numpy") else np.asarray(u)
    u = np.asarray(u, dtype=np.float64)
    p = reference_path(track)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, u=u)
    meta = dict(meta)
    meta["shape"] = list(u.shape)
    meta["sha256_float32"] = hashlib.sha256(np.ascontiguousarray(u.astype(np.float32)).tobytes()).hexdigest()
    meta["sha256_npz"] = _sha256(p)
    meta["norm_l2"] = float(np.linalg.norm(u))
    meta["max_abs"] = float(np.abs(u).max())
    (p.parent / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return p


def fetch_reference(track: str) -> Path:
    """Local path of the track's reference, downloading it from the release if missing. A download must match
    the sha256 recorded in reference/meta.json."""
    import urllib.request
    track = resolve(track)
    p = reference_path(track)
    if p.exists():
        return p
    meta = json.loads((p.parent / "meta.json").read_text())
    url = f"{REFS_RELEASE}/{track}.npz"
    tmp = p.with_suffix(".npz.part")
    print(f"    downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    got = _sha256(tmp)
    if got != meta["sha256_npz"]:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {url}: {got} != {meta['sha256_npz']}. Rebuild the reference with\n"
                           f"  uv run python -m speedrun.make_reference --track {track}")
    tmp.rename(p)
    return p


def load_reference(track: str) -> torch.Tensor:
    return torch.from_numpy(np.load(fetch_reference(track))["u"]).to(torch.float64)
