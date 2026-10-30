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

# Ordered list of tracks: ODEs, then 1D + time, 2D steady, 2D + time, 3D + time. Folder name == track id ==
# "track<KK>_<dim>_<ProblemName>" with a two-digit number and dim one of 0DT, 1D, 1DT, 2D, 2DT, 3D, 3DT
# (spatial dimension, T = time-dependent).
TRACKS = [
    "track01_0DT_VanDerPol",             # stiff ODE: Van der Pol oscillator, mu = 1000
    "track02_1DT_DampedWave",            # damped wave, narrow pulse, many reflections
    "track03_1DT_Burgers",               # viscous Burgers, periodic
    "track04_1DT_KuramotoSivashinsky",   # Kuramoto-Sivashinsky, chaotic, periodic (published data)
    "track05_1DT_AllenCahn",             # Allen-Cahn, periodic (published data)
    "track06_1DT_KdV",                   # Korteweg-de Vries, periodic (published data)
    "track07_2D_HighContrastDiffusion",  # variable-coefficient diffusion, contrast 1e3
    "track08_2D_Helmholtz",              # Helmholtz, heterogeneous medium (on hold)
    "track09_2D_AdvectionDiffusion",     # advection-diffusion, high Peclet
    "track10_2D_Kovasznay",              # Kovasznay flow, Re = 40 (steady Navier-Stokes)
    "track11_2D_LidDrivenCavity",        # lid-driven cavity, Re = 5000, steady (published data)
    "track12_2DT_Burgers",               # viscous Burgers, periodic
    "track13_2DT_DecayingEddies",        # decaying eddies (Walsh 1992), periodic
    "track14_2DT_CylinderFlow",          # flow past a cylinder (TODO)
    "track15_2DT_GreyScott",             # Grey-Scott reaction-diffusion, periodic (published data)
    "track16_2DT_GinzburgLandau",        # complex Ginzburg-Landau, periodic (published data)
    "track17_2DT_RayleighTaylor",        # Rayleigh-Taylor instability, Ra = 1e6 (published data)
    "track18_2DT_KolmogorovFlow",        # Kolmogorov flow, Re = 1e4, periodic (published data)
    "track19_3DT_BeltramiFlow",          # 3D unsteady Beltrami flow, Re = 1 (exact solution)
]

# Reference solutions are not in git: they are assets of this GitHub release, one <track id>.npz per track,
# downloaded on first use into tracks/<track>/reference/ and checked against the sha256 in its meta.json.
REFS_RELEASE = "https://github.com/vpuri3/NeuralPDESpeedrun.py/releases/download/refs-v1"


def resolve(track: str) -> str:
    """Full track id from the full id, a short id ("track03") or a bare key ("03", "3")."""
    if track in TRACKS:
        return track
    m = re.fullmatch(r"(?:track)?0*(\d+)(?:_\w+)?", track)
    key = f"track{int(m.group(1)):02d}" if m else track
    hits = [k for k in TRACKS if k.split("_", 1)[0] == key]
    if len(hits) != 1:
        raise KeyError(f"unknown track {track!r}; choose from {TRACKS}")
    return hits[0]


def key(track: str) -> str:
    """Two-digit key of a track: "track03_1DT_Burgers" -> "03"."""
    return resolve(track).split("_", 1)[0].removeprefix("track")


def dim_label(track: str) -> str:
    """Human label of the dimension tag: "1DT" -> "1D + Time", "2D" -> "2D", "0DT" -> "ODE"."""
    tag = resolve(track).split("_")[1]
    if tag == "0DT":
        return "ODE"
    return f"{tag[:2]} + Time" if tag.endswith("T") else tag


def name(track: str) -> str:
    """Problem name of a track: "track03_1DT_Burgers" -> "Burgers"."""
    return resolve(track).split("_", 2)[2]


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
