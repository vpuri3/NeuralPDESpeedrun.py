"""Reference data from external sources, pinned by commit and sha256.

Tracks 05C, 07 and 11-16 use the reference solutions published with the PirateNet / SOAP-for-PINNs
benchmarks (github.com/PredictiveIntelligenceLab/jaxpi, branch `pirate`). Files are downloaded once into
~/.cache/neuralpdespeedrun/ and checked against the sha256 recorded here. Pickled .npy files are read with a
restricted unpickler that only reconstructs numpy arrays and plain containers.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np

JAXPI_COMMIT = "9b5196b846285c32a8f2e337982d19699de85956"
JAXPI_RAW = f"https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/{JAXPI_COMMIT}/examples/"
JAXPI_FILES = {
    "allen_cahn/data/allen_cahn.mat": "ce640f188e334520f636d3d650cae6056a92a486546d40889bff93610bdbfa71",
    "kdv/data/kdv.mat": "0df03aee69597c8913f7120324303f663db617be3789b9ff35beb29303b1c087",
    "ks_chaotic/data/ks_chaotic.mat": "8319ea9b6db085f26eefe9f0649d9367e1cca78f1c0a59f629d2cdaa6df3ed9a",
    "grey_scott/data/grey_scott.mat": "9894248f0c1fde33d94d5f6a9ca812bbc8cb9caeb23d39b5ab1e6831c5ae4cea",
    "ginzburg_landau/data/ginzburg_landau_square.mat":
        "322c320802a0229c6c9d88e186539cbb0e0b3a8b07c7587e2f0b1e1f247e6030",
    "ldc/data/ldc_Re5000.mat": "95f465f56b95379ae06127a027aeeed1c4d0d8b71d44df5dbdb50b432f5a460c",
    "rayleigh_taylor/data/rayleigh_taylor_high_Ra.npy":
        "8436139e0f8652c2bf9403b487dfca054e998890841a92edb2bd259d156867d9",
    "kolmogorov_flow/data/kolmogorov_flow_Re10000_256.npy":
        "a0cd72e29a8ae2cadeeb032fb8a460de73e5efef4f7dac9ce21a6bf6cf31c1d4",
}
CACHE = Path(os.environ.get("SPEEDRUN_CACHE", Path.home() / ".cache" / "neuralpdespeedrun"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jaxpi_file(rel: str) -> Path:
    """Local path of a pinned jaxpi data file, downloading and verifying it if needed."""
    sha = JAXPI_FILES[rel]
    path = CACHE / "jaxpi" / JAXPI_COMMIT[:7] / rel
    if not path.exists() or _sha256(path) != sha:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        print(f"    downloading {JAXPI_RAW + rel}")
        urllib.request.urlretrieve(JAXPI_RAW + rel, tmp)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink()
            raise RuntimeError(f"sha256 mismatch for {rel}: {got} != {sha}")
        tmp.rename(path)
    return path


def load_mat(rel: str) -> dict:
    import scipy.io
    return {k: v for k, v in scipy.io.loadmat(jaxpi_file(rel)).items() if not k.startswith("__")}


_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("builtins", "dict"), ("builtins", "list"),
            ("builtins", "tuple"), ("collections", "OrderedDict")}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name} while reading reference data")


def load_npy_dict(rel: str) -> dict:
    """A .npy file holding a pickled dict of arrays (np.save of a dict), read without arbitrary unpickling."""
    with open(jaxpi_file(rel), "rb") as fh:
        version = np.lib.format.read_magic(fh)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        reader(fh)
        obj = _Restricted(fh).load()
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    return obj
