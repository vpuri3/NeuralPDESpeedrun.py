"""Acceptance metrics: RelL2 and RelL∞ against the reference, with a per-track accuracy bar."""
from __future__ import annotations

import torch

# Default accuracy bar. A track may set its own (Problem.rel_l2_tol / Problem.rel_linf_tol). All fields and
# evaluation points are flattened into one vector ((n_fields, ...) -> (n_fields * N,)) and both errors are taken
# relative to that vector.
REL_L2_TOL = 5e-3     # ||u - ref||_2 / ||ref||_2
REL_LINF_TOL = 1e-2   # max |u - ref| / max |ref| over the scored points


def rel_l2(u: torch.Tensor, ref: torch.Tensor) -> float:
    u, ref = u.to(torch.float64).reshape(-1), ref.to(torch.float64).reshape(-1)
    return float(torch.linalg.norm(u - ref) / torch.linalg.norm(ref))


def rel_linf(u: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    e, r = (u.to(torch.float64) - ref.to(torch.float64)).abs(), ref.to(torch.float64).abs()
    if mask is not None:
        m = mask.expand_as(ref)
        e, r = e[m], r[m]
    return float(e.max() / r.max())


def score(u: torch.Tensor, ref: torch.Tensor, linf_mask: torch.Tensor | None = None,
          rel_l2_tol: float = REL_L2_TOL, rel_linf_tol: float = REL_LINF_TOL) -> dict:
    """linf_mask: optional boolean mask of the points where RelL∞ is scored (RelL2 always uses all points)."""
    if tuple(u.shape) != tuple(ref.shape):
        raise ValueError(f"solution shape {tuple(u.shape)} != expected {tuple(ref.shape)}")
    if not bool(torch.isfinite(u).all()):
        return {"rel_l2": float("inf"), "rel_linf": float("inf"), "passed": False}
    r, m = rel_l2(u, ref), rel_linf(u, ref, linf_mask)
    return {"rel_l2": r, "rel_linf": m, "passed": (r <= rel_l2_tol) and (m <= rel_linf_tol)}
