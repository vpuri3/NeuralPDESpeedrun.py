"""Acceptance metrics: RelL2 and L-inf against the reference, with a per-track accuracy bar."""
from __future__ import annotations

import torch

# Default accuracy bar. A track may set its own (Problem.rel_l2_tol / Problem.linf_tol).
REL_L2_TOL = 1e-4   # ||u - u_ref||_2 / ||u_ref||_2 over all fields and evaluation points
LINF_TOL = 1e-3     # max |u - u_ref| over all fields and evaluation points (absolute)


def rel_l2(u: torch.Tensor, ref: torch.Tensor) -> float:
    u = u.to(torch.float64)
    ref = ref.to(torch.float64)
    return float(torch.linalg.norm(u - ref) / torch.linalg.norm(ref))


def linf(u: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    e = (u.to(torch.float64) - ref.to(torch.float64)).abs()
    return float(e.max() if mask is None else e[mask.expand_as(e)].max())


def score(u: torch.Tensor, ref: torch.Tensor, linf_mask: torch.Tensor | None = None,
          rel_l2_tol: float = REL_L2_TOL, linf_tol: float = LINF_TOL) -> dict:
    """linf_mask: optional boolean mask of the points where L-inf is scored (RelL2 always uses all points)."""
    if tuple(u.shape) != tuple(ref.shape):
        raise ValueError(f"solution shape {tuple(u.shape)} != expected {tuple(ref.shape)}")
    if not bool(torch.isfinite(u).all()):
        return {"rel_l2": float("inf"), "linf": float("inf"), "passed": False}
    r, m = rel_l2(u, ref), linf(u, ref, linf_mask)
    return {"rel_l2": r, "linf": m, "passed": (r <= rel_l2_tol) and (m <= linf_tol)}
