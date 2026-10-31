"""Acceptance metrics: RelL2 and RelL∞ against the reference, per field, with a per-track accuracy bar."""
from __future__ import annotations

import torch

# Default accuracy bar. A track may set its own (Problem.rel_l2_tol / Problem.rel_linf_tol). Both errors are
# relative to the size of each field and the worst field counts, so a small field (e.g. v next to u) is held to
# the same relative accuracy as a large one.
REL_L2_TOL = 5e-3     # max_k ||u_k - ref_k||_2 / ||ref_k||_2 over the evaluation points
REL_LINF_TOL = 1e-2   # max_k max |u_k - ref_k| / max |ref_k| over the scored points


def _fields(u: torch.Tensor, ref: torch.Tensor):
    return u.to(torch.float64).reshape(u.shape[0], -1), ref.to(torch.float64).reshape(ref.shape[0], -1)


def rel_l2(u: torch.Tensor, ref: torch.Tensor) -> float:
    u, ref = _fields(u, ref)
    return float((torch.linalg.norm(u - ref, dim=1) / torch.linalg.norm(ref, dim=1)).max())


def rel_linf(u: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        m = mask.expand_as(ref).reshape(ref.shape[0], -1)
    u, ref = _fields(u, ref)
    e, r = (u - ref).abs(), ref.abs()
    if mask is not None:
        e, r = e.where(m, 0.0), r.where(m, 0.0)
    return float((e.amax(dim=1) / r.amax(dim=1)).max())


def score(u: torch.Tensor, ref: torch.Tensor, linf_mask: torch.Tensor | None = None,
          rel_l2_tol: float = REL_L2_TOL, rel_linf_tol: float = REL_LINF_TOL) -> dict:
    """u, ref: (n_fields, ...). linf_mask: optional boolean mask of the points where RelL∞ is scored (RelL2
    always uses all points)."""
    if tuple(u.shape) != tuple(ref.shape):
        raise ValueError(f"solution shape {tuple(u.shape)} != expected {tuple(ref.shape)}")
    if not bool(torch.isfinite(u).all()):
        return {"rel_l2": float("inf"), "rel_linf": float("inf"), "passed": False}
    r, m = rel_l2(u, ref), rel_linf(u, ref, linf_mask)
    return {"rel_l2": r, "rel_linf": m, "passed": (r <= rel_l2_tol) and (m <= rel_linf_tol)}
