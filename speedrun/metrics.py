"""Acceptance metrics. Identical for every track."""
from __future__ import annotations

import torch

REL_L2_TOL = 5e-3   # ||u - u_ref||_2 / ||u_ref||_2 over all fields and evaluation points
LINF_TOL = 1e-2     # max |u - u_ref| over all fields and evaluation points (absolute)


def rel_l2(u: torch.Tensor, ref: torch.Tensor) -> float:
    u = u.to(torch.float64)
    ref = ref.to(torch.float64)
    return float(torch.linalg.norm(u - ref) / torch.linalg.norm(ref))


def linf(u: torch.Tensor, ref: torch.Tensor) -> float:
    return float((u.to(torch.float64) - ref.to(torch.float64)).abs().max())


def score(u: torch.Tensor, ref: torch.Tensor) -> dict:
    if tuple(u.shape) != tuple(ref.shape):
        raise ValueError(f"solution shape {tuple(u.shape)} != expected {tuple(ref.shape)}")
    if not bool(torch.isfinite(u).all()):
        return {"rel_l2": float("inf"), "linf": float("inf"), "passed": False}
    r, m = rel_l2(u, ref), linf(u, ref)
    return {"rel_l2": r, "linf": m, "passed": (r <= REL_L2_TOL) and (m <= LINF_TOL)}
