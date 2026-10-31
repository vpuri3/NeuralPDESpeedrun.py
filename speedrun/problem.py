"""Problem specification shared by every track.

A `Problem` is the ONLY thing a submission receives. It contains the governing
equations (as text + parameters), the problem data (initial/boundary conditions,
forcing, coefficients) as callables, and the fixed evaluation points at which the
submission must return its solution. It never contains (any part of) the solution.

The callables in `fns` are written against the array API (`xp_of`), so they take and return torch tensors or
JAX arrays alike (same namespace, device and dtype out as in). Axes and sampled data are torch tensors; a JAX
submission converts them once in `setup` (e.g. `jnp.asarray(a.cpu().numpy())`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .metrics import REL_L2_TOL, REL_LINF_TOL

BUDGET_S = 1000.0   # default wall-clock budget of the timed solve, enforced by the harness


def xp_of(*arrays):
    """Array namespace (torch or jax.numpy, array-API compatible) of the arguments."""
    from array_api_compat import array_namespace
    return array_namespace(*arrays)


def array_device(x):
    from array_api_compat import device
    return device(x)


@dataclass
class Problem:
    track: str                      # e.g. "track07_2D_HighContrastDiffusion"
    title: str                      # short human title
    statement: str                  # markdown statement of the PDE problem (equations, domain, BC/IC)
    fields: tuple[str, ...]         # names of the output fields, e.g. ("u",) or ("u", "v")
    axes: dict[str, torch.Tensor]   # ordered tensor-product evaluation axes, e.g. {"t": ..., "x": ...}
    params: dict = field(default_factory=dict)              # scalar constants (viscosity, wavenumber, ...)
    fns: dict[str, Callable] = field(default_factory=dict)  # problem data: IC, BC, forcing, coefficients
    domain: dict = field(default_factory=dict)              # bounds / time span
    periodic: dict = field(default_factory=dict)            # per spatial axis, e.g. {"x": True, "y": False}
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    rel_l2_tol: float = REL_L2_TOL  # accuracy bar (the speedrun-wide one on every track)
    rel_linf_tol: float = REL_LINF_TOL
    linf_margin: float = 0.0        # RelL∞ is not scored within this distance of the (x, y) domain boundary
    budget_s: float = BUDGET_S      # the harness stops the timed solve once this much wall-clock time is used

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(self.axes.keys())

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Shape of the tensor a submission must return from `solve(problem)`:
        (n_fields, len(axis_0), len(axis_1), ...)."""
        return (len(self.fields), *[int(a.numel()) for a in self.axes.values()])

    def grid(self) -> list[torch.Tensor]:
        """Tensor-product mesh of the evaluation axes (indexing='ij')."""
        return list(torch.meshgrid(*self.axes.values(), indexing="ij"))

    def is_periodic(self, axis: str) -> bool:
        return bool(self.periodic.get(axis, False))

    def linf_mask(self) -> torch.Tensor | None:
        """Boolean mask (broadcastable to out_shape) of the points where RelL∞ is scored, or None (all)."""
        if self.linf_margin <= 0:
            return None
        names = self.axis_names
        m = torch.ones([a.numel() for a in self.axes.values()], dtype=torch.bool)
        for i, k in enumerate(names):
            if k in ("x", "y"):
                a = self.axes[k].detach().cpu()
                lo, hi = self.domain[k]
                keep = torch.minimum(a - lo, hi - a) >= self.linf_margin - 1e-12
                shape = [1] * len(names)
                shape[i] = -1
                m = m & keep.reshape(shape)
        return m[None]

    def describe(self) -> str:
        ax = ", ".join(f"{k}[{v.numel()}]" for k, v in self.axes.items())
        return f"[{self.track}] {self.title}\n  fields={self.fields}  axes={ax}  out_shape={self.out_shape}"
