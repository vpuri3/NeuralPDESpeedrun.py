"""Problem specification shared by every track.

A `Problem` is the ONLY thing a submission receives. It contains the governing
equations (as text + parameters), the problem data (initial/boundary conditions,
forcing, coefficients) as callables, and the fixed evaluation points at which the
submission must return its solution. It never contains (any part of) the solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class Problem:
    track: str                      # e.g. "track3"
    title: str                      # short human title
    statement: str                  # markdown statement of the PDE problem (equations, domain, BC/IC)
    fields: tuple[str, ...]         # names of the output fields, e.g. ("u",) or ("u", "v")
    axes: dict[str, torch.Tensor]   # ordered tensor-product evaluation axes, e.g. {"t": ..., "x": ...}
    params: dict = field(default_factory=dict)              # scalar constants (viscosity, wavenumber, ...)
    fns: dict[str, Callable] = field(default_factory=dict)  # problem data: IC, BC, forcing, coefficients
    domain: dict = field(default_factory=dict)              # bounds / periodicity / time span
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

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

    def describe(self) -> str:
        ax = ", ".join(f"{k}[{v.numel()}]" for k, v in self.axes.items())
        return f"[{self.track}] {self.title}\n  fields={self.fields}  axes={ax}  out_shape={self.out_shape}"
