"""Vanilla physics-informed neural network (PINN) building blocks.

A standard MLP u_theta(x) (tanh, inputs rescaled to [-1, 1]), PDE residuals by autograd, and plain
minimization of the mean-squared residual + boundary/initial-condition losses with Adam, collocation points
resampled uniformly every step. The only stopping criterion is a wall-clock budget (1000 s).
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Sequence

import torch
from torch import nn

BUDGET_S = 1000.0


def budget_s() -> float:
    """Training budget. The harness's untimed, unscored GPU check only needs a short run."""
    return 10.0 if os.environ.get("SPEEDRUN_DEVICE_CHECK") else BUDGET_S


class MLP(nn.Module):
    def __init__(self, lo: Sequence[float], hi: Sequence[float], out_dim: int, width: int = 64, depth: int = 4):
        super().__init__()
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        dims = [len(lo)] + [width] * depth
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        self.net = nn.Sequential(*layers, nn.Linear(width, out_dim))

    def forward(self, x):
        return self.net(2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y / d x for pointwise y (n,) and x (n, d) -> (n, d), differentiable."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def partials(y: torch.Tensor, x: torch.Tensor, *dims: int) -> torch.Tensor:
    """Mixed partial derivative of y w.r.t. x[:, dims[0]], x[:, dims[1]], ..."""
    for i in dims:
        y = grad(y, x)[:, i]
    return y


def uniform(n: int, lo: Sequence[float], hi: Sequence[float], device) -> torch.Tensor:
    lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
    hi_t = torch.tensor(hi, dtype=torch.float32, device=device)
    return (lo_t + (hi_t - lo_t) * torch.rand(n, len(lo), device=device)).requires_grad_(True)


def box_boundary(n: int, lo: Sequence[float], hi: Sequence[float], dims: Sequence[int], device) -> torch.Tensor:
    """n points uniform on the faces of the box normal to the given dims (the others uniform in range)."""
    x = uniform(n, lo, hi, device).detach()
    k = torch.randint(len(dims), (n,), device=device)
    side = torch.randint(2, (n,), device=device).to(x.dtype)
    for j, d in enumerate(dims):
        m = k == j
        x[m, d] = lo[d] + side[m] * (hi[d] - lo[d])
    return x.requires_grad_(True)


def train(model: nn.Module, loss_fn: Callable[[], torch.Tensor], budget: float | None = None,
          lr: float = 1e-3, lr_end: float = 1e-5) -> list[tuple[float, float]]:
    """Adam on loss_fn() until the wall-clock budget is spent; learning rate decays geometrically in time
    from lr to lr_end. Returns the (elapsed seconds, loss) history every 100 steps."""
    budget = budget_s() if budget is None else budget
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, t0, it = [], time.perf_counter(), 0
    while True:
        el = time.perf_counter() - t0
        if el >= budget:
            break
        for g in opt.param_groups:
            g["lr"] = lr * math.exp(math.log(lr_end / lr) * el / budget)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            hist.append((el, loss.item()))
        it += 1
    return hist


class PINN:
    """Solver object: network + loss. Built in `warmup`, trained in `solve`, evaluated in `query`."""

    def __init__(self, model: nn.Module, loss_fn: Callable[[], torch.Tensor], lr: float = 1e-3, lr_end: float = 1e-5):
        self.model, self.loss_fn, self.lr, self.lr_end = model, loss_fn, lr, lr_end
        self.history: list[tuple[float, float]] = []

    def compile(self):
        """Run the loss and its gradient once (CUDA kernels, autograd graph) without updating the weights."""
        self.loss_fn().backward()
        self.model.zero_grad(set_to_none=True)
        return self

    def train(self, budget: float | None = None):
        self.history = train(self.model, self.loss_fn, budget, self.lr, self.lr_end)
        return self


@torch.no_grad()
def predict_grid(model: nn.Module, axes: Sequence[torch.Tensor], to_points: Callable | None = None,
                 fields: Sequence[int] = (0,), dtype=torch.float64, chunk: int = 1 << 18) -> torch.Tensor:
    """Evaluate the model on the tensor grid of `axes` -> (len(fields), *grid). `to_points` maps the grid
    coordinates (list of flattened tensors) to model inputs (n, d); default: stack them."""
    grids = torch.meshgrid(*[a.to(torch.float32) for a in axes], indexing="ij")
    flat = [g.reshape(-1) for g in grids]
    X = to_points(flat) if to_points is not None else torch.stack(flat, dim=1)
    out = torch.cat([model(X[i:i + chunk])[:, list(fields)] for i in range(0, X.shape[0], chunk)])
    return out.T.reshape(len(fields), *grids[0].shape).to(dtype)
