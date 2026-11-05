"""Shallow PINNs trained by Levenberg-Marquardt, after Shahab, Mukhlash & Susanto, "Do physics-informed neural
networks (PINNs) need to be deep? Shallow PINNs using the Levenberg-Marquardt algorithm" (arXiv:2602.08515).

- Network: tanh MLP with 1 or 2 hidden layers of `width` neurons and one output, u~(z). Inputs: a periodic axis
  enters as (sin, cos)(2 pi (x - lo) / L) (exact periodicity, as the paper), any other axis scaled to [-1, 1].
  Glorot-normal weights, zero biases, seed 0. All parameters in one flat vector w (ravel_pytree).
- Ansatz: the track may build the initial and boundary conditions into the output, u(z) = ansatz(u~, z), e.g.
  Burgers u = u~ (1 - x^2) t - sin(pi x) (the paper's eq. 42), or impose them as residual rows (bcs).
- Residual: F(w) = PDE residual at n_colloc fixed uniform random points of the domain (drawn once, seed 0), plus
  rows for any condition not built into the ansatz (scaled so each group counts like a mean).
  Derivatives in the inputs by Taylor-mode AD (jax.experimental.jet); the Jacobian J = dF/dw by reverse mode
  per point (vmap(jacrev)). Nothing hand-derived.
- Optimizer: textbook Levenberg-Marquardt, as the paper (MATLAB fsolve 'levenberg-marquardt'):
      solve (J^T J + mu I) d = J^T F (Cholesky, no preconditioning / scaling),  trial w - d;
      if ||F(w - d)||^2 < ||F(w)||^2: accept, mu <- mu / 10;  else: reject, mu <- 10 mu, retry the same iterate.
  mu0 = 0.01. Stops after max_iters iterations, or when ||d|| < step_tol (step_tol + ||w||) (fsolve's relative
  step tolerance), or when no damping up to mu_max decreases the loss.
- Execution: float64 throughout. The LM iteration (with its retries) runs on the device, check_every iterations
  per compiled call; the host checks the stopping flags and checkpoints between calls. Everything is compiled
  ahead of time in setup.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # the harness also uses torch on this GPU

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
F64 = jnp.float64
VERBOSE = bool(os.environ.get("PR0004_VERBOSE"))


@dataclass(frozen=True)
class Config:
    width: int = 32
    layers: int = 2             # hidden layers (1 or 2)
    n_colloc: int = 10000       # fixed collocation points
    max_iters: int = 20000
    mu0: float = 1e-2
    factor: float = 10.0
    mu_max: float = 1e16
    step_tol: float = 1e-12
    check_every: int = 50       # LM iterations per compiled call (one host sync + checkpoint each)
    seed: int = 0


def derivs(f: Callable, z, i: int, k: int):
    """(f, d f / d z_i, ..., d^k f / d z_i^k) at z, f: R^d -> R, by one Taylor-mode pass."""
    e = jnp.zeros_like(z).at[i].set(1.0)
    series = [e] + [jnp.zeros_like(z)] * (k - 1)
    y0, ys = jet.jet(f, (z,), (series,))
    return (y0, *ys)


def arch(width: int, layers: int):
    """(width, layers) of the track's network; PR0004_ARCH=<width>x<layers> overrides it (development sweeps)."""
    a = os.environ.get("PR0004_ARCH")
    return tuple(map(int, a.split("x"))) if a else (width, layers)


def face(problem, axis: str, value: float, n: int = 500, seed: int = 1):
    """n points on the face {axis = value} of the domain box, uniform random in the other axes."""
    names = problem.axis_names
    rng = np.random.default_rng(seed)
    Z = np.empty((n, len(names)))
    for i, k in enumerate(names):
        lo, hi = map(float, problem.domain[k])
        Z[:, i] = value if k == axis else lo + (hi - lo) * rng.random(n)
    return Z


def _init(key, sizes):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (a, b), F64) * math.sqrt(2.0 / (a + b)), jnp.zeros(b, F64)))
    return params


def _mlp(params, h):
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]


@dataclass(frozen=True)
class State:
    model: "LMPINN"
    w: jax.Array
    mu: float
    sse: float
    it: int


class LMPINN:
    """problem: the track's Problem. residual(u, z) -> scalar PDE residual, u: z -> scalar (use `derivs`), z the
    global coordinates in the order of problem.axes. ansatz(nn, z) -> u(z), nn: z -> u~(z): optional output
    transform building conditions into the network (default u = u~). bcs: extra residual groups (points (m, d),
    fn(u, z) -> scalar, weight) for conditions not built in; their rows are scaled by weight sqrt(n_colloc / m),
    so the least-squares loss is mean(PDE^2) + sum_k weight_k^2 mean(cond_k^2), times n_colloc."""

    def __init__(self, problem, residual: Callable, cfg: Config, ansatz: Callable | None = None, bcs=()):
        self.cfg, self.problem = cfg, problem
        names = problem.axis_names
        self.lo = np.array([float(problem.domain[k][0]) for k in names])
        self.hi = np.array([float(problem.domain[k][1]) for k in names])
        self.per = [problem.is_periodic(k) for k in names]
        n_in = sum(2 if p else 1 for p in self.per)
        sizes = [n_in] + [cfg.width] * cfg.layers + [1]
        w0, self.unravel = ravel_pytree(_init(jax.random.PRNGKey(cfg.seed), sizes))
        self.w0 = w0
        self.n_params = int(w0.size)
        rng = np.random.default_rng(cfg.seed)
        self.Z = self.lo + (self.hi - self.lo) * rng.random((cfg.n_colloc, len(names)))
        self.axes = [np.asarray(problem.axes[k].detach().cpu().numpy(), np.float64) for k in names]
        self.ansatz = ansatz or (lambda nn, z: nn(z))
        self.groups = [(self.Z, residual, 1.0)] + [
            (np.atleast_2d(np.asarray(Zb, np.float64)), fn, wt * math.sqrt(cfg.n_colloc / len(Zb)))
            for Zb, fn, wt in bcs]

    # network ---------------------------------------------------------------------------------------------
    def features(self, z):
        out = []
        for i, p in enumerate(self.per):
            s = (z[i] - self.lo[i]) / (self.hi[i] - self.lo[i])
            out += [jnp.sin(2 * jnp.pi * s), jnp.cos(2 * jnp.pi * s)] if p else [2 * s - 1]
        return jnp.stack(out)

    def u(self, w):
        params = self.unravel(w)
        nn = lambda z: _mlp(params, self.features(z))
        return lambda z: self.ansatz(nn, z)

    def F(self, w):
        return jnp.concatenate([jax.vmap(lambda z: c * fn(self.u(w), z))(jnp.asarray(Z)) for Z, fn, c in self.groups])

    def J(self, w):
        return jnp.concatenate([jax.vmap(jax.grad(lambda w, z: c * fn(self.u(w), z)), (None, 0))(w, jnp.asarray(Z))
                                for Z, fn, c in self.groups])

    # LM ----------------------------------------------------------------------------------------------------
    def _chunk(self, w, mu, sse):
        cfg = self.cfg
        F, Jf = self.F, self.J
        I = jnp.eye(w.size, dtype=F64)

        def iteration(carry):
            w, mu, sse, stop, k = carry
            r, J = F(w), Jf(w)
            A, g = J.T @ J, J.T @ r

            def trial(c):
                mu, _, _, _, _ = c
                L = jnp.linalg.cholesky(A + mu * I)
                d = jax.scipy.linalg.cho_solve((L, True), g)
                w2 = w - d
                s2 = jnp.sum(F(w2) ** 2)
                ok = s2 < sse          # NaN (failed factorization) counts as a rejection
                return jnp.where(ok, mu / cfg.factor, mu * cfg.factor), w2, s2, ok, jnp.linalg.norm(d)

            mu, w2, s2, ok, dn = jax.lax.while_loop(lambda c: (~c[3]) & (c[0] <= cfg.mu_max), trial,
                                                    (mu, w, sse, False, jnp.array(0.0, F64)))
            w = jnp.where(ok, w2, w)
            sse = jnp.where(ok, s2, sse)
            small = dn < cfg.step_tol * (cfg.step_tol + jnp.linalg.norm(w))
            stop = jnp.where(~ok, 2, jnp.where(small, 1, 0))     # 1: step tolerance, 2: mu > mu_max
            return w, mu, sse, stop, k + 1

        w, mu, sse, stop, k = jax.lax.while_loop(lambda c: (c[3] == 0) & (c[4] < cfg.check_every), iteration,
                                                 (w, mu, sse, jnp.array(0), jnp.array(0)))
        return w, mu, sse, stop, k

    def compile(self):
        s = jax.ShapeDtypeStruct
        sw, sc = s((self.n_params,), F64), s((), F64)
        self._chunk_c = jax.jit(self._chunk).lower(sw, sc, sc).compile()
        self._sse_c = jax.jit(lambda w: jnp.sum(self.F(w) ** 2)).lower(sw).compile()
        grid = jnp.asarray(np.stack([g.ravel() for g in np.meshgrid(*self.axes, indexing="ij")], -1))
        shape = (1, *[len(a) for a in self.axes])
        self._pred_c = jax.jit(lambda w: jax.vmap(self.u(w))(grid).reshape(shape)).lower(sw).compile()
        return self

    def train(self, checkpoint):
        cfg = self.cfg
        w, mu = self.w0, jnp.asarray(cfg.mu0, F64)
        sse = self._sse_c(w)
        it, t0 = 0, time.perf_counter()
        state = State(self, w, float(mu), float(sse), 0)
        checkpoint(state)
        while it < cfg.max_iters:
            w, mu, sse, stop, k = self._chunk_c(w, mu, sse)
            it += int(k)
            state = State(self, w, float(mu), float(sse), it)
            checkpoint(state)
            if VERBOSE:
                print(f"  it {it:5d}  loss {float(sse) / cfg.n_colloc:.3e}  mu {float(mu):.1e}  "
                      f"{time.perf_counter() - t0:7.1f} s", flush=True)
            if int(stop):
                if VERBOSE:
                    print(f"  stop: {'step tolerance' if int(stop) == 1 else 'mu > mu_max'}", flush=True)
                break
        return state

    def predict(self, state: State):
        return self._pred_c(state.w)


def solve(problem, model: LMPINN, checkpoint):
    return model.train(checkpoint)


def query(problem, state):
    return state.model.predict(state)
