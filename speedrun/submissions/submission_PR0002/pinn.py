"""Physics-informed networks trained by Gauss-Newton (gn.py), for every track shape: steady (no time axis) or
time-marching (JAX-NG's windowed hard-IC scheme), 1-3 spatial axes, vector outputs.

Outputs U = (state fields..., free fields...). State fields carry the initial condition; free fields (e.g.
pressure) have none. Steady problems (no "t" axis): U = hard(z, phi(x)), one network, `steps` GN steps.

Time-dependent problems: [t0, t1] is split into n_windows windows of length T, each with its own network phi_w
(warm-started from phi_{w-1}). On window w, t = t_w + tau, tau in [0, T]; for the state fields

    order 1:  U = s_w(x) + phi_w(tau, x) - phi_w(0, x),   s_w = u0 + sum_{j<w} [phi_j(T) - phi_j(0)]
    order 2:  U = s_w + tau v_w + phi_w(tau) - phi_w(0) - tau phi_w,tau(0),
              v_w = v0 + sum_{j<w} dv_j,  s_w = u0 + w T v0 + sum_{j<w} [du_j + T (w - 1 - j) dv_j],
              du_j = phi_j(T) - phi_j(0) - T phi_j,tau(0),   dv_j = phi_j,tau(T) - phi_j,tau(0)

so the initial condition and the continuity between windows hold exactly; free fields are phi_w(tau, x).
The initial state is given as JAX-traceable callables, or as samples on a grid (ic_data), which are then fitted
by a separate network (GN, ic_steps steps, at the start of solve; timed).

phi is a sine MLP of (2 tau / T - 1, features of x): (sin, cos)(2 pi k x / L), k = 1..n_modes, for a periodic
axis (exact periodicity), the coordinate scaled to [-1, 1] otherwise. An axis in `odd_axes` (U = 0 on both
walls) is treated as periodic with period 2 L and phi made odd about its lower wall.

Loss: the PDE residual (residual(U, z) -> vector, z = (t, x, ...) global coordinates, U(z) -> outputs; use
net.partials for derivatives) at n_colloc uniformly resampled points per step (axes in cluster_axes sampled
with the Chebyshev density, denser at the walls), plus boundary terms (BC: residual on one or both walls of an
axis, n_bc points each). Each window runs `steps` GN steps, or fewer once the loss is below loss_tol.

All networks are flat parameter vectors, the windows' stacked (n_windows, n); the checkpointed state is
(model, mem, w), w the current window. The query accumulates the window start states on the output grid, so
its cost does not grow with the number of windows. Everything is compiled ahead of time in setup.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # the harness also uses torch on this GPU

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .gn import GNConfig, Term, gn_step
from .net import init_mlp, mlp

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64
VERBOSE = bool(os.environ.get("PR0002_VERBOSE"))   # per-check loss lines (development)


@dataclass(frozen=True)
class Config:
    n_windows: int = 1
    width: int = 50
    depth: int = 4                 # hidden layers
    n_modes: int = 10              # Fourier modes per periodic axis
    n_colloc: int = 20000          # interior collocation points per step (resampled every step)
    n_bc: int = 2000               # points per boundary term per step
    steps: int = 100               # GN steps per window (at most)
    loss_tol: float = 0.0          # move to the next window once the loss is below this
    time_order: int = 1            # 1: u_t = ...; 2: u_tt = ...; ignored without a time axis
    odd_axes: tuple = ()           # spatial axes with U = 0 on both walls (odd periodic extension)
    cluster_axes: tuple = ()       # spatial axes sampled denser at the walls (Chebyshev density)
    check_every: int = 10          # steps between host reads of the loss (logging, loss_tol)
    ic_steps: int = 100            # GN steps fitting the initial-state network (ic_data only)
    ic_points: int = 20000         # grid samples per IC-fit step
    ic_width: int = 64
    ic_depth: int = 4
    gn: GNConfig = field(default_factory=GNConfig)
    seed: int = 0


@dataclass(frozen=True)
class BC:
    """Boundary residual fn(U, z) -> vector on the wall(s) of `axis`: side "lo", "hi" or "both"."""
    axis: str
    fn: Callable
    side: str = "both"
    weight: float = 1.0


class PINN:
    """problem: the track's Problem. residual(U, z) -> vector. n_out: network outputs (problem fields first).
    ic: time-dependent problems, order 1: tuple of callables u0_k(*xs), one per state field; order 2: a pair of
    such tuples (u0s, v0s). ic_data: alternatively {"axes": [spatial axes], "values": (n_state, *grid)}.
    bcs: boundary terms. hard(z, U) -> U: optional output transform (e.g. a factor vanishing on the walls)."""

    def __init__(self, problem, residual: Callable, n_out: int, cfg: Config, ic=None, ic_data=None, bcs=(),
                 hard: Callable | None = None):
        names = problem.axis_names
        self.cfg, self.residual, self.n_out, self.bcs, self.hard = cfg, residual, n_out, tuple(bcs), hard
        self.n_fields = len(problem.fields)
        self.timed = names[0] == "t"
        self.space = names[1:] if self.timed else names
        self.order = cfg.time_order if self.timed else 0
        self.nw = cfg.n_windows if self.timed else 1
        self.t0, t1 = map(float, problem.domain["t"]) if self.timed else (0.0, 1.0)
        self.T = (t1 - self.t0) / self.nw
        self.lo = [float(problem.domain[k][0]) for k in self.space]
        self.hi = [float(problem.domain[k][1]) for k in self.space]
        self.odd = [k in cfg.odd_axes for k in self.space]
        self.periodic = [problem.is_periodic(k) or o for k, o in zip(self.space, self.odd)]
        self.period = [(h - lo) * (2.0 if o else 1.0) for lo, h, o in zip(self.lo, self.hi, self.odd)]
        self.cluster = [k in cfg.cluster_axes for k in self.space]
        self.axes = [np.asarray(problem.axes[k].detach().cpu().numpy(), dtype=np.float64) for k in names]
        if self.timed:
            if ic_data is not None:
                assert self.order == 1
                self.ic = None
                self.n_state = len(ic_data["values"])
            else:
                self.ic = (tuple(ic),) if self.order == 1 else (tuple(ic[0]), tuple(ic[1]))
                self.n_state = len(self.ic[0])
        else:
            self.ic, self.n_state = None, 0
        self.ic_data = ic_data
        n_in = int(self.timed) + sum(2 * cfg.n_modes if p else 1 for p in self.periodic)
        p0 = init_mlp(jax.random.PRNGKey(cfg.seed), [n_in] + [cfg.width] * cfg.depth + [n_out])
        self.theta0, self.unravel = ravel_pytree(p0)
        self.n_params = self.theta0.size
        if ic_data is not None:
            q0 = init_mlp(jax.random.PRNGKey(cfg.seed + 7),
                          [n_in - 1] + [cfg.ic_width] * cfg.ic_depth + [self.n_state])
            self.ic_theta0, self.ic_unravel = ravel_pytree(q0)
        self.log: list[dict] = []

    # ------------------------------------------------------------------ networks
    def features(self, tau, xs):
        f = [2.0 * tau / self.T - 1.0] if (self.timed and tau is not None) else []
        for x, lo, hi, per, L in zip(xs, self.lo, self.hi, self.periodic, self.period):
            if per:
                w = 2.0 * jnp.pi * (x - lo) / L * jnp.arange(1, self.cfg.n_modes + 1, dtype=F64)
                f += [jnp.sin(w), jnp.cos(w)]
            else:
                f.append(2.0 * (x - lo) / (hi - lo) - 1.0)
        return jnp.concatenate([jnp.atleast_1d(a) for a in f])

    def _odd(self, net, xs):
        """net(xs) made odd about the lower wall of every axis in odd_axes."""
        out = 0.0
        for flips in itertools.product((False, True), repeat=sum(self.odd)):
            it, ys = iter(flips), []
            for x, lo, o in zip(xs, self.lo, self.odd):
                ys.append(2.0 * lo - x if (o and next(it)) else x)
            out = out + (-1.0) ** sum(flips) * net(ys)
        return out

    def phi(self, theta, tau, xs):
        p = self.unravel(theta)
        return self._odd(lambda ys: mlp(p, self.features(tau, ys)), xs)

    def phi_tau(self, theta, tau, xs):
        return jax.jvp(lambda s: self.phi(theta, s, xs), (tau,), (jnp.ones_like(tau),))[1]

    def u0(self, mem, xs):
        """The initial state (state fields), and its time derivative for order 2."""
        if self.ic is None:
            q = self.ic_unravel(mem["ic"])
            return (self._odd(lambda ys: mlp(q, self.features(None, ys)), xs),)
        return tuple(jnp.stack([f(*xs) for f in fs]) for fs in self.ic)

    def u(self, theta, mem, w, tau, xs):
        """All outputs at local time tau of window w (network theta); past windows' networks in mem["theta"]."""
        if not self.timed:
            return self._hard(None, xs, self.phi(theta, None, xs))
        T, tau, ns = self.T, jnp.asarray(tau, F64), self.n_state
        z = jnp.zeros((), F64)
        stack = mem["theta"]
        past = (jnp.arange(stack.shape[0]) < w)[:, None]
        ic = self.u0(mem, xs)
        p_t, p_0 = self.phi(theta, tau, xs), self.phi(theta, z, xs)
        if self.order == 1:
            jump = jax.vmap(lambda q: self.phi(q, T + z, xs)[:ns] - self.phi(q, z, xs)[:ns])(stack)
            state = ic[0] + jnp.sum(jnp.where(past, jump, 0.0), 0) + p_t[:ns] - p_0[:ns]
        else:
            def jumps(q):
                d0, dT = self.phi_tau(q, z, xs)[:ns], self.phi_tau(q, T + z, xs)[:ns]
                return self.phi(q, T + z, xs)[:ns] - self.phi(q, z, xs)[:ns] - T * d0, dT - d0
            du, dv = jax.vmap(jumps)(stack)
            j = jnp.arange(stack.shape[0])[:, None]
            v_w = ic[1] + jnp.sum(jnp.where(past, dv, 0.0), 0)
            s_w = ic[0] + w * T * ic[1] + jnp.sum(jnp.where(past, du + T * (w - 1 - j) * dv, 0.0), 0)
            state = s_w + tau * v_w + p_t[:ns] - p_0[:ns] - tau * self.phi_tau(theta, z, xs)[:ns]
        return self._hard(self.t0 + w * T + tau, xs, jnp.concatenate([state, p_t[ns:]]))

    def _hard(self, t, xs, U):
        if self.hard is None:
            return U
        z = jnp.stack(([t] if self.timed else []) + list(xs))
        return self.hard(z, U)

    # ------------------------------------------------------------------ training
    def _uniform(self, key, n, axes_lohi):
        """n points, uniform per axis (Chebyshev density on clustered spatial axes)."""
        lo = jnp.asarray([a[0] for a in axes_lohi], F64)
        hi = jnp.asarray([a[1] for a in axes_lohi], F64)
        r = jax.random.uniform(key, (n, lo.size), F64)
        cl = jnp.asarray(([False] if self.timed else []) + self.cluster)
        r = jnp.where(cl, 0.5 * (1.0 - jnp.cos(jnp.pi * r)), r)
        return lo + (hi - lo) * r

    def _step(self, theta, mem, w, key):
        key, *subs = jax.random.split(key, 2 + 2 * len(self.bcs))
        tw = self.t0 + w * self.T
        box = ([(0.0, self.T)] if self.timed else []) + list(zip(self.lo, self.hi))
        off = jnp.asarray([tw] + [0.0] * len(self.space) if self.timed else [0.0] * len(self.space), F64)
        ts = int(self.timed)

        def U_of(th):
            if self.timed:
                return lambda z: self.u(th, mem, w, z[0] - tw, list(z[1:]))
            return lambda z: self.u(th, mem, w, None, list(z))

        terms = [Term(lambda th, zi: jnp.atleast_1d(jnp.asarray(self.residual(U_of(th), zi))),
                      self._uniform(subs[0], self.cfg.n_colloc, box) + off)]
        for b, (bc, k) in enumerate(zip(self.bcs, subs[1:])):
            i = ts + self.space.index(bc.axis)
            zb = self._uniform(k, self.cfg.n_bc, box) + off
            lo, hi = self.lo[i - ts], self.hi[i - ts]
            if bc.side == "both":
                side = jnp.where(jnp.arange(self.cfg.n_bc) % 2 == 0, lo, hi)
            else:
                side = jnp.full((self.cfg.n_bc,), lo if bc.side == "lo" else hi, F64)
            zb = zb.at[:, i].set(side)
            terms.append(Term(lambda th, zi, bc=bc: jnp.atleast_1d(jnp.asarray(bc.fn(U_of(th), zi))), zb, bc.weight))
        theta, loss, _ = gn_step(terms, theta, self.cfg.gn)
        return theta, key, loss

    def _ic_step(self, q, key):
        """One GN step of the initial-state fit on a random subset of the IC grid."""
        grid = [jnp.asarray(a, F64) for a in self.ic_data["axes"]]
        vals = jnp.asarray(self.ic_data["values"], F64).reshape(self.n_state, -1).T      # (G, n_state)
        shape = [g.size for g in grid]
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (self.cfg.ic_points,), 0, vals.shape[0])
        sub_i = jnp.stack(jnp.unravel_index(idx, shape), 1)
        pts = jnp.concatenate([jnp.stack([grid[d][sub_i[:, d]] for d in range(len(grid))], 1),
                               vals[idx]], 1)
        nd = len(grid)

        def res(qq, pz):
            return self.u0({"ic": qq}, list(pz[:nd]))[0] - pz[nd:]

        q, loss, _ = gn_step([Term(res, pts)], q, self.cfg.gn)
        return q, key, loss

    # ------------------------------------------------------------------ query
    def _predict(self, mem, w_done):
        """The scored fields on the output grid: problem.out_shape."""
        sp = [jnp.asarray(a, F64) for a in (self.axes[1:] if self.timed else self.axes)]
        X = jnp.stack([g.reshape(-1) for g in jnp.meshgrid(*sp, indexing="ij")], 1)    # (G, d)
        gshape = [a.size for a in sp]
        nf = self.n_fields
        if not self.timed:
            U = jax.vmap(lambda x: self.u(mem["theta"][0], mem, 0, None, list(x)))(X)
            return U[:, :nf].T.reshape(nf, *gshape)
        t = self.axes[0]
        win = np.clip(np.ceil((t - self.t0) / self.T - 1e-9).astype(int) - 1, 0, self.nw - 1)
        T, ns, z = self.T, self.n_state, jnp.zeros((), F64)
        ic = jax.vmap(lambda x: self.u0(mem, list(x)))(X)
        S = ic[0]
        V = ic[1] if self.order == 2 else None
        rows = []
        for j in range(self.nw):
            jj = jnp.minimum(j, w_done)
            theta, tj = mem["theta"][jj], self.t0 + jj * T
            p0 = jax.vmap(lambda x, th=theta: self.phi(th, z, list(x)))(X)
            d0 = jax.vmap(lambda x, th=theta: self.phi_tau(th, z, list(x)))(X) if self.order == 2 else None
            idx = np.nonzero(win == j)[0]
            if idx.size:
                taus = jnp.asarray(t[idx], F64) - tj

                def at(tau, x, s, v, a, b, th=theta, tj=tj):
                    p = self.phi(th, tau, list(x))
                    st = s + p[:ns] - a[:ns]
                    if self.order == 2:
                        st = st + tau * (v - b[:ns])
                    return self._hard(tj + tau, list(x), jnp.concatenate([st, p[ns:]]))

                f = jax.vmap(jax.vmap(at, (None, 0, 0, 0 if V is not None else None, 0, 0 if d0 is not None else None)),
                             (0, None, None, None, None, None))
                rows.append(f(taus, X, S, V, p0, d0)[..., :nf])
            pT = jax.vmap(lambda x, th=theta: self.phi(th, T + z, list(x)))(X)
            if self.order == 1:
                S_next = S + pT[:, :ns] - p0[:, :ns]
            else:
                dT = jax.vmap(lambda x, th=theta: self.phi_tau(th, T + z, list(x)))(X)
                S_next = S + T * V + pT[:, :ns] - p0[:, :ns] - T * d0[:, :ns]
                V = jnp.where(j < w_done, V + dT[:, :ns] - d0[:, :ns], V)
            S = jnp.where(j < w_done, S_next, S)
        U = jnp.concatenate(rows)                         # (nt, G, nf)
        return jnp.moveaxis(U, 2, 0).reshape(nf, len(t), *gshape)

    # ------------------------------------------------------------------ driver
    def compile(self):
        """Untimed: compile every jitted function (nothing is executed)."""
        S = jax.ShapeDtypeStruct
        n, k, w = self.n_params, jax.random.PRNGKey(0), S((), jnp.int32)
        mem = {"theta": S((self.nw, n), F64)}
        if self.ic_data is not None:
            mem["ic"] = S(self.ic_theta0.shape, F64)
            self.ic_step_c = jax.jit(self._ic_step).lower(mem["ic"], k).compile()
        self.step_c = jax.jit(self._step).lower(S((n,), F64), mem, w, k).compile()
        self.predict_c = jax.jit(self._predict).lower(mem, w).compile()
        return self

    def train(self, checkpoint):
        cfg = self.cfg
        theta = self.theta0
        mem = {"theta": jnp.zeros((self.nw, self.n_params), F64)}
        key = jax.random.PRNGKey(cfg.seed + 1)
        t_start = time.perf_counter()
        if self.ic_data is not None:
            q = self.ic_theta0
            for _ in range(cfg.ic_steps):
                q, key, loss_d = self.ic_step_c(q, key)
                checkpoint((self, {**mem, "ic": q}, 0))      # a valid state: the fitted IC, no evolution yet
            mem["ic"] = q
            print(f"ic fit: {cfg.ic_steps} steps, loss {float(loss_d):.3e}, {time.perf_counter() - t_start:.1f} s",
                  flush=True)
        for w in range(self.nw):
            wj, t_w, loss = jnp.int32(w), time.perf_counter(), float("nan")
            for s in range(1, cfg.steps + 1):
                theta, key, loss_d = self.step_c(theta, mem, wj, key)
                checkpoint((self, {**mem, "theta": mem["theta"].at[w].set(theta)}, w))
                if s % cfg.check_every == 0 or s == cfg.steps:
                    loss = float(loss_d)
                    if VERBOSE:
                        print(f"  window {w} step {s}: loss {loss:.3e}", flush=True)
                    if not math.isfinite(loss) or loss < cfg.loss_tol:
                        break
            mem = {**mem, "theta": mem["theta"].at[w].set(theta)}
            self.log.append({"window": w, "steps": s, "loss": loss, "time_s": time.perf_counter() - t_w})
            print(f"window {w}: {s} steps, loss {loss:.3e}, {time.perf_counter() - t_w:.1f} s "
                  f"(total {time.perf_counter() - t_start:.1f} s)", flush=True)
        return self, mem, self.nw - 1

    def predict(self, mem, w_done):
        return self.predict_c(mem, jnp.int32(w_done))


# ----------------------------------------------------------------------------- the per-track solve.py API
def solve(problem, model: PINN, checkpoint):
    return model.train(checkpoint)


def query(problem, state):
    model, mem, w = state
    return model.predict(mem, w)
