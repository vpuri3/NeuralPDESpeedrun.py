"""Shallow PINNs trained by Levenberg-Marquardt, after Shahab, Mukhlash & Susanto, "Do physics-informed neural
networks (PINNs) need to be deep? Shallow PINNs using the Levenberg-Marquardt algorithm" (arXiv:2602.08515).
All key hyperparameters are in `Config` right below; each track's solve.py sets its own at its top. The choices and
their reasons are discussed in this directory's README.md.

- Network: tanh MLP, `layers` hidden layers of `width`, one output; Glorot-normal weights, zero biases. Inputs: a
  periodic axis as (sin, cos)(2 pi (x - lo) / L) (exact periodicity: the only condition built into the network),
  any other axis scaled to [-1, 1]. All parameters in one flat vector w.
- Residual F(w): the PDE residual at n_colloc uniform random interior points, plus one group of rows per initial
  or boundary condition (`Cond`: n_face random points on a face of the domain box, or given fixed points), each
  group scaled by weight * sqrt(n_colloc / its size), so ||F||^2 / n_colloc = mean(PDE^2) + sum_k w_k^2
  mean(cond_k^2). Every resample_every iterations all drawn points (interior and faces) are redrawn; the loss to
  beat is then recomputed on the new points, so "accept" always compares two losses on the same points.
- Derivatives: all by JAX AD. Input derivatives by Taylor mode (jax.experimental.jet); J = dF/dw by reverse mode
  per point (vmap(grad)).
- Levenberg-Marquardt: solve (J^T J + mu I) d = J^T F by Cholesky, no preconditioning or scaling; trial w - d.
  If ||F(w - d)||^2 < ||F(w)||^2 the step is accepted; otherwise mu <- mu_up * mu and the same iterate is retried.
  mu <- mu / mu_down only after mu_patience consecutive iterations whose FIRST trial was accepted (the textbook
  rule divides after every acceptance, which here made every other trial a rejection). Stops at max_iters (if
  set), when ||d|| < step_tol (step_tol + ||w||), when no mu <= mu_max lowers the loss, or at the harness budget.
- Execution: float64; everything compiled ahead of time in setup; the LM loop runs on the device in chunks of at
  most check_every iterations and about check_s seconds, the host checkpoints between chunks.
- Inspection (RunLog, always on): per iteration mu, loss, ||d||, gain ratio, trials, redraw flag; per chunk the
  wall time; on a schedule the residual probe and a weight snapshot. Written to
  run/tracks/PR0005/<track>_<name>_<datetime>/; inspect_run.py adds errors, spectra and the figure afterwards.
  Diagnostic mode (PR0005_DIAG=1, development): also every check_every iterations the singular values of J and
  the probe, with their cost kept off the training clock, which then stops the run at the track budget (run the
  harness with --no-impose-budget).
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # the harness also uses torch on this GPU

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
F64 = jnp.float64
VERBOSE = bool(os.environ.get("PR0005_VERBOSE"))


# ======================================= key hyperparameters (defaults) =======================================
@dataclass(frozen=True)
class Config:
    # network
    width: int = 32                     # neurons per hidden layer (largest used: 32)
    layers: int = 4                     # hidden layers (largest used: 4)
    # residual points
    n_colloc: int = 10_000              # interior collocation points (the paper's count)
    n_face: int = 500                   # points per initial / boundary condition face
    resample_every: int = 50            # LM iterations between fresh draws of all drawn points; 0: never (paper)
    # Levenberg-Marquardt
    mu0: float = 1e-2                   # initial damping (the paper / MATLAB fsolve)
    mu_up: float = 10.0                 # mu <- mu_up * mu after a rejected trial
    mu_down: float = 10.0               # mu <- mu / mu_down after mu_patience first-trial acceptances in a row
    mu_patience: int = 10
    mu_max: float = 1e16                # no acceptable step below this: stop
    step_tol: float = 1e-12             # stop when ||d|| < step_tol (step_tol + ||w||) (fsolve's StepTolerance)
    max_iters: int | None = None        # None: the harness's wall-clock budget stops the run
    # execution and inspection
    check_every: int = 50               # at most this many iterations per device call (checkpoint, diagnostics)
    check_s: float = 1.0                # ... and about this many seconds
    seed: int = 0
# ===============================================================================================================


@dataclass(frozen=True)
class Cond:
    """An initial or boundary condition imposed as residual rows fn(u, z) -> scalar, on n_face points drawn on the
    face {axis = value} of the domain (redrawn with the interior points), on fixed given `points` (rows may carry
    extra columns, e.g. a target value, after the coordinates), or on points from a `sample(key)` function."""
    name: str
    fn: Callable
    face: tuple | None = None           # (axis name, value)
    points: np.ndarray | None = None
    weight: float = 1.0
    sample: Callable | None = None      # or a sampler key -> (m, d) points, redrawn with the interior


def derivs(f: Callable, z, i: int, k: int):
    """(f, d f / d z_i, ..., d^k f / d z_i^k) at z, f: R^d -> R, by one Taylor-mode pass."""
    e = jnp.zeros_like(z).at[i].set(1.0)
    series = [e] + [jnp.zeros_like(z)] * (k - 1)
    y0, ys = jet.jet(f, (z,), (series,))
    return (y0, *ys)


def trig_interp(values, axes, domain):
    """JAX-traceable trigonometric interpolant of samples on a uniform periodic grid (the same interpolant as the
    tracks' periodic_interp, Re sum_k c_k e^{2 pi i k (x - lo) / L}), with its coefficients as numpy constants, so
    ahead-of-time compiled functions do not capture device arrays. values: (n0, n1, ...) on the grids `axes`;
    domain: per axis (lo, hi), the period hi - lo; a grid that includes hi duplicates its first sample, which is then
    dropped."""
    v = np.asarray(values, np.float64)
    for i, (x, (lo, hi)) in enumerate(zip(axes, domain)):
        if abs(float(x[-1]) - hi) < 1e-9 * (hi - lo):
            v = np.take(v, range(v.shape[i] - 1), axis=i)
    c = np.fft.fftn(v) / v.size
    ks = [np.fft.fftfreq(n, d=1.0 / n) for n in v.shape]
    los = [float(x[0]) for x in axes]          # the grid's first sample (cell-centred grids start at dx / 2)
    Ls = [float(hi - lo) for lo, hi in domain]  # the period

    def f(*xs):
        es = [jnp.exp(2j * jnp.pi * k * (x - lo) / L) for x, k, lo, L in zip(xs, ks, los, Ls)]
        out = jnp.asarray(c)
        for e in reversed(es):                 # contract the last axis first
            out = out @ e
        return jnp.real(out)

    return f


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
    out = h @ W + b
    return out[0] if out.shape[0] == 1 else out


@dataclass(frozen=True)
class State:
    model: "LMPINN"
    w: jax.Array
    mu: float
    sse: float
    it: int


# per-iteration log columns written by the device loop
ITER_COLS = ("loss", "mu", "step", "rho", "trials", "redrawn")


class LMPINN:
    """problem: the track's Problem. residual(u, z) -> PDE residual (scalar or vector), u: z -> network output
    (scalar if n_out == 1, else (n_out,): the problem's fields first, then free fields such as pressure; use
    `derivs`), z the global coordinates in the order of problem.axes. conds: the initial / boundary conditions
    (`Cond`; fn may also return a vector). Vector residuals become stacked rows; each group still counts as a mean.
    interior: optional interior points used instead of n_colloc uniform random ones: fixed points (m, d) (never
    redrawn), or a sampler key -> (m, d) (redrawn every resample_every iterations, e.g. stratified)."""

    def __init__(self, problem, residual: Callable, cfg: Config, conds=(), n_out: int = 1, interior=None):
        env = os.environ                        # development overrides (sweeps); never set in scored runs
        if "PR0005_ARCH" in env:
            w, l = map(int, env["PR0005_ARCH"].split("x"))
            cfg = replace(cfg, width=w, layers=l)
        if "PR0005_MAX_ITERS" in env:
            cfg = replace(cfg, max_iters=None if env["PR0005_MAX_ITERS"].lower() == "none" else int(env["PR0005_MAX_ITERS"]))
        if interior is not None:
            if not callable(interior):
                interior = np.atleast_2d(np.asarray(interior, np.float64))
            m = (interior(jax.random.PRNGKey(0)) if callable(interior) else interior).shape[0]
            cfg = replace(cfg, n_colloc=int(m))
        self.interior = interior
        self.cfg, self.problem, self.residual, self.conds = cfg, problem, residual, tuple(conds)
        self.diag = bool(env.get("PR0005_DIAG"))
        names = problem.axis_names
        self.names = names
        self.lo = np.array([float(problem.domain[k][0]) for k in names])
        self.hi = np.array([float(problem.domain[k][1]) for k in names])
        self.per = [problem.is_periodic(k) for k in names]
        self.n_out, self.n_fields = n_out, len(problem.fields)
        sizes = [sum(2 if p else 1 for p in self.per)] + [cfg.width] * cfg.layers + [n_out]
        self.w0, self.unravel = ravel_pytree(_init(jax.random.PRNGKey(cfg.seed), sizes))
        self.n_params = int(self.w0.size)
        self.sizes = [cfg.n_colloc] + [cfg.n_face if c.face else
                                       (c.sample(jax.random.PRNGKey(0)).shape[0] if c.sample else len(c.points))
                                       for c in self.conds]
        self.scale = [1.0] + [c.weight * math.sqrt(cfg.n_colloc / m) for c, m in zip(self.conds, self.sizes[1:])]
        self.group_names = [c.name for c in self.conds]
        self.fns = [residual] + [c.fn for c in self.conds]
        self.Zs0 = self.draw(jax.random.PRNGKey(cfg.seed + 1))
        self.k = [math.prod(jax.eval_shape(lambda z: fn(self.u(self.w0), z), Z[0]).shape)   # rows per point
                  for Z, fn in zip(self.Zs0, self.fns)]
        self.n_rows = sum(m * k for m, k in zip(self.sizes, self.k))
        lo_, hi_ = jnp.asarray(self.lo), jnp.asarray(self.hi)               # held-out interior points (probe only):
        self.Zheld = lo_ + (hi_ - lo_) * jax.random.uniform(jax.random.PRNGKey(12345),  # uniform, independent of
                                                            (cfg.n_colloc, len(self.lo)), F64)  # the training ones
        self.axes = [np.asarray(problem.axes[k].detach().cpu().numpy(), np.float64) for k in names]

    # points ----------------------------------------------------------------------------------------------
    def draw(self, key):
        """All residual points: (interior, condition groups...). Faces and the interior are uniform random."""
        keys = jax.random.split(key, 1 + len(self.conds))
        lo, hi = jnp.asarray(self.lo), jnp.asarray(self.hi)
        if self.interior is None:
            Zs = [lo + (hi - lo) * jax.random.uniform(keys[0], (self.cfg.n_colloc, len(self.lo)), F64)]
        else:
            Zs = [self.interior(keys[0]) if callable(self.interior) else jnp.asarray(self.interior)]
        for k, c in zip(keys[1:], self.conds):
            if c.sample:
                Zs.append(c.sample(k))
            elif c.face:
                i = self.names.index(c.face[0])
                Z = lo + (hi - lo) * jax.random.uniform(k, (self.cfg.n_face, len(self.lo)), F64)
                Zs.append(Z.at[:, i].set(float(c.face[1])))
            else:
                Zs.append(jnp.asarray(np.atleast_2d(np.asarray(c.points, np.float64))))
        return tuple(Zs)

    # network ---------------------------------------------------------------------------------------------
    def features(self, z):
        out = []
        for i, p in enumerate(self.per):
            s = (z[i] - self.lo[i]) / (self.hi[i] - self.lo[i])
            out += [jnp.sin(2 * jnp.pi * s), jnp.cos(2 * jnp.pi * s)] if p else [2 * s - 1]
        return jnp.stack(out)

    def u(self, w):
        params = self.unravel(w)
        return lambda z: _mlp(params, self.features(z))

    def F(self, w, Zs):
        return jnp.concatenate([jax.vmap(lambda z: c * fn(self.u(w), z))(Z).reshape(-1)
                                for Z, fn, c in zip(Zs, self.fns, self.scale)])

    def J(self, w, Zs):
        return jnp.concatenate([jax.vmap(jax.jacrev(lambda w, z: c * fn(self.u(w), z)), (None, 0))(w, Z)
                                .reshape(-1, w.size) for Z, fn, c in zip(Zs, self.fns, self.scale)])

    # LM ----------------------------------------------------------------------------------------------------
    def _chunk(self, w, mu, sse, key, Zs, streak, it0, n):
        """Up to n LM iterations on the device, starting at global iteration it0. Returns the new state and the
        per-iteration log (n_max rows, ITER_COLS)."""
        cfg = self.cfg
        I = jnp.eye(w.size, dtype=F64)
        nan = jnp.array(jnp.nan, F64)

        def iteration(c):
            w, mu, sse, key, Zs, streak, stop, k, log = c
            redraw = (cfg.resample_every > 0) & ((it0 + k) % max(cfg.resample_every, 1) == 0) & (it0 + k > 0)

            def fresh(a):
                key, Zs, sse = a
                key, sub = jax.random.split(key)
                Zs = self.draw(sub)
                return key, Zs, jnp.sum(self.F(w, Zs) ** 2)

            key, Zs, sse = jax.lax.cond(redraw, fresh, lambda a: a, (key, Zs, sse))
            r, J = self.F(w, Zs), self.J(w, Zs)
            A, g = J.T @ J, J.T @ r

            def trial(t):
                mu_t, _, _, _, _, _, ntr = t
                L = jnp.linalg.cholesky(A + mu_t * I)
                d = jax.scipy.linalg.cho_solve((L, True), g)
                w2 = w - d
                s2 = jnp.sum(self.F(w2, Zs) ** 2)
                ok = s2 < sse                  # NaN (failed factorization) counts as a rejection
                pred = 2 * d @ g - d @ (A @ d)  # decrease of ||F||^2 predicted by the linear model
                return jnp.where(ok, mu_t, mu_t * cfg.mu_up), w2, s2, ok, jnp.linalg.norm(d), pred, ntr + 1

            z = jnp.array(0.0, F64)
            mu, w2, s2, ok, dn, pred, ntr = jax.lax.while_loop(lambda t: (~t[3]) & (t[0] <= cfg.mu_max), trial,
                                                               (mu, w, sse, False, z, z, jnp.array(0)))
            streak = jnp.where(ok & (ntr == 1), streak + 1, 0)
            down = streak >= cfg.mu_patience
            mu = jnp.where(down, mu / cfg.mu_down, mu)
            streak = jnp.where(down, 0, streak)
            row = jnp.stack([jnp.where(ok, s2, sse) / cfg.n_colloc, mu, jnp.where(ok, dn, nan),
                             jnp.where(ok, (sse - s2) / pred, nan), ntr.astype(F64), redraw.astype(F64)])
            log = log.at[k].set(row)
            w = jnp.where(ok, w2, w)
            sse = jnp.where(ok, s2, sse)
            small = ok & (dn < cfg.step_tol * (cfg.step_tol + jnp.linalg.norm(w)))
            stop = jnp.where(~ok, 2, jnp.where(small, 1, 0))     # 1: step tolerance, 2: mu > mu_max
            return w, mu, sse, key, Zs, streak, stop, k + 1, log

        log0 = jnp.full((cfg.check_every, len(ITER_COLS)), jnp.nan, F64)
        w, mu, sse, key, Zs, streak, stop, k, log = jax.lax.while_loop(
            lambda c: (c[6] == 0) & (c[7] < n), iteration,
            (w, mu, sse, key, Zs, streak, jnp.array(0), jnp.array(0), log0))
        return w, mu, sse, key, Zs, streak, stop, k, log

    def _probe(self, w, Zs):
        """Per residual group (PDE on the current training points, PDE on fixed held-out points, each condition on
        its current points): RMS and max |r| of the unscaled residual; then ||J^T F|| (current points), ||w||."""
        u = self.u(w)
        out = []
        for Z, fn in [(Zs[0], self.fns[0]), (self.Zheld, self.fns[0])] + list(zip(Zs[1:], self.fns[1:])):
            r = jax.vmap(lambda z: fn(u, z))(Z).reshape(-1)
            out += [jnp.sqrt(jnp.mean(r ** 2)), jnp.max(jnp.abs(r))]
        grad = jax.grad(lambda w: 0.5 * jnp.sum(self.F(w, Zs) ** 2))(w)
        return jnp.stack(out + [jnp.linalg.norm(grad), jnp.linalg.norm(w)])

    def compile(self):
        s = jax.ShapeDtypeStruct
        sw, sc, si = s((self.n_params,), F64), s((), F64), s((), jnp.int32)
        sZ = tuple(s(Z.shape, F64) for Z in self.Zs0)
        self._chunk_c = jax.jit(self._chunk).lower(sw, sc, sc, s((2,), jnp.uint32), sZ, si, si, si).compile()
        self._probe_c = jax.jit(self._probe).lower(sw, sZ).compile()
        self._sse_c = jax.jit(lambda w, Zs: jnp.sum(self.F(w, Zs) ** 2)).lower(sw, sZ).compile()
        if self.diag:
            self._svd_c = jax.jit(lambda w, Zs: jnp.linalg.svd(self.J(w, Zs), compute_uv=False)).lower(sw, sZ).compile()
        grid = jnp.asarray(np.stack([g.ravel() for g in np.meshgrid(*self.axes, indexing="ij")], -1))
        shape = (self.n_fields, *[len(a) for a in self.axes])

        def pred(w):
            U = jax.vmap(self.u(w))(grid)
            U = U[:, None] if U.ndim == 1 else U
            return U[:, :self.n_fields].T.reshape(shape)

        self._pred_c = jax.jit(pred).lower(sw).compile()
        return self

    def train(self, checkpoint):
        cfg = self.cfg
        w, mu, Zs = self.w0, jnp.asarray(cfg.mu0, F64), self.Zs0
        sse = self._sse_c(w, Zs)
        state = State(self, w, float(mu), float(sse), 0)
        checkpoint(state)
        log = RunLog(self)
        try:
            out = self._loop(checkpoint, w, mu, sse, Zs, state, log)
            if log.stop is None:
                log.stop = f"{cfg.max_iters}-iteration cap"
            return out
        except BaseException as e:             # the harness's BudgetExceeded (a BaseException) included
            log.stop = "budget" if type(e).__name__ == "BudgetExceeded" else type(e).__name__
            raise
        finally:
            log.save()                         # in a background thread: no disk I/O on the clock

    def _loop(self, checkpoint, w, mu, sse, Zs, state, log):
        cfg = self.cfg
        key, streak = jax.random.PRNGKey(cfg.seed + 2), jnp.int32(0)
        it, n = 0, 1                           # n: iterations per call, grown toward check_s seconds
        while cfg.max_iters is None or it < cfg.max_iters:
            if self.diag:
                n = cfg.check_every            # diagnostics exactly every check_every iterations
            if cfg.max_iters is not None:
                n = min(n, cfg.max_iters - it)
            tc = time.perf_counter()
            w, mu, sse, key, Zs, streak, stop, k, rows = self._chunk_c(w, mu, sse, key, Zs, streak, jnp.int32(it),
                                                                       jnp.int32(n))
            k = int(k)
            it += k
            dt = time.perf_counter() - tc
            n = max(1, min(cfg.check_every, int(n * cfg.check_s / max(dt, 1e-9)), 2 * n))
            state = State(self, w, float(mu), float(sse), it)
            log.record(it, np.asarray(rows)[:k], w, Zs)
            if self.diag:
                log.diagnose(it, w, Zs, float(mu))
            checkpoint(state)
            if VERBOSE:
                print(f"  it {it:6d}  loss {float(sse) / cfg.n_colloc:.3e}  mu {float(mu):.1e}  {log.clock():7.1f} s",
                      flush=True)
            if self.diag and log.clock() >= self.problem.budget_s:
                log.stop = "budget (training clock; diagnostics excluded)"
                break
            if int(stop):
                log.stop = "step tolerance" if int(stop) == 1 else "mu > mu_max"
                break
        return state

    def predict(self, state: State):
        return self._pred_c(state.w)


REPO = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO / "run" / "tracks" / "PR0005"


class RunLog:
    """Inspection record of one run, written to run/tracks/PR0005/<track>_<name>_<datetime>/ (history.npz,
    config.json). Per iteration: ITER_COLS. Per chunk: wall time and iteration count. On a schedule (log-spaced
    early, then about once per second): the residual probe and a weight snapshot (inspect_run.py turns those into
    errors against the reference). Diagnostic mode, every chunk: singular values of J (max, min, numerical rank,
    number above sqrt(mu); full spectra at log-spaced times) and the probe, off the training clock.
    name: PR0005_NAME, else <width>x<layers>[_diag]. Only the first run in a process is written (the harness's
    device check reruns setup + solve in the same process)."""

    def __init__(self, model):
        self.m, self.t0, self.stop, self.excluded = model, time.perf_counter(), None, 0.0
        self.iters, self.chunk = [], {"t": [], "it": []}
        self.p = {k: [] for k in ("t", "it", "probe", "w")}
        self.d = {k: [] for k in ("t", "it", "mu", "probe", "sv_max", "sv_min", "rank", "n_active", "spec_t",
                                  "spec_it", "spec")}
        self.next_probe = self.next_spec = 0.0

    def clock(self):
        return time.perf_counter() - self.t0 - self.excluded

    def record(self, it, rows, w, Zs):
        t = self.clock()
        self.iters.append(rows)
        self.chunk["t"].append(t); self.chunk["it"].append(it)
        if t >= self.next_probe:
            tc = time.perf_counter()
            for k, v in zip(("t", "it", "probe", "w"), (t, it, np.asarray(self.m._probe_c(w, Zs)), np.asarray(w))):
                self.p[k].append(v)
            self.next_probe = t + min(1.0, max(0.02, 0.15 * t))
            if self.m.diag:
                self.excluded += time.perf_counter() - tc

    def diagnose(self, it, w, Zs, mu):
        tc, t = time.perf_counter(), self.clock()
        sv = np.asarray(self.m._svd_c(w, Zs))
        probe = np.asarray(self.m._probe_c(w, Zs))
        rank = int((sv > sv[0] * max(self.m.n_rows, self.m.n_params) * np.finfo(np.float64).eps).sum())
        for k, v in zip(("t", "it", "mu", "probe", "sv_max", "sv_min", "rank", "n_active"),
                        (t, it, mu, probe, sv[0], sv[-1], rank, int((sv > math.sqrt(mu)).sum()))):
            self.d[k].append(v)
        if t >= self.next_spec:                # full spectra at log-spaced times
            self.d["spec_t"].append(t); self.d["spec_it"].append(it); self.d["spec"].append(sv)
            self.next_spec = max(t * 1.15, t + 0.05)
        self.excluded += time.perf_counter() - tc

    def save(self):
        if os.environ.get("PR0005_RUN_DIR_WRITTEN"):
            return
        m, cfg = self.m, self.m.cfg
        name = os.environ.get("PR0005_NAME") or f"{cfg.width}x{cfg.layers}" + ("_diag" if m.diag else "")
        d = RUN_ROOT / f"{m.problem.track}_{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        os.environ["PR0005_RUN_DIR_WRITTEN"] = str(d)
        print(f"PR0005 run dir: {d}", flush=True)
        conf = {"track": m.problem.track, "name": name, "arch": f"{cfg.width}x{cfg.layers}", "n_params": m.n_params,
                "rows": m.n_rows, "config": asdict(cfg),
                "groups": ["PDE (training points)", "PDE (held-out points)"] + m.group_names,
                "stop": self.stop, "solve_time_s": self.clock(), "diagnostic_mode": m.diag,
                "diagnostics_time_s": self.excluded, "iter_cols": list(ITER_COLS)}
        rows = np.concatenate(self.iters) if self.iters else np.zeros((0, len(ITER_COLS)))
        arrs = {"iter": rows, "chunk_t": np.asarray(self.chunk["t"]), "chunk_it": np.asarray(self.chunk["it"])}
        arrs |= {f"probe_{k}": np.asarray(v) for k, v in self.p.items()}
        if m.diag:
            arrs |= {f"diag_{k}": np.asarray(v) for k, v in self.d.items()}

        def write():
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / "history.npz", **arrs)
            (d / "config.json").write_text(json.dumps(conf, indent=1))

        threading.Thread(target=write).start()


def solve(problem, model: LMPINN, checkpoint):
    return model.train(checkpoint)


def query(problem, state):
    return state.model.predict(state)
