"""Submission 1: PirateNet + SOAP physics-informed training (Wang et al., "Gradient Alignment in Physics-informed
Neural Networks: A Second-Order Optimization Perspective"; PirateNets, Wang et al. 2024), in JAX, following the
authors' reference implementation (github.com/PredictiveIntelligenceLab/jaxpi, branch `pirate`):

- PirateNet as in `jaxpi.archs`: per-point network on raw coordinates; period embeddings (cos, sin)(w x) of
  periodic inputs, w = 2 pi / L; trainable random Fourier features; weight-factorized dense layers
  (W = g * V, log g ~ N(1, 0.1)); gated residual blocks whose skip weights alpha start at 0 (identity at init).
- SOAP from `soap_jax` (github.com/haydn-jones/SOAP_JAX, the implementation jaxpi uses) with jaxpi's settings:
  b1 = 0.9, b2 = 0.999, precondition_frequency = 2, no weight decay, eps = 1e-8. Learning rate: linear warmup
  then exponential decay (optax), as `jaxpi.models._create_optimizer`.
- Grad-norm loss weights (mean grad norm / grad norm of each loss), running average with momentum 0.9,
  refreshed every 1000 iterations starting at iteration 0, as `jaxpi.models.PINN.compute_weights`.
- Causal weights for evolution problems: residuals sorted in time, split into chunks, chunk i weighted by
  exp(-tol * sum_{j<i} L_j), as the jaxpi examples' `res_and_w`.
- Physics-informed initialization: the last layer is the least-squares fit of the initial condition (tiled
  over time) on the network's features at initialization, as the jaxpi examples' `use_pi_init`.

Speedrun specifics: one time window (the jaxpi runs time-march over several windows of 1e4-1e5 iterations each,
far beyond the budget); the harness calls `step`, which runs CHUNK training iterations in one jitted
lax.fori_loop; `done` once the loss plateaus or becomes negligible; the wall-clock budget is the harness's.
Everything is compiled ahead of time in `setup` (lower + compile, nothing is executed there); the
physics-informed initialization runs in the first (timed) step.

Each track's solve.py defines its losses on top of `PINN`.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # the harness also uses torch on this GPU

import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Callable, Sequence  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import linen as nn  # noqa: E402
from jax.nn.initializers import constant, glorot_normal, normal, zeros  # noqa: E402
from soap_jax import soap  # noqa: E402

CHUNK = 100          # training iterations per harness step


# ----------------------------------------------------------------------------- network (jaxpi.archs)
def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = jax.random.split(key)
        w = init_fn(key1, shape)
        g = jnp.exp(mean + normal(stddev)(key2, (shape[-1],)))
        return g, w / g

    return init


class Dense(nn.Module):
    features: int
    reparam: bool = True

    @nn.compact
    def __call__(self, x):
        if self.reparam:
            g, v = self.param("kernel", _weight_fact(glorot_normal(), mean=1.0, stddev=0.1),
                              (x.shape[-1], self.features))
            kernel = g * v
        else:
            kernel = self.param("kernel", glorot_normal(), (x.shape[-1], self.features))
        bias = self.param("bias", zeros, (self.features,))
        return jnp.dot(x, kernel) + bias


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    nonlinearity: float = 0.0

    @nn.compact
    def __call__(self, x, u, v):
        identity = x
        x = jnp.tanh(Dense(self.hidden_dim)(x))
        x = x * u + (1 - x) * v
        x = jnp.tanh(Dense(self.hidden_dim)(x))
        x = x * u + (1 - x) * v
        x = jnp.tanh(Dense(self.output_dim)(x))
        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        return alpha * x + (1 - alpha) * identity


class PirateNet(nn.Module):
    out_dim: int
    periods: tuple = ()             # ((input index, w), ...): (cos, sin)(w x_i) replaces x_i
    num_layers: int = 3
    hidden_dim: int = 256
    embed_scale: float = 1.0
    embed_dim: int = 256
    pi_init: bool = False           # last layer = a plain kernel, set by least squares on the IC

    @nn.compact
    def __call__(self, z):
        w = dict(self.periods)
        feats = []
        for i in range(z.shape[-1]):
            if i in w:
                feats += [jnp.cos(w[i] * z[i]), jnp.sin(w[i] * z[i])]
            else:
                feats.append(z[i])
        x = jnp.stack(feats)
        B = self.param("fourier_kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2))
        x = jnp.concatenate([jnp.cos(x @ B), jnp.sin(x @ B)])
        u = jnp.tanh(Dense(self.hidden_dim)(x))
        v = jnp.tanh(Dense(self.hidden_dim)(x))
        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(self.hidden_dim, x.shape[-1])(x, u, v)
        if self.pi_init:
            y = x @ self.param("pi_init", zeros, (x.shape[-1], self.out_dim))
        else:
            y = Dense(self.out_dim)(x)
        return x, y


# ----------------------------------------------------------------------------- configuration
@dataclass
class Config:
    num_layers: int = 3
    hidden_dim: int = 256
    embed_scale: float = 1.0
    lr: float = 1e-3
    warmup_steps: int = 5000
    decay_steps: int = 2000
    decay_rate: float = 0.9
    batch: int = 4096
    init_weights: dict = field(default_factory=dict)   # initial loss weights (default 1)
    weight_every: int = 1000
    momentum: float = 0.9
    causal_tol: float = 1.0
    num_chunks: int = 32
    pi_init: bool = False
    pi_init_times: int = 20       # IC tiled over this many times for the least-squares fit
    pi_init_points: int = 40_000  # at most this many rows in it
    plateau_s: float = 150.0      # stop when the loss has not improved by 2% for max(plateau_s, elapsed / 4)


def schedule(cfg: Config):
    decay = optax.exponential_decay(cfg.lr, cfg.decay_steps, cfg.decay_rate)
    if cfg.warmup_steps <= 0:
        return decay
    warm = optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
    return optax.join_schedules([warm, decay], [cfg.warmup_steps])


# ----------------------------------------------------------------------------- derivatives of per-point fns
def d(f: Callable, *argnums: int) -> Callable:
    """Mixed partial of a scalar per-point function f(params, *coords) in the coordinates `argnums` (one
    entry per derivative, e.g. d(u, 2, 2) = u_xx when x is argument 2)."""
    for a in argnums:
        f = jax.grad(f, argnums=a)
    return f


def jet(m, params, *coords):
    """All outputs Y (k,), their gradients J (k, d) and Hessians H (k, d, d) in the inputs at one point, in one
    forward-over-reverse pass."""
    z = jnp.stack(coords)
    f = lambda z: m.net.apply(params, z)[1]  # noqa: E731

    def g(z):
        Y, vjp = jax.vjp(f, z)
        J = jax.vmap(vjp)(jnp.eye(Y.shape[0], dtype=Y.dtype))[0]
        return J, (J, Y)

    H, (J, Y) = jax.jacfwd(g, has_aux=True)(z)
    return Y, J, H


def to_jax(a, dtype=jnp.float32):
    """A torch tensor (problem axes, sampled data) as a JAX array."""
    return jnp.asarray(np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a), dtype=dtype)


# ----------------------------------------------------------------------------- solver
class PINN:
    """inputs: problem axes fed to the network, in order (e.g. ("t", "x")); out_dim: network outputs, the
    first len(problem.fields) of which are the scored fields (extra ones, e.g. pressure, follow).

    losses(m, params, key) -> {name: scalar} defines the training problem; `m` is this object (its `u`,
    `sample`, `causal`, ... helpers). ic_data(m) -> (Z (n, d), U (n, k)): initial-condition samples for the
    physics-informed initialization (k = number of fitted outputs; the others are fitted to 0)."""

    def __init__(self, problem, inputs: Sequence[str], out_dim: int, losses: Callable, cfg: Config,
                 ic_data: Callable | None = None, fields: Sequence[int] | None = None):
        self.problem, self.inputs, self.out_dim, self.cfg = problem, tuple(inputs), out_dim, cfg
        self.fields = tuple(fields) if fields is not None else tuple(range(len(problem.fields)))
        self.lo = [float(problem.domain[k][0]) for k in self.inputs]
        self.hi = [float(problem.domain[k][1]) for k in self.inputs]
        periods = tuple((i, 2 * math.pi / (self.hi[i] - self.lo[i]))
                        for i, k in enumerate(self.inputs) if problem.is_periodic(k))
        self.net = PirateNet(out_dim, periods, cfg.num_layers, cfg.hidden_dim, cfg.embed_scale, cfg.hidden_dim,
                             pi_init=cfg.pi_init and ic_data is not None)
        self.ic_data = ic_data
        self.loss_fn = lambda params, key: losses(self, params, key)
        self.tx = soap(learning_rate=schedule(cfg), b1=0.9, b2=0.999, weight_decay=0.0, precondition_frequency=2)
        self.M = jnp.triu(jnp.ones((cfg.num_chunks, cfg.num_chunks)), k=1).T
        self.axes = [to_jax(problem.axes[k]) for k in problem.axes]
        self.history: list[tuple[float, int, float]] = []
        self.stop_reason = ""

    # --- per-point network and helpers used by the tracks' losses
    def apply(self, params, *coords):
        return self.net.apply(params, jnp.stack(coords))[1]

    def u(self, k: int) -> Callable:
        """Scalar per-point function (params, *coords) -> output k."""
        return lambda params, *coords: self.apply(params, *coords)[k]

    def sample(self, key, n: int, **fixed):
        """n uniform points of the box as a tuple of coordinate arrays (one per input); fixed={"t": t0} pins a
        coordinate."""
        u = jax.random.uniform(key, (n, len(self.inputs)))
        cols = []
        for i, k in enumerate(self.inputs):
            cols.append(jnp.full((n,), fixed[k]) if k in fixed else self.lo[i] + (self.hi[i] - self.lo[i]) * u[:, i])
        return tuple(cols)

    def faces(self, key, n: int, walls: Sequence[str]):
        """n uniform points on the faces of the box normal to the inputs `walls` (other coordinates uniform)."""
        k1, k2, k3 = jax.random.split(key, 3)
        cols = list(self.sample(k1, n))
        which = jax.random.randint(k2, (n,), 0, len(walls))
        side = jax.random.randint(k3, (n,), 0, 2)
        for j, k in enumerate(walls):
            i = self.inputs.index(k)
            cols[i] = jnp.where(which == j, self.lo[i] + side * (self.hi[i] - self.lo[i]), cols[i])
        return tuple(cols)

    def grid_sample(self, key, n: int, axes: Sequence, values: Sequence):
        """n random nodes of a tensor grid (JAX axes): (coordinate arrays, values at those nodes)."""
        shape = [a.shape[0] for a in axes]
        flat = jax.random.randint(key, (n,), 0, math.prod(shape))
        idx = jnp.unravel_index(flat, shape)
        return tuple(a[i] for a, i in zip(axes, idx)), tuple(v.reshape(-1)[flat] for v in values)

    def ic_tiled(self, spatial: Sequence, values):
        """IC samples for the physics-informed initialization, as the jaxpi examples build them: the initial
        state at the spatial points `spatial` (one array per spatial input), tiled over pi_init_times times
        (inputs[0] must be "t"); values (n, k). At most pi_init_points rows (uniform stride)."""
        T = jnp.linspace(self.lo[0], self.hi[0], self.cfg.pi_init_times)
        n = spatial[0].shape[0]
        Z = jnp.concatenate([jnp.stack([jnp.full((n,), t), *spatial], 1) for t in T])
        U = jnp.tile(values, (T.shape[0], 1))
        stride = max(1, Z.shape[0] // self.cfg.pi_init_points)
        return Z[::stride], U[::stride]

    def causal(self, r2: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Causally weighted mean of the squared residuals r2 (n,) at times t (n,)."""
        order = jnp.argsort(t)
        lc = jnp.mean(r2[order].reshape(self.cfg.num_chunks, -1), axis=1)
        w = jax.lax.stop_gradient(jnp.exp(-self.cfg.causal_tol * (self.M @ lc)))
        return jnp.mean(lc * w)

    # --- training
    def _total(self, params, weights, key):
        L = self.loss_fn(params, key)
        return sum(weights[k] * L[k] for k in L), L

    def _chunk(self, params, opt_state, weights, key):
        def body(i, carry):
            params, opt_state, key = carry
            key, sub = jax.random.split(key)
            g = jax.grad(lambda p: self._total(p, weights, sub)[0])(params)
            upd, opt_state = self.tx.update(g, opt_state, params)
            return optax.apply_updates(params, upd), opt_state, key

        params, opt_state, key = jax.lax.fori_loop(0, CHUNK, body, (params, opt_state, key))
        key, sub = jax.random.split(key)
        raw = sum(self.loss_fn(params, sub).values())
        return params, opt_state, key, raw

    def _new_weights(self, params, weights, key):
        """Grad-norm weighting with momentum (jaxpi.models.PINN.compute_weights + TrainState.apply_weights)."""
        grads = jax.jacrev(self.loss_fn)(params, key)
        norms = {k: jnp.linalg.norm(jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(g)]))
                 for k, g in grads.items()}
        mean = jnp.mean(jnp.stack(list(norms.values())))
        m = self.cfg.momentum
        return {k: m * weights[k] + (1 - m) * mean / (norms[k] + 1e-5 * mean) for k in norms}

    def _pi_init(self, params):
        """Least-squares fit of the last layer to the IC on the features at initialization."""
        Z, U = self.ic_data(self)
        feats = jax.vmap(lambda z: self.net.apply(params, z)[0])(Z)
        U = jnp.concatenate([U, jnp.zeros((U.shape[0], self.out_dim - U.shape[1]))], axis=1)
        coef = jnp.linalg.lstsq(feats, U)[0]
        return {**params, "params": {**params["params"], "pi_init": coef}}

    def _predict(self, params, start):
        """Network outputs at flat grid indices [start, start + QCHUNK) of the evaluation grid (clipped)."""
        shape = [a.shape[0] for a in self.axes]
        flat = jnp.minimum(start + jnp.arange(self._qchunk), math.prod(shape) - 1)
        idx = jnp.unravel_index(flat, shape)
        names = list(self.problem.axes)
        coords = [self.axes[names.index(k)][idx[names.index(k)]] for k in self.inputs]
        Y = jax.vmap(lambda *c: self.apply(params, *c))(*coords)
        return Y[:, list(self.fields)]

    def compile(self):
        """Untimed: initialize parameters and optimizer state, compile every jitted function (no execution)."""
        key = jax.random.PRNGKey(0)
        key, kinit = jax.random.split(key)
        self.params = self.net.init(kinit, jnp.ones(len(self.inputs)))
        self.opt_state = self.tx.init(self.params)
        self.weights = {k: jnp.float32(self.cfg.init_weights.get(k, 1.0))
                        for k in jax.eval_shape(self.loss_fn, self.params, key)}
        self.key = key
        n_out = math.prod(a.shape[0] for a in self.axes)
        self._qchunk = min(1 << 16, n_out)
        aval = lambda tree: jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), tree)  # noqa: E731
        P, S, W, K = aval(self.params), aval(self.opt_state), aval(self.weights), aval(self.key)
        self.chunk_c = jax.jit(self._chunk, donate_argnums=(0, 1)).lower(P, S, W, K).compile()
        self.weights_c = jax.jit(self._new_weights).lower(P, W, K).compile()
        self.predict_c = jax.jit(self._predict).lower(P, jax.ShapeDtypeStruct((), jnp.int32)).compile()
        self.pi_init_c = (jax.jit(self._pi_init).lower(P).compile()
                          if self.net.pi_init else None)
        self.iters = 0
        return self

    def step(self) -> bool:
        """One harness step: CHUNK training iterations. Returns done."""
        if self.iters == 0:
            self.t0 = time.perf_counter()
            self.best, self.best_t = float("inf"), 0.0
            if self.pi_init_c is not None:
                self.params = self.pi_init_c(self.params)
        if self.iters % self.cfg.weight_every == 0:
            self.key, sub = jax.random.split(self.key)
            self.weights = self.weights_c(self.params, self.weights, sub)
        self.params, self.opt_state, self.key, raw = self.chunk_c(self.params, self.opt_state, self.weights,
                                                                   self.key)
        self.iters += CHUNK
        raw, el = float(raw), time.perf_counter() - self.t0
        self.history.append((el, self.iters, raw))
        if not math.isfinite(raw):
            self.stop_reason = "non-finite loss"
            return True
        if raw < 0.98 * self.best:
            self.best, self.best_t = raw, el
        if raw < 1e-10:
            self.stop_reason = "loss < 1e-10"
            return True
        if el - self.best_t > max(self.cfg.plateau_s, 0.25 * el) and el > 2 * self.cfg.plateau_s:
            self.stop_reason = f"plateau (no 2% improvement in {el - self.best_t:.0f} s)"
            return True
        return False

    def predict(self):
        """The scored fields on the evaluation grid, shape problem.out_shape."""
        shape = [a.shape[0] for a in self.axes]
        n = math.prod(shape)
        out = [self.predict_c(self.params, jnp.int32(s)) for s in range(0, n, self._qchunk)]
        Y = jnp.concatenate(out)[:n]
        return Y.T.reshape(len(self.fields), *shape)


# ----------------------------------------------------------------------------- the per-track solve.py API
def step(problem, state: PINN):
    return state, state.step()


def query(problem, state: PINN):
    return state.predict()
