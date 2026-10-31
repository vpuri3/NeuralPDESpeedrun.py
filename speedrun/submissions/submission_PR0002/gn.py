"""Gauss-Newton (natural-gradient) step on a stacked least-squares residual, after Jnini & Vella, "Dual Natural
Gradient Descent for Scalable Training of Physics-Informed Neural Networks" (TMLR 2025; code: JAX-NG,
github.com/HicrestLaboratory/JAX-NG, and the paper's supplementary material).

The loss is L(theta) = 1/2 |r(theta)|^2, r in R^m stacked from residual terms (PDE interior, boundary, ...),
each scaled by sqrt(weight / N) so that L = 1/2 sum_i weight_i mean(r_i^2). With J = dr/dtheta (m x n), the
damped Gauss-Newton direction is

    primal:  d = (J^T J + lam)^{-1} J^T r            (n x n solve)
    dual:    d = J^T (J J^T + lam)^{-1} r            (m x m solve; the same step, the paper's D-NGD)

both Jacobi-scaled (the normal matrix symmetrically scaled to unit diagonal before damping, as JAX-NG), damped
by a fixed lam or, with lm_max > 0, by the Levenberg-Marquardt rule lam = min(loss, lm_max) of the paper's code
(floored at `damping`), solved by Cholesky, and picked by shape ("auto": dual when n > m). Optional geodesic acceleration (the paper's
correction): a = -J^+ f_vv, f_vv = d^2/ds^2 r(theta + s v) at s = 0 along the GN velocity v = -d, solved with the
same factorization; used when 2|a|/|v| <= geo_ratio. The step length comes from a grid line search over
eta in ls_steps: theta + eta v + eta^2/2 a (non-finite trial losses count as +inf).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GNConfig:
    damping: float = 1e-8                    # added to the unit diagonal of the Jacobi-scaled normal matrix
    lm_max: float = 0.0                      # > 0: Levenberg-Marquardt damping clip(loss, damping, lm_max)
    mode: str = "auto"                       # "auto" | "primal" | "dual"
    ls_steps: tuple = tuple(0.5 ** k for k in range(10))
    geodesic: bool = False                   # geodesic acceleration
    geo_ratio: float = 0.5                   # accept the acceleration only if 2 |a| / |v| <= geo_ratio


@dataclass(frozen=True)
class Term:
    """One residual class: fn(theta, z) -> residual at point z, a scalar or a vector (k,) (k rows per point);
    points (N, d); loss weight (the term contributes weight * mean over points of |r|^2 / 2)."""
    fn: Callable
    points: jnp.ndarray
    weight: float = 1.0


def _vec(fn):
    return lambda th, z: jnp.atleast_1d(fn(th, z))


def _stacked(terms: Sequence[Term], theta):
    """r(theta) (m,), all terms, scaled."""
    return jnp.concatenate([jnp.sqrt(t.weight / t.points.shape[0])
                            * jax.vmap(_vec(t.fn), (None, 0))(theta, t.points).reshape(-1) for t in terms])


def _jacobian(terms: Sequence[Term], theta):
    """(r (m,), J (m, n)): reverse mode per point (one pass per residual component), vmapped."""
    rs, Js = [], []
    for t in terms:
        s = jnp.sqrt(t.weight / t.points.shape[0])
        f = _vec(t.fn)
        J, r = jax.vmap(jax.jacrev(lambda th, z, f=f: (f(th, z), f(th, z)), has_aux=True), (None, 0))(theta, t.points)
        rs.append(s * r.reshape(-1))
        Js.append(s * J.reshape(-1, theta.size))
    return jnp.concatenate(rs), jnp.concatenate(Js)


def _factor(J, cfg: GNConfig, lam):
    """A solver for the Jacobi-scaled damped normal equations: returns f(b) = d with (J^T J + lam) d ~ J^T b
    (primal) or d = J^T (J J^T + lam)^{-1} b (dual), b in R^m."""
    m, n = J.shape
    dual = cfg.mode == "dual" or (cfg.mode == "auto" and n > m)
    M = J @ J.T if dual else J.T @ J
    M = 0.5 * (M + M.T)
    s = 1.0 / (jnp.sqrt(jnp.diag(M)) + 1e-30)
    L = jnp.linalg.cholesky(M * s[:, None] * s[None, :] + lam * jnp.eye(M.shape[0], dtype=M.dtype))
    cho = lambda b: jax.scipy.linalg.cho_solve((L, True), b)
    if dual:
        return lambda b: J.T @ (s * cho(s * b))
    return lambda b: s * cho(s * (J.T @ b))


def gn_step(terms: Sequence[Term], theta, cfg: GNConfig):
    """One Gauss-Newton step. Returns (theta_new, loss at theta, info)."""
    r, J = _jacobian(terms, theta)
    loss = 0.5 * jnp.sum(r ** 2)
    lam = jnp.clip(loss, cfg.damping, cfg.lm_max) if cfg.lm_max > 0 else cfg.damping
    solve = _factor(J, cfg, lam)
    v = -solve(r)
    info = {}
    if cfg.geodesic:
        f = lambda th: _stacked(terms, th)
        fvv = jax.jvp(lambda th: jax.jvp(f, (th,), (v,))[1], (theta,), (v,))[1]
        a = -solve(fvv)
        ratio = 2.0 * jnp.linalg.norm(a) / (jnp.linalg.norm(v) + 1e-30)
        a = jnp.where(ratio <= cfg.geo_ratio, 1.0, 0.0) * a
        info["geo_ratio"] = ratio
    else:
        a = jnp.zeros_like(v)
    etas = jnp.asarray(cfg.ls_steps, dtype=theta.dtype)
    trial = lambda eta: theta + eta * v + 0.5 * eta ** 2 * a
    losses = jax.vmap(lambda eta: 0.5 * jnp.sum(_stacked(terms, trial(eta)) ** 2))(etas)
    losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
    eta = etas[jnp.argmin(losses)]      # the best trial step, even if no trial lowers the loss (as JAX-NG)
    info["eta"] = eta
    return trial(eta), loss, info
