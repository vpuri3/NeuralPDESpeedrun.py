"""Network and derivative helpers: a sine-activated MLP (as JAX-NG's KdV / KS problems) and jet-based derivatives."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.jet import jet


def init_mlp(key, sizes, dtype=jnp.float64):
    """[(W, b)] per layer; W ~ N(0, 2 / (fan_in + fan_out)) (Glorot normal), b = 0."""
    params = []
    for k, m, n in zip(jax.random.split(key, len(sizes) - 1), sizes[:-1], sizes[1:]):
        params.append((jax.random.normal(k, (n, m), dtype) * jnp.sqrt(2.0 / (m + n)), jnp.zeros(n, dtype)))
    return params


def mlp(params, h):
    """sin hidden layers, linear output; h (d,) -> (out,)."""
    for W, b in params[:-1]:
        h = jnp.sin(W @ h + b)
    W, b = params[-1]
    return W @ h + b


def derivs(f, z, k: int):
    """[f(z), f'(z), ..., f^(k)(z)] of a scalar function of a scalar, by Taylor-mode AD (jet)."""
    if k == 0:
        return [f(z)]
    y, ys = jet(f, (z,), ((jnp.ones_like(z),) + (jnp.zeros_like(z),) * (k - 1),))
    return [y, *ys]


def periodic_interp(values, lo, length, endpoint=True):
    """Trigonometric interpolant of samples on a uniform periodic grid (x_j = lo + j length / n), as a
    JAX-traceable scalar function. The coefficients are numpy constants (closed-over JAX arrays would become
    hidden inputs of the ahead-of-time compiled functions). endpoint: the last sample duplicates the first."""
    v = np.asarray(values.detach().cpu().numpy() if hasattr(values, "detach") else values, dtype=np.float64)
    v = v[:-1] if endpoint else v
    n = v.size
    c = np.fft.rfft(v) / n
    w = np.where((np.arange(c.size) == 0) | ((n % 2 == 0) & (np.arange(c.size) == n // 2)), 1.0, 2.0)
    a, b, k = w * c.real, -w * c.imag, np.arange(c.size, dtype=np.float64)

    def f(z):
        th = 2.0 * np.pi * k * ((z - lo) / length)
        return jnp.sum(a * jnp.cos(th) + b * jnp.sin(th))

    return f


def partials(U, z, i: int, k: int):
    """[U, d_i U, ..., d_i^k U] at the point z (vector-valued U(z)), along coordinate i."""
    e = jnp.zeros_like(z).at[i].set(1.0)          # (jet has no scatter rule: shift along a one-hot instead)
    return derivs(lambda s: U(z + e * (s - z[i])), z[i], k)
