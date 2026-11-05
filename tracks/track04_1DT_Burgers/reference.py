"""Reference: the exact Cole-Hopf solution (Basdevant et al. 1986, eq. for the -sin(pi x) initial state),

    u(t, x) = - int sin(pi (x - s)) f(x - s) G(s) ds / int f(x - s) G(s) ds,
    f(y) = exp(-cos(pi y) / (2 pi nu)),  G(s) = exp(-s^2 / (4 nu t)),

evaluated with s = sqrt(4 nu t) z by the trapezoidal rule in z on [-Z, Z] (exponentially convergent for this
smooth, rapidly decaying integrand), the exponents shifted by their maximum (log-sum-exp). Checked against
half the nodes and a shorter interval."""
import math

import torch

from speedrun import plotting

from .problem import make_problem


def cole_hopf(t, x, nu, n_nodes, z_max, chunk=8):
    """u on the tensor grid t x x (t[0] may be 0)."""
    z = torch.linspace(-z_max, z_max, n_nodes, dtype=t.dtype, device=t.device)
    w = torch.full_like(z, 2 * z_max / (n_nodes - 1))
    w[0] = w[-1] = w[0] / 2
    out = torch.empty(t.numel(), x.numel(), dtype=t.dtype, device=t.device)
    for i in range(0, t.numel(), chunk):
        tc = t[i:i + chunk]
        y = x[None, :, None] - torch.sqrt(4 * nu * tc)[:, None, None] * z          # (nt, nx, nz)
        e = -torch.cos(math.pi * y) / (2 * math.pi * nu) - z**2
        p = torch.exp(e - e.amax(-1, keepdim=True)) * w
        out[i:i + chunk] = -(torch.sin(math.pi * y) * p).sum(-1) / p.sum(-1)
    out[t == 0] = -torch.sin(math.pi * x)
    out[:, 0] = out[:, -1] = 0.0          # exact by symmetry (the formula gives O(1e-16))
    return out


def compute(device="cpu"):
    p = make_problem(device=device)
    t, x, nu = p.axes["t"], p.axes["x"], p.params["nu"]
    ref = cole_hopf(t, x, nu, 4001, 14.0)
    coarse = cole_hopf(t, x, nu, 2001, 14.0)
    short = cole_hopf(t, x, nu, 4001, 12.0)
    meta = {"method": "exact Cole-Hopf solution, trapezoidal rule in z = s / sqrt(4 nu t), 4001 nodes on [-14, 14]",
            "check_nodes2001_vs_4001_linf": float((coarse - ref).abs().max()),
            "check_z12_vs_z14_linf": float((short - ref).abs().max())}
    return ref[None], meta, {}


def plot(problem, u, extras, figdir):
    t, x = problem.axes["t"], problem.axes["x"]
    plotting.spacetime(x, t, u[0], figdir / "reference.png", title="Track 04: Burgers 1D, nu=0.01/pi")
    plotting.snapshots1d(x, t, u[0], figdir / "reference_snapshots.png", title="Track 04 snapshots")
    plotting.gif1d(x, t, u[0], figdir / "reference.gif", title="Track 04")
