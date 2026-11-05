"""Reference: Chebyshev collocation on one cell (the baseline's solver) at N = 128, mirrored to [1, 2],
cross-checked against N = 96 and N = 112, and against a direct solve on the full domain [0, 1] x [0, 2]
(two cells, no symmetry assumed) at N = 96 x 192."""
import torch

from speedrun import plotting

from .baseline import chebyshev_cell, to_output
from .problem import make_problem

N_REF = 128


def full_domain(p, N, M):
    """Collocation on [0, 1] x [0, 2] directly (one Chebyshev grid in y), for the symmetry check."""
    from speedrun.numerics import bary_matrix, cheb
    dt, dev = p.dtype, p.device
    x, Dx = cheb(N, 0.0, 1.0, dtype=dt, device=dev)
    y, Dy = cheb(M, 0.0, 2.0, dtype=dt, device=dev)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u, v = p.fns["velocity"](X, Y)
    Ix, Iy = torch.eye(N + 1, dtype=dt, device=dev), torch.eye(M + 1, dtype=dt, device=dev)
    DX, DY = torch.kron(Dx, Iy), torch.kron(Ix, Dy)
    A = (torch.kron(Dx @ Dx, Iy) + torch.kron(Ix, Dy @ Dy)) / p.params["Pe"] - u.reshape(-1, 1) * DX \
        - v.reshape(-1, 1) * DY
    rhs = torch.zeros((N + 1) * (M + 1), dtype=dt, device=dev)
    idx = torch.arange((N + 1) * (M + 1), device=dev).reshape(N + 1, M + 1)
    for j in (0, M):
        A[idx[:, j]] = DY[idx[:, j]]
    for i in (0, N):
        A[idx[i]] = 0.0
        A[idx[i], idx[i]] = 1.0
        rhs[idx[i]] = p.fns["bc"](X[i], Y[i])
    T = torch.linalg.solve(A, rhs).reshape(N + 1, M + 1)
    return (bary_matrix(x, p.axes["x"]) @ T @ bary_matrix(y, p.axes["y"]).T)[None]


def compute(device="cpu"):
    p = make_problem(device=device)
    u = to_output(p, *chebyshev_cell(p, N_REF))
    meta = {"method": f"Chebyshev collocation on one cell, N = {N_REF}, mirrored about y = 1"}
    for n in (96, 112):
        meta[f"check_N{n}_linf"] = float((to_output(p, *chebyshev_cell(p, n)) - u).abs().max())
    meta["check_full_domain_N96x192_linf"] = float((full_domain(p, 96, 192) - u).abs().max())
    x, y, T = chebyshev_cell(p, N_REF)
    from speedrun.numerics import cheb
    Dx = cheb(N_REF, 0.0, 1.0, dtype=p.dtype, device=p.device)[1]
    from speedrun.numerics import clenshaw_curtis_weights
    w = clenshaw_curtis_weights(N_REF, 0.0, 1.0, dtype=p.dtype, device=p.device)
    meta["nusselt"] = float(((Dx @ T)[0] * w).sum())      # mean wall heat flux dT/dx at x = 0 (conduction: 1)
    return u, meta, {}


def plot(problem, u, extras, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    pe = problem.params["Pe"]
    plotting.fields2d(x, y, {"T": u[0]}, figdir / "reference.png", cmap="inferno",
                      suptitle=f"Track 09: steady advection-diffusion, cellular flow, Pe = Re Pr = {pe:g}")
    iy = lambda v: int((y - v).abs().argmin())  # noqa: E731
    plotting.line(x, {"T(x, y=0.5)": u[0, :, iy(0.5)], "T(x, y=1)": u[0, :, iy(1.0)],
                      "T(x, y=0)": u[0, :, iy(0.0)]},
                  figdir / "reference_slices.png", title="Track 09: slices across the cell", ylabel="T")
