"""Shared definition of the Kovasznay-flow tracks 6A (rectangle) and 6B (curved trapezoid), following the
`kov` case of https://github.com/vpuri3/spec (nsSpectral/driver.m, navierStokesSEM/driver.m, geom/para.m)."""
import math

import torch

from speedrun import plotting

RE = 40.0
LAM = RE / 2 - math.sqrt(RE**2 / 4 + 4 * math.pi**2)

EQUATIONS = rf"""Steady incompressible Navier-Stokes, Re = {RE:g}:

    (u . grad) u + grad p = (1/Re) Lap u,      div u = 0

Dirichlet velocity on the whole boundary, equal to the Kovasznay (1948) solution
    u = 1 - exp(lam x) cos(2 pi y),   v = (lam / (2 pi)) exp(lam x) sin(2 pi y),
    lam = Re/2 - sqrt(Re^2/4 + 4 pi^2)
(only its boundary values are data; the interior is unknown to the solver). Pressure is NOT scored."""


def u_bc(x, y):
    return 1.0 - torch.exp(LAM * x) * torch.cos(2 * math.pi * y)


def v_bc(x, y):
    return LAM / (2 * math.pi) * torch.exp(LAM * x) * torch.sin(2 * math.pi * y)


def sem_solve(problem, N):
    """Spec discretization (PN-PN-2 Legendre, single element), steady system solved by Newton."""
    from speedrun.numerics.sem import SEMNavierStokes
    S = SEMNavierStokes(N, problem.fns["map"], dtype=problem.dtype, device=problem.device)
    u, v = S.steady(problem.params["nu"], problem.fns["u_bc"], problem.fns["v_bc"])
    return S.interp_rs(torch.stack([u, v]), problem.axes["r"], problem.axes["s"])


def plot(problem, u, figdir, title):
    if "x" in problem.axes:
        plotting.fields2d(problem.axes["x"], problem.axes["y"], {"u": u[0], "v": u[1]}, figdir / "reference.png",
                          suptitle=title, cmap="viridis", streamplot=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, S = torch.meshgrid(problem.axes["r"].cpu(), problem.axes["s"].cpu(), indexing="ij")
    X, Y = problem.fns["map"](R, S)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, f, name in zip(axs, u, ("u", "v")):
        h = ax.pcolormesh(X.numpy(), Y.numpy(), f.cpu().numpy(), shading="gouraud", cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(h, ax=ax)
    fig.suptitle(title)
    fig.savefig(figdir / "reference.png", dpi=120)
    plt.close(fig)
