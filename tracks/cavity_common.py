"""Shared definition of the regularized lid-driven cavity used by tracks 7A, 7B (Navier-Stokes) and 8 (Stokes)."""
import torch

from speedrun import plotting
from speedrun.numerics.streamfunction import StreamfunctionSolver
from speedrun.problem import Problem

N_EVAL = 129


def u_bc(x, y):
    """Regularized lid u = 16 x^2 (1 - x)^2 on y = 1; no-slip elsewhere."""
    return torch.where(y > 1.0 - 1e-12, 16.0 * x**2 * (1.0 - x) ** 2, torch.zeros_like(x))


def v_bc(x, y):
    return torch.zeros_like(x)


def statement(Re):
    eq = ("Steady incompressible Navier-Stokes, Re = %g:\n\n"
          "    (u . grad) u + grad p = (1/Re) Lap u,      div u = 0" % Re) if Re else \
         ("Steady incompressible Stokes flow:\n\n"
          "    - Lap u + grad p = 0,      div u = 0")
    return f"""
Lid-driven cavity on the unit square [0, 1]^2 with a *regularized* lid.
{eq}

Boundary conditions (Dirichlet velocity):
    u = 16 x^2 (1 - x)^2,  v = 0     on the lid y = 1
    u = v = 0                        on the other three walls

The regularized lid removes the corner velocity discontinuities of the classical (u = 1) lid, so the
solution is smooth enough for a pointwise (L-infinity) reference. Pressure is NOT scored.

Output: (u, v) on the uniform {N_EVAL} x {N_EVAL} tensor grid x, y = linspace(0, 1, {N_EVAL});
shape (2, {N_EVAL}, {N_EVAL}).
"""


def make_cavity_problem(track, Re, device="cpu", dtype=torch.float64):
    g = torch.linspace(0.0, 1.0, N_EVAL, dtype=dtype, device=device)
    title = f"Lid-driven cavity (regularized lid), Re = {Re:g}" if Re else "Stokes lid-driven cavity (regularized lid)"
    return Problem(
        track=track,
        title=title,
        statement=statement(Re),
        fields=("u", "v"),
        axes={"x": g, "y": g.clone()},
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        params={"Re": Re, "nu": (1.0 / Re) if Re else 1.0, "stokes": Re == 0},
        fns={"u_bc": u_bc, "v_bc": v_bc},
        device=device,
        dtype=dtype,
    )


def continuation_path(Re):
    return [1.0 / r for r in (100.0, 400.0) if r < Re]


def chebyshev_solve(p, N):
    S = StreamfunctionSolver(N, N, p.domain["x"], p.domain["y"], p.fns["u_bc"], p.fns["v_bc"],
                             dtype=p.dtype, device=p.device)
    S.solve(p.params["nu"], stokes=p.params["stokes"], nu_path=continuation_path(p.params["Re"]))
    return S.evaluate(p.axes["x"], p.axes["y"])


def reference(p, N_ref, N_chk):
    u = chebyshev_solve(p, N_ref)
    c = chebyshev_solve(p, N_chk)
    meta = {"method": f"Chebyshev streamfunction collocation, N={N_ref}, Newton"
                      + (" with Re-continuation" if continuation_path(p.params['Re']) else ""),
            f"check_N{N_chk}_vs_N{N_ref}_linf": float((c - u).abs().max()),
            f"check_N{N_chk}_vs_N{N_ref}_rel_l2": float(torch.linalg.norm(c - u) / torch.linalg.norm(u))}
    return u, meta


def plot(problem, u, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    plotting.fields2d(x, y, {"u": u[0], "v": u[1]}, figdir / "reference.png", suptitle=problem.title,
                      streamplot=True)
