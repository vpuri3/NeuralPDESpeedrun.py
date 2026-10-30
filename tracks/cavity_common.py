"""Shared definition of the lid-driven cavity tracks 7A, 7B (unsteady Navier-Stokes) and 8 (steady Stokes),
following the `LDC` case of https://github.com/vpuri3/spec (nsSpectral/driver.m): the square [-1, 1]^2,
classical lid u = 1 on y = 1 (corners included), fluid at rest at t = 0, run to T = 5."""
import math

import torch

from speedrun import plotting
from speedrun.numerics.sem import SEMNavierStokes, gll, rectangle_map
from speedrun.problem import Problem

N_EVAL = 129
T_END = 5.0
NT = 10                 # snapshots t = 0.5, 1.0, ..., 5.0
LINF_MARGIN = 0.05      # L-inf is not scored within this distance of the walls (lid-corner singularities)
CFL = 2.0               # reference/baseline time step: dt = CFL * (smallest GLL spacing), |lid| = 1


def u_bc(x, y):
    """Lid u = 1 on y = 1, including the two top corners (as in the spec); no-slip elsewhere."""
    return (y > 1.0 - 1e-12).to(x.dtype)


def v_bc(x, y):
    return torch.zeros_like(x)


def statement(Re):
    if Re:
        eq = (f"Unsteady incompressible Navier-Stokes, Re = {Re:g} (nu = 1/Re), t in [0, {T_END:g}]:\n\n"
              "    u_t + (u . grad) u + grad p = nu Lap u,      div u = 0\n"
              "    u(0, x, y) = v(0, x, y) = 0      (impulsively started lid)")
        out = (f"Output: (u, v) on the uniform {N_EVAL} x {N_EVAL} grid x, y = linspace(-1, 1, {N_EVAL}) at the {NT}\n"
               f"times t = 0.5, 1.0, ..., {T_END:g}; shape (2, {NT}, {N_EVAL}, {N_EVAL}).")
    else:
        eq = "Steady incompressible Stokes flow:\n\n    - Lap u + grad p = 0,      div u = 0"
        out = (f"Output: (u, v) on the uniform {N_EVAL} x {N_EVAL} grid x, y = linspace(-1, 1, {N_EVAL});\n"
               f"shape (2, {N_EVAL}, {N_EVAL}).")
    return f"""
Lid-driven cavity on the square [-1, 1]^2.
{eq}

Boundary conditions (Dirichlet velocity):
    u = 1, v = 0     on the lid y = 1 (including the corners (+-1, 1))
    u = v = 0        on the other three walls

The velocity is discontinuous at the two top corners. RelL2 is scored over all output points; L-inf
only over points at distance >= {LINF_MARGIN:g} from the walls (problem.linf_margin). Pressure is NOT scored.

{out}
"""


def make_cavity_problem(track, Re, device="cpu", dtype=torch.float64):
    g = torch.linspace(-1.0, 1.0, N_EVAL, dtype=dtype, device=device)
    axes = {"x": g, "y": g.clone()}
    domain = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    fns = {"u_bc": u_bc, "v_bc": v_bc}
    if Re:
        axes = {"t": torch.linspace(T_END / NT, T_END, NT, dtype=dtype, device=device), **axes}
        domain["t"] = (0.0, T_END)
        fns.update(u0=lambda x, y: torch.zeros_like(x), v0=lambda x, y: torch.zeros_like(x))
        title = f"Lid-driven cavity, Re = {Re:g}, impulsive start, t in [0, {T_END:g}]"
    else:
        title = "Stokes lid-driven cavity"
    return Problem(
        track=track,
        title=title,
        statement=statement(Re),
        fields=("u", "v"),
        axes=axes,
        domain=domain,
        params={"Re": Re, "nu": (1.0 / Re) if Re else 1.0, "stokes": Re == 0},
        fns=fns,
        device=device,
        dtype=dtype,
        linf_margin=LINF_MARGIN,
    )


def time_step(N, t_out, cfl=CFL):
    """Largest dt <= cfl * (smallest GLL spacing) that divides the snapshot interval."""
    z, _ = gll(N)
    every = float(t_out[0])
    return every / math.ceil(every / (cfl * float(z[1] - z[0])))


def sem_solve(p, N, cfl=CFL, callback=None):
    """The spec's discretization: single-element Legendre PN-PN-2. Unsteady: BDF3/EXT3 + pressure projection
    with fast-diagonalization solves; Stokes: Uzawa CG."""
    S = SEMNavierStokes(N, rectangle_map(-1.0, 1.0, -1.0, 1.0), dtype=p.dtype, device=p.device)
    if p.params["stokes"]:
        u = torch.stack(S.stokes_uzawa(p.params["nu"], p.fns["u_bc"], p.fns["v_bc"]))
    else:
        t = p.axes["t"]
        u = S.unsteady(p.params["nu"], p.fns["u0"], p.fns["v0"], p.fns["u_bc"], p.fns["v_bc"], t,
                       time_step(N, t, cfl), callback=callback)
    return S.interp_rect(u, p.axes["x"], p.axes["y"])


def reference(p, N_ref, N_chk):
    every = max(1, int(round(0.25 / time_step(N_ref, p.axes["t"])))) if "t" in p.axes else 0
    cb = (lambda n, t, u, v: print(f"    t = {t:.3f}", flush=True) if n % every == 0 else None) if every else None
    u = sem_solve(p, N_ref, callback=cb)
    c = sem_solve(p, N_chk)
    m = p.linf_mask()
    e = (c - u).abs()
    meta = {"method": f"single-element Legendre PN-PN-2 (spec), N={N_ref}"
                      + (f", BDF3/EXT3, dt={time_step(N_ref, p.axes['t']):.3e}" if "t" in p.axes else ", Uzawa CG"),
            f"check_N{N_chk}_vs_N{N_ref}_linf_scored": float(e[m.expand_as(e)].max()),
            f"check_N{N_chk}_vs_N{N_ref}_linf_all": float(e.max()),
            f"check_N{N_chk}_vs_N{N_ref}_rel_l2": float(torch.linalg.norm(c - u) / torch.linalg.norm(u))}
    return u, meta


def plot(problem, u, figdir):
    x, y = problem.axes["x"], problem.axes["y"]
    if "t" not in problem.axes:
        plotting.fields2d(x, y, {"u": u[0], "v": u[1]}, figdir / "reference.png", suptitle=problem.title,
                          streamplot=True)
        return
    t = problem.axes["t"]
    for k in (0, len(t) // 2 - 1, len(t) - 1):
        plotting.fields2d(x, y, {"u": u[0, k], "v": u[1, k]}, figdir / f"reference_t{float(t[k]):.1f}.png",
                          suptitle=f"{problem.title}: t = {float(t[k]):.1f}", streamplot=True)
    speed = torch.sqrt(u[0] ** 2 + u[1] ** 2)
    plotting.gif2d(x, y, speed, t, figdir / "reference.gif", title=f"{problem.track}: |u|", cmap="viridis")
