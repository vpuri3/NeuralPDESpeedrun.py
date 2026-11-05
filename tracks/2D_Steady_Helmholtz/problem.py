"""2D_Steady_Helmholtz: scattering of a plane wave by a square in a disk, the MATLAB PDE Toolbox example "Helmholtz's
Equation on Disk with Square Hole" (https://www.mathworks.com/help/pde/ug/helmholtzs-equation-on-disk-with-a-square-hole.html;
geometry from its scatterg.m, boundary conditions from scatterb.m). The output points are the nodes of a quadratic
triangle mesh like the example's generateMesh(model, Hmax=0.02), built in Python by make_mesh.py (data/mesh.npz).
"""
import math
from pathlib import Path

import numpy as np
import torch

from speedrun.problem import Problem, xp_of

K = 60.0                                  # wave number
C = (0.8, 0.5)                            # center of the disk and of the square
R = 0.45                                  # radius of the disk
A = 0.1 / math.sqrt(2.0)                  # half-diagonal of the square (side 0.1, rotated by 45 degrees)
MESH = Path(__file__).parent / "data" / "mesh.npz"

STATEMENT = rf"""
Helmholtz's equation for the wave r(x, y) reflected by a square obstacle in a disk (time-harmonic, e^(-i omega t)):

    -Lap r - k^2 r = 0,        k = {K:g}
    domain: the disk |(x, y) - c| < R minus the square |x - c_x| + |y - c_y| <= a,
            c = ({C[0]:g}, {C[1]:g}),  R = {R:g},  a = 0.1 / sqrt(2)   (a square of side 0.1, rotated by 45 degrees)
    on the square:  r = -exp(-i k x)            (the incident wave exp(-i k x), traveling in -x, is reflected:
                                                 the total field vanishes on the obstacle)
    on the circle:  dr/dn - i k r = 0           (nonreflecting boundary condition, n the outward normal)

r is complex. The square's 4 corners are re-entrant (interior angle 270 degrees): grad r ~ rho^(-1/3) there.

Output: Re r and Im r at the nodes of the given mesh, problem.mesh: quadratic triangles of edge length <= 0.02
(as MATLAB's generateMesh(model, Hmax=0.02)); "points" (n, 2), "triangles" (m, 6: corners counterclockwise, then
the midside nodes of edges 12, 23, 31), "obstacle" / "outer": ids of the nodes on the square / on the circle.
Shape (2, n).
"""


def obstacle_value(x, y):
    """Dirichlet data on the square, r = -exp(-i k x), as (Re, Im)."""
    xp = xp_of(x)
    return -xp.cos(K * x), xp.sin(K * x)


def make_problem(device="cpu", dtype=torch.float64) -> Problem:
    m = np.load(MESH)
    tt = lambda k: torch.as_tensor(m[k], device=device)  # noqa: E731
    pts = tt("points").to(dtype)
    return Problem(
        track="2D_Steady_Helmholtz",
        title="2D Helmholtz, plane wave scattered by a square in a disk, k = 60",
        statement=STATEMENT,
        fields=("r_re", "r_im"),
        axes={"node": torch.arange(pts.shape[0], device=device)},
        domain={"x": (C[0] - R, C[0] + R), "y": (C[1] - R, C[1] + R)},
        periodic={"x": False, "y": False},
        params={"k": K, "center": C, "radius": R, "half_diagonal": A},
        fns={"obstacle": obstacle_value},
        mesh={"points": pts, "triangles": tt("triangles"), "obstacle": tt("obstacle"), "outer": tt("outer")},
        device=device,
        dtype=dtype,
    )
