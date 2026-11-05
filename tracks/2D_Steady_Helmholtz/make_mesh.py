"""Build the mesh of 2D_Steady_Helmholtz: linear triangles of edge length ~HMAX on the disk minus the square, with
Shewchuk's Triangle (the `triangle` package, needed only here; the mesh is in git as data/mesh.npz):

    uv run --with triangle python -m tracks.2D_Steady_Helmholtz.make_mesh

Writes points (n, 2); triangles (m, 3), counterclockwise; obstacle / outer: ids of the nodes on the square / circle;
outer_edges (b, 2): the boundary edges on the circle (the circle is the polygon through its boundary nodes).
"""
import math
from pathlib import Path

import numpy as np

from .problem import A, C, R

HMAX = 0.011       # ~10 nodes per wavelength (2 pi / 60 = 0.105); ~10k nodes


def build(hmax=HMAX):
    import triangle
    nc = math.ceil(2 * math.pi * R / hmax)                     # segments on the circle
    ns = math.ceil(math.sqrt(2) * A / hmax)                    # segments per edge of the square
    phi = 2 * math.pi * np.arange(nc) / nc
    circle = np.stack([C[0] + R * np.cos(phi), C[1] + R * np.sin(phi)], -1)
    corners = [np.array(C) + A * np.array([math.cos(q), math.sin(q)]) for q in np.arange(4) * math.pi / 2]
    square = np.concatenate([corners[i] + (corners[(i + 1) % 4] - corners[i]) * np.arange(ns)[:, None] / ns
                             for i in range(4)])
    verts = np.concatenate([circle, square])
    seg = np.concatenate([np.stack([np.arange(nc), (np.arange(nc) + 1) % nc], -1),
                          nc + np.stack([np.arange(4 * ns), (np.arange(4 * ns) + 1) % (4 * ns)], -1)])
    out = triangle.triangulate({"vertices": verts, "segments": seg, "holes": [list(C)]},
                               f"pq30a{math.sqrt(3) / 4 * hmax**2:.12f}YQ")   # Y: no new points on the boundary
    p, t = out["vertices"], out["triangles"]
    area = 0.5 * ((p[t[:, 1], 0] - p[t[:, 0], 0]) * (p[t[:, 2], 1] - p[t[:, 0], 1])
                  - (p[t[:, 2], 0] - p[t[:, 0], 0]) * (p[t[:, 1], 1] - p[t[:, 0], 1]))
    t[area < 0] = t[area < 0][:, [0, 2, 1]]
    return {"points": p, "triangles": t, "obstacle": np.arange(nc, nc + 4 * ns), "outer": np.arange(nc),
            "outer_edges": np.stack([np.arange(nc), (np.arange(nc) + 1) % nc], -1)}


if __name__ == "__main__":
    m = build()
    path = Path(__file__).parent / "data" / "mesh.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **m)
    print(f"{path}: {len(m['points'])} nodes, {len(m['triangles'])} triangles, "
          f"{len(m['obstacle'])} on the square, {len(m['outer'])} on the circle")
