"""Record #0 (classical baseline): LSODA (adaptive BDF/Adams switching; scipy) with rtol 1e-8 and the analytic
Jacobian, on the CPU. Stiff time integration is sequential, so the GPU check reports CPU."""
import numpy as np
import torch
from scipy.integrate import solve_ivp

RTOL = 1e-8


def solve(problem, state=None):
    mu = problem.params["mu"]
    t = problem.axes["t"].cpu().numpy()
    f = lambda s, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]  # noqa: E731
    jac = lambda s, y: [[0.0, 1.0], [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]]  # noqa: E731
    sol = solve_ivp(f, (t[0], t[-1]), [problem.fns["y0"], problem.fns["dy0"]], method="LSODA", t_eval=t,
                    rtol=RTOL, atol=RTOL * 1e-2, jac=jac)
    y = torch.from_numpy(np.ascontiguousarray(sol.y[0]))
    return y.to(device=problem.device, dtype=problem.dtype)[None]
