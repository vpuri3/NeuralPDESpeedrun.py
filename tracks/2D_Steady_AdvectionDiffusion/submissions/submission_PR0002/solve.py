"""Submission 2: PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Steady advection-diffusion
by a cellular flow (Pe = 64): PDE residual + the wall temperatures (x = 0, 1) and the insulated walls (y = 0, 2) as
boundary terms, collocation clustered at the walls, float64."""

from speedrun.submissions.submission_PR0002.gn import GNConfig
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    pe, fn = problem.params["Pe"], problem.fns

    def residual(U, z):
        _, T_x, T_xx = partials(U, z, 0, 2)
        _, T_y, T_yy = partials(U, z, 1, 2)
        u, v = fn["velocity"](z[0], z[1])
        return (T_xx + T_yy) / pe - u * T_x - v * T_y

    def walls(U, z):
        return U(z)[:1] - fn["bc"](z[0], z[1])

    def insulated(U, z):
        return partials(U, z, 1, 1)[1]

    cfg = Config(gn=GNConfig(lm_max=0.1), width=50, depth=4, n_colloc=8000, n_bc=1000, steps=6000,
                 cluster_axes=("x", "y"))
    return PINN(problem, residual, 1, cfg, bcs=[BC("x", walls, weight=10.0), BC("y", insulated, weight=10.0)]).compile()
