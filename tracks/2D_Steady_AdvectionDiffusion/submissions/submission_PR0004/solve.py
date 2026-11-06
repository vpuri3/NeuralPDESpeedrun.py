"""Submission 4: shallow PINN + Levenberg-Marquardt (speedrun/submissions/submission_PR0004; Shahab et al.,
arXiv:2602.08515). Steady cellular-flow heat transport (1/Pe) Lap T - u T_x - v T_y = 0 on
[0, 1] x [0, 2], T = T~(x, y); the walls T(0, y) = 0, T(1, y) = 1 and the insulated walls T_y = 0 at y = 0, 2 are
residual rows."""
from speedrun.submissions.submission_PR0004.pinn import Cond, Config, LMPINN, derivs, query, solve  # noqa: F401

# hyperparameters (all others: the defaults in pinn.Config)
CONFIG = Config(width=32, layers=4, resample_every=50)


def setup(problem):
    pe, fn = problem.params["Pe"], problem.fns
    (x0, x1), (y0, y1) = problem.domain["x"], problem.domain["y"]

    def residual(u, z):
        _, T_x, T_xx = derivs(u, z, 0, 2)
        _, T_y, T_yy = derivs(u, z, 1, 2)
        a, b = fn["velocity"](z[0], z[1])
        return (T_xx + T_yy) / pe - a * T_x - b * T_y

    dirichlet = lambda u, z: u(z) - fn["bc"](z[0], z[1])
    insulated = lambda u, z: derivs(u, z, 1, 1)[1]
    conds = [Cond("cold wall x = 0", dirichlet, face=("x", x0)), Cond("hot wall x = 1", dirichlet, face=("x", x1)),
             Cond("insulated y = 0", insulated, face=("y", y0)), Cond("insulated y = 2", insulated, face=("y", y1))]
    return LMPINN(problem, residual, CONFIG, conds).compile()
