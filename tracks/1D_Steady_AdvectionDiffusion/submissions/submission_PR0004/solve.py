"""Submission 4: shallow PINN trained by Levenberg-Marquardt (speedrun/submissions/submission_PR0004), after
Shahab et al. (arXiv:2602.08515). Steady advection-diffusion -eps u'' + b u' = f, u = u~(x); the walls
u(0) = u(1) = 0 are residual rows. tanh MLP, 10k fixed collocation points, float64."""
import numpy as np

from speedrun.submissions.submission_PR0004.pinn import Config, LMPINN, arch, derivs, query, solve  # noqa: F401


def setup(problem):
    eps, b, f, g = problem.params["eps"], problem.params["b"], problem.fns["forcing"], problem.fns["bc"]

    def residual(u, z):
        _, u_x, u_xx = derivs(u, z, 0, 2)
        return -eps * u_xx + b * u_x - f(z[0])

    width, layers = arch(32, 2)
    walls = np.array([[0.0], [1.0]])
    return LMPINN(problem, residual, Config(width=width, layers=layers), bcs=[(walls, lambda u, z: u(z) - g(z[0]), 1.0)]).compile()
