"""Submission 4: shallow PINN trained by Levenberg-Marquardt (speedrun/submissions/submission_PR0004), after
Shahab et al. (arXiv:2602.08515). KdV u_t + u u_x + delta^2 u_xxx = 0, x periodic (exact, through (sin, cos)(pi x)
inputs): one tanh MLP on the whole space-time domain, u = u~(t, x); the initial state is residual rows (500
points). 10k fixed collocation points, float64."""
from speedrun.submissions.submission_PR0004.pinn import Config, LMPINN, arch, derivs, face, query, solve  # noqa: F401


def setup(problem):
    d2, u0 = problem.params["delta"] ** 2, problem.fns["u0"]

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, u_x, _, u_xxx = derivs(u, z, 1, 3)
        return u_t + u_ * u_x + d2 * u_xxx

    bcs = [(face(problem, "t", 0.0, seed=1), lambda u, z: u(z) - u0(z[1]), 1.0)]
    width, layers = arch(32, 2)
    return LMPINN(problem, residual, Config(width=width, layers=layers), bcs=bcs).compile()
