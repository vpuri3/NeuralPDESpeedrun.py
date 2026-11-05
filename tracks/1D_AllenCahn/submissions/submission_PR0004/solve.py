"""Submission 4: shallow PINN trained by Levenberg-Marquardt (speedrun/submissions/submission_PR0004), after
Shahab et al. (arXiv:2602.08515). Allen-Cahn u_t - eps u_xx + kappa (u^3 - u) = 0, x periodic (exact, through
(sin, cos)(pi x) inputs): one tanh MLP on the whole space-time domain, u = u~(t, x); the initial state is residual
rows (500 points). 10k fixed collocation points, float64."""
from speedrun.submissions.submission_PR0004.pinn import Config, LMPINN, arch, derivs, face, query, solve  # noqa: F401


def setup(problem):
    eps, kappa, u0 = problem.params["eps"], problem.params["kappa"], problem.fns["u0"]

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, _, u_xx = derivs(u, z, 1, 2)
        return u_t - eps * u_xx + kappa * (u_ ** 3 - u_)

    bcs = [(face(problem, "t", 0.0, seed=1), lambda u, z: u(z) - u0(z[1]), 1.0)]
    width, layers = arch(32, 2)
    return LMPINN(problem, residual, Config(width=width, layers=layers), bcs=bcs).compile()
