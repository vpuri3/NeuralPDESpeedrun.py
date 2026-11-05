"""Submission 4: shallow PINN trained by Levenberg-Marquardt (speedrun/submissions/submission_PR0004), after
Shahab et al. (arXiv:2602.08515). Damped wave u_tt = c^2 u_xx - gamma u_t on [0, 4] x [0, 1]: one tanh MLP on the
whole space-time domain, u = u~(t, x); u(0, x), u_t(0, x) and the walls u(t, 0) = u(t, 1) = 0 are residual rows
(500 points each). 10k fixed collocation points, float64."""
from speedrun.submissions.submission_PR0004.pinn import Config, LMPINN, arch, derivs, face, query, solve  # noqa: F401


def setup(problem):
    c2, gamma, fn = problem.params["c"] ** 2, problem.params["gamma"], problem.fns
    lo, hi = problem.domain["x"]

    def residual(u, z):
        _, u_t, u_tt = derivs(u, z, 0, 2)
        return u_tt - c2 * derivs(u, z, 1, 2)[2] + gamma * u_t

    ic = face(problem, "t", 0.0, seed=1)
    bcs = [(ic, lambda u, z: u(z) - fn["u0"](z[1]), 1.0),
           (ic, lambda u, z: derivs(u, z, 0, 1)[1] - fn["ut0"](z[1]), 1.0),
           (face(problem, "x", lo, seed=2), lambda u, z: u(z) - fn["bc_left"](z[0]), 1.0),
           (face(problem, "x", hi, seed=3), lambda u, z: u(z) - fn["bc_right"](z[0]), 1.0)]
    width, layers = arch(32, 2)
    return LMPINN(problem, residual, Config(width=width, layers=layers), bcs=bcs).compile()
