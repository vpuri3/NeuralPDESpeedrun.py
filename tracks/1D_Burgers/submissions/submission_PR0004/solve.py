"""Submission 4: shallow PINN trained by Levenberg-Marquardt (speedrun/submissions/submission_PR0004), after
Shahab et al. (arXiv:2602.08515). Raissi's Burgers u_t + u u_x = nu u_xx on [0, 1] x [-1, 1]: one tanh MLP on the
whole space-time domain, u = u~(t, x); the initial state and the walls u(t, +-1) = 0 are residual rows (500 points
each). 10k fixed collocation points, float64."""
from speedrun.submissions.submission_PR0004.pinn import Config, LMPINN, arch, derivs, face, query, solve  # noqa: F401


def setup(problem):
    nu, fn = problem.params["nu"], problem.fns

    def residual(u, z):
        u_t = derivs(u, z, 0, 1)[1]
        u_, u_x, u_xx = derivs(u, z, 1, 2)
        return u_t + u_ * u_x - nu * u_xx

    bcs = [(face(problem, "t", 0.0, seed=1), lambda u, z: u(z) - fn["u0"](z[1]), 1.0),
           (face(problem, "x", -1.0, seed=2), lambda u, z: u(z) - fn["bc_left"](z[0]), 1.0),
           (face(problem, "x", 1.0, seed=3), lambda u, z: u(z) - fn["bc_right"](z[0]), 1.0)]
    width, layers = arch(32, 4)
    return LMPINN(problem, residual, Config(width=width, layers=layers, max_iters=None), bcs=bcs).compile()
