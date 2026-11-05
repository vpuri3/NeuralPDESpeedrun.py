"""Submission 2: time-marching PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Damped
wave: second order in time (u and u_t continued exactly across windows), u = 0 on both walls by an odd periodic
extension in x, 16 windows of a sine MLP 50 x 4 on (t, 30 Fourier modes of x), float64."""
from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import PINN, Config, query, solve  # noqa: F401


def setup(problem):
    c2, gamma = problem.params["c"] ** 2, problem.params["gamma"]

    def residual(U, z):
        _, u_t, u_tt = partials(U, z, 0, 2)
        return u_tt - c2 * partials(U, z, 1, 2)[2] + gamma * u_t

    cfg = Config(n_windows=16, width=50, depth=4, n_modes=30, n_colloc=20000, steps=100, time_order=2,
                 odd_axes=("x",))
    return PINN(problem, residual, 1, cfg, ic=((problem.fns["u0"],), (problem.fns["ut0"],))).compile()
