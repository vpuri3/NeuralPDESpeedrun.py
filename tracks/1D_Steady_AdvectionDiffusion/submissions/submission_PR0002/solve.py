"""Submission 2: PINN trained by Gauss-Newton (speedrun/submissions/submission_PR0002). Steady 1D advection-diffusion
-eps u'' + b u' = f with an outflow boundary layer: PDE residual + the Dirichlet walls u(0) = u(1) = 0 as boundary
terms, collocation clustered at the walls, fixed damping 1e-8 (no Levenberg-Marquardt clip), float64."""

from speedrun.submissions.submission_PR0002.net import partials
from speedrun.submissions.submission_PR0002.pinn import BC, PINN, Config, query, solve  # noqa: F401


def setup(problem):
    eps, b, fn = problem.params["eps"], problem.params["b"], problem.fns

    def residual(U, z):
        _, u_x, u_xx = partials(U, z, 0, 2)
        return -eps * u_xx + b * u_x - fn["forcing"](z[0])

    def walls(U, z):
        return U(z)[:1] - fn["bc"](z[0])

    cfg = Config(width=50, depth=4, n_colloc=4000, n_bc=2, steps=4000, cluster_axes=("x",))
    return PINN(problem, residual, 1, cfg, bcs=[BC("x", walls, weight=10.0)]).compile()
