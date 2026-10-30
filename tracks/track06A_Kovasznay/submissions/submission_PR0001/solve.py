"""Vanilla PINN (submission_PR0001): standard tanh MLP (x, y) -> 4 x 64 -> (u, v, p), Adam on the mean-squared
strong-form residual + the Dirichlet velocity loss, 4096 collocation points resampled uniformly every step,
stop after 1000 s; `query` evaluates the network on the output points.
"""

from speedrun.submissions.submission_PR0001.pinn import MLP, box_boundary, partials, predict_grid, train, uniform


def solve(problem):
    dev, f, prm = problem.device, problem.fns, problem.params
    nu = prm["nu"]
    (x0, x1), (y0, y1) = problem.domain["x"], problem.domain["y"]
    lo, hi = [x0, y0], [x1, y1]
    interior = lambda n: uniform(n, lo, hi, dev)  # noqa: E731
    boundary = lambda n: box_boundary(n, lo, hi, [0, 1], dev).detach()  # noqa: E731
    model = MLP(lo, hi, 3).to(dev)

    def loss():
        X = interior(4096)
        Y = model(X)
        u, v, p = Y[:, 0], Y[:, 1], Y[:, 2]
        ux, uy, vx, vy = partials(u, X, 0), partials(u, X, 1), partials(v, X, 0), partials(v, X, 1)
        lap_u = partials(ux, X, 0) + partials(uy, X, 1)
        lap_v = partials(vx, X, 0) + partials(vy, X, 1)
        ru = u * ux + v * uy + partials(p, X, 0) - nu * lap_u
        rv = u * vx + v * vy + partials(p, X, 1) - nu * lap_v
        Xb = boundary(1024)
        Yb = model(Xb)
        bc = (((Yb[:, 0] - f["u_bc"](Xb[:, 0], Xb[:, 1])) ** 2).mean()
              + ((Yb[:, 1] - f["v_bc"](Xb[:, 0], Xb[:, 1])) ** 2).mean())
        return (ru ** 2).mean() + (rv ** 2).mean() + ((ux + vy) ** 2).mean() + bc

    train(model, loss)
    return model


def query(model, problem):
    """Projection of the trained network onto the evaluation points (timed separately, <= 5 s)."""
    return predict_grid(model, [problem.axes["x"], problem.axes["y"]], fields=(0, 1), dtype=problem.dtype)
