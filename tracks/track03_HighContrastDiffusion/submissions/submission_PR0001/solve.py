"""Vanilla PINN (submission_PR0001): standard tanh MLP (x, y) -> 4 x 64 -> u, Adam on the mean-squared strong-
form residual + the Dirichlet loss, 4096 collocation points resampled uniformly every step, stop after 1000 s;
`query` evaluates the network on the output points. Strong form -a Lap u = f inside each material (a is
piecewise constant); the flux-continuity conditions at the inclusion edges are not imposed separately, as in a
vanilla PINN.
"""

from speedrun.submissions.submission_PR0001.pinn import MLP, box_boundary, partials, predict_grid, train, uniform


def solve(problem):
    dev, f = problem.device, problem.fns
    (x0, x1), (y0, y1) = problem.domain["x"], problem.domain["y"]
    lo, hi = [x0, y0], [x1, y1]
    model = MLP(lo, hi, 1).to(dev)

    def loss():
        X = uniform(4096, lo, hi, dev)
        u = model(X)[:, 0]
        x, y = X[:, 0], X[:, 1]
        lap = partials(u, X, 0, 0) + partials(u, X, 1, 1)
        r = -f["a"](x, y) * lap - f["forcing"](x, y)
        Xb = box_boundary(1024, lo, hi, [0, 1], dev)
        return (r ** 2).mean() + ((model(Xb)[:, 0] - f["bc"](Xb[:, 0], Xb[:, 1])) ** 2).mean()

    train(model, loss)
    return model


def query(model, problem):
    """Projection of the trained network onto the evaluation points (timed separately, <= 5 s)."""
    return predict_grid(model, [problem.axes["x"], problem.axes["y"]], dtype=problem.dtype)
