"""Vanilla PINN (draft, not a submission): standard tanh MLP (x, y) -> 4 x 64 -> u, Adam on the mean-squared strong-
form residual + the Dirichlet loss, 4096 collocation points resampled uniformly every step, stop after 1000 s;
`query` evaluates the network on the output points.
"""

from speedrun.submissions.draft_vanilla_pinn.pinn import MLP, PINN, box_boundary, partials, predict_grid, uniform


def warmup(problem):
    """Untimed: build the network and the loss, run one gradient evaluation (no training)."""
    dev, f, prm = problem.device, problem.fns, problem.params
    (x0, x1), (y0, y1) = problem.domain["x"], problem.domain["y"]
    lo, hi = [x0, y0], [x1, y1]
    model = MLP(lo, hi, 1).to(dev)

    def loss():
        X = uniform(4096, lo, hi, dev)
        u = model(X)[:, 0]
        x, y = X[:, 0], X[:, 1]
        lap = partials(u, X, 0, 0) + partials(u, X, 1, 1)
        r = (-prm["eps"] * lap + prm["b"][0] * partials(u, X, 0) + prm["b"][1] * partials(u, X, 1)
             - f["forcing"](x, y))
        Xb = box_boundary(1024, lo, hi, [0, 1], dev)
        return (r ** 2).mean() + ((model(Xb)[:, 0] - f["bc"](Xb[:, 0], Xb[:, 1])) ** 2).mean()

    return PINN(model, loss).compile()


def solve(problem, solver):
    """Timed: train for the 1000 s budget."""
    return solver.train()


def query(problem, solver):
    """Timed separately (<= 5 s): evaluate the trained network on the evaluation points."""
    model = solver.model
    return predict_grid(model, [problem.axes["x"], problem.axes["y"]], dtype=problem.dtype)
