"""Vanilla PINN (draft, not a submission): standard tanh MLP (t, x) -> 4 x 64 -> u, Adam on the mean-squared strong-
form residual + initial (u, u_t) and Dirichlet losses, 4096 collocation points resampled uniformly every step,
stop after 1000 s; `query` evaluates the network on the output points.
"""
import torch

from speedrun.submissions.draft_vanilla_pinn.pinn import MLP, PINN, box_boundary, partials, predict_grid, uniform


def warmup(problem):
    """Untimed: build the network and the loss, run one gradient evaluation (no training)."""
    dev, f, prm = problem.device, problem.fns, problem.params
    (t0, t1), (x0, x1) = problem.domain["t"], problem.domain["x"]
    lo, hi = [t0, x0], [t1, x1]
    model = MLP(lo, hi, 1).to(dev)

    def loss():
        X = uniform(4096, lo, hi, dev)
        u = model(X)[:, 0]
        r = partials(u, X, 0, 0) - prm["c"] ** 2 * partials(u, X, 1, 1) + prm["gamma"] * partials(u, X, 0)
        Xi = uniform(1024, [t0, x0], [t0, x1], dev)                     # t = t0
        ui = model(Xi)[:, 0]
        ic = ((ui - f["u0"](Xi[:, 1])) ** 2).mean() + ((partials(ui, Xi, 0) - f["ut0"](Xi[:, 1])) ** 2).mean()
        Xb = box_boundary(1024, lo, hi, [1], dev)                       # x = x0 or x1
        ub = model(Xb)[:, 0]
        g = torch.where(Xb[:, 1] < 0.5 * (x0 + x1), f["bc_left"](Xb[:, 0]), f["bc_right"](Xb[:, 0]))
        return (r ** 2).mean() + ic + ((ub - g) ** 2).mean()

    return PINN(model, loss).compile()


def solve(problem, solver):
    """Timed: train for the 1000 s budget."""
    return solver.train()


def query(problem, solver):
    """Timed separately (<= 5 s): evaluate the trained network on the evaluation points."""
    model = solver.model
    return predict_grid(model, [problem.axes["t"], problem.axes["x"]], dtype=problem.dtype)
