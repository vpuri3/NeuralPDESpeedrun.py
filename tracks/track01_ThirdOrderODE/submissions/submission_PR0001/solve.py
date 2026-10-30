"""Vanilla PINN (submission_PR0001): standard tanh MLP 1 -> 4 x 64 -> 1, Adam on the mean-squared strong-form
residual + the three boundary conditions, 1024 collocation points resampled uniformly every step, stop after
1000 s; `query` evaluates the network on the output points.
"""
import torch

from speedrun.submissions.submission_PR0001.pinn import MLP, PINN, partials, predict_grid, uniform


def warmup(problem):
    """Untimed: build the network and the loss, run one gradient evaluation (no training)."""
    dev, f = problem.device, problem.fns
    lo, hi = problem.domain["x"]
    model = MLP([lo], [hi], 1).to(dev)
    xb = torch.tensor([[lo], [hi]], device=dev, requires_grad=True)

    def loss():
        x = uniform(1024, [lo], [hi], dev)
        u = model(x)[:, 0]
        r = partials(u, x, 0, 0, 0) - f["forcing"](x[:, 0])
        ub = model(xb)[:, 0]
        dub = partials(ub, xb, 0)
        bc = (ub[0] - f["u_left"]) ** 2 + (ub[1] - f["u_right"]) ** 2 + (dub[1] - f["du_right"]) ** 2
        return (r ** 2).mean() + bc

    return PINN(model, loss).compile()


def solve(problem, solver):
    """Timed: train for the 1000 s budget."""
    return solver.train()


def query(problem, solver):
    """Timed separately (<= 5 s): evaluate the trained network on the evaluation points."""
    model = solver.model
    return predict_grid(model, [problem.axes["x"]], dtype=problem.dtype)
