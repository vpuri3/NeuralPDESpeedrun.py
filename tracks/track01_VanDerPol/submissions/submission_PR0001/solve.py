"""Vanilla PINN (submission_PR0001): standard tanh MLP t -> 4 x 64 -> y, Adam on the mean-squared strong-form
residual + the two initial conditions, 4096 collocation points resampled uniformly every step, stop after 1000 s;
`query` evaluates the network on the output points.
"""
import torch

from speedrun.submissions.submission_PR0001.pinn import MLP, PINN, partials, predict_grid, uniform


def warmup(problem):
    """Untimed: build the network and the loss, run one gradient evaluation (no training)."""
    dev, f, mu = problem.device, problem.fns, problem.params["mu"]
    t0, t1 = problem.domain["t"]
    model = MLP([t0], [t1], 1).to(dev)
    ti = torch.tensor([[t0]], device=dev, requires_grad=True)

    def loss():
        t = uniform(4096, [t0], [t1], dev)
        y = model(t)[:, 0]
        dy = partials(y, t, 0)
        r = partials(dy, t, 0) - mu * (1 - y**2) * dy + y
        yi = model(ti)[:, 0]
        ic = (yi[0] - f["y0"]) ** 2 + (partials(yi, ti, 0)[0] - f["dy0"]) ** 2
        return (r ** 2).mean() + ic

    return PINN(model, loss).compile()


def solve(problem, solver):
    """Timed: train for the 1000 s budget."""
    return solver.train()


def query(problem, solver):
    """Timed separately (<= 5 s): evaluate the trained network on the evaluation points."""
    return predict_grid(solver.model, [problem.axes["t"]], dtype=problem.dtype)
