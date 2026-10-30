"""Vanilla PINN (submission_PR0001): standard tanh MLP (t, x) -> 4 x 64 -> u, Adam on the mean-squared strong-
form residual + initial-condition and periodicity (u, u_x) losses, 4096 collocation points resampled uniformly
every step, stop after 1000 s; `query` evaluates the network on the output points.
"""
import torch

from speedrun.submissions.submission_PR0001.pinn import MLP, PINN, partials, predict_grid, uniform


def warmup(problem):
    """Untimed: build the network and the loss, run one gradient evaluation (no training)."""
    dev, f, prm = problem.device, problem.fns, problem.params
    (t0, t1), (x0, x1) = problem.domain["t"], problem.domain["x"]
    lo, hi = [t0, x0], [t1, x1]
    model = MLP(lo, hi, 1).to(dev)

    def loss():
        X = uniform(4096, lo, hi, dev)
        u = model(X)[:, 0]
        ux = partials(u, X, 1)
        r = partials(u, X, 0) + u * ux - prm["nu"] * partials(ux, X, 1)
        Xi = uniform(1024, [t0, x0], [t0, x1], dev)
        ic = ((model(Xi)[:, 0] - f["u0"](Xi[:, 1])) ** 2).mean()
        tb = uniform(1024, [t0], [t1], dev).detach()                   # periodicity: u, u_x at x0 and x1
        A = torch.cat([tb, torch.full_like(tb, x0)], 1).requires_grad_(True)
        B = torch.cat([tb, torch.full_like(tb, x1)], 1).requires_grad_(True)
        ua, ub = model(A)[:, 0], model(B)[:, 0]
        per = ((ua - ub) ** 2).mean() + ((partials(ua, A, 1) - partials(ub, B, 1)) ** 2).mean()
        return (r ** 2).mean() + ic + per

    return PINN(model, loss).compile()


def solve(problem, solver):
    """Timed: train for the 1000 s budget."""
    return solver.train()


def query(problem, solver):
    """Timed separately (<= 5 s): evaluate the trained network on the evaluation points."""
    model = solver.model
    return predict_grid(model, [problem.axes["t"], problem.axes["x"]], dtype=problem.dtype)
