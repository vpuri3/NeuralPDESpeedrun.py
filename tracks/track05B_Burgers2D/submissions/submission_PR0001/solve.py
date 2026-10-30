"""Vanilla PINN (submission_PR0001): standard tanh MLP (t, x, y) -> 4 x 64 -> (u, v), Adam on the mean-squared
strong-form residual + initial-condition and periodicity (value + first derivative) losses, 4096 collocation
points resampled uniformly every step, stop after 1000 s; `query` evaluates the network on the output points.
"""

from speedrun.submissions.submission_PR0001.pinn import MLP, partials, predict_grid, train, uniform


def _periodic(model, lo, hi, dev, n=1024):
    """u and its first derivatives match on opposite faces x = x0 / x1 and y = y0 / y1."""
    out = 0.0
    for d in (1, 2):
        P = uniform(n, lo, hi, dev).detach()
        A, B = P.clone(), P.clone()
        A[:, d], B[:, d] = lo[d], hi[d]
        A.requires_grad_(True), B.requires_grad_(True)
        ya, yb = model(A), model(B)
        for k in range(2):
            out = out + ((ya[:, k] - yb[:, k]) ** 2).mean()
            out = out + ((partials(ya[:, k], A, d) - partials(yb[:, k], B, d)) ** 2).mean()
    return out


def solve(problem):
    dev, f, prm = problem.device, problem.fns, problem.params
    (t0, t1), (x0, x1), (y0, y1) = problem.domain["t"], problem.domain["x"], problem.domain["y"]
    lo, hi = [t0, x0, y0], [t1, x1, y1]
    nu = prm["nu"]
    model = MLP(lo, hi, 2).to(dev)

    def loss():
        X = uniform(4096, lo, hi, dev)
        Y = model(X)
        u, v = Y[:, 0], Y[:, 1]
        ux, uy, vx, vy = partials(u, X, 1), partials(u, X, 2), partials(v, X, 1), partials(v, X, 2)
        lap_u = partials(ux, X, 1) + partials(uy, X, 2)
        lap_v = partials(vx, X, 1) + partials(vy, X, 2)
        ru = partials(u, X, 0) + u * ux + v * uy - nu * lap_u
        rv = partials(v, X, 0) + u * vx + v * vy - nu * lap_v
        res = (ru ** 2).mean() + (rv ** 2).mean()
        Xi = uniform(1024, [t0, x0, y0], [t0, x1, y1], dev)
        Yi = model(Xi)
        ic = (((Yi[:, 0] - f["u0"](Xi[:, 1], Xi[:, 2])) ** 2).mean()
              + ((Yi[:, 1] - f["v0"](Xi[:, 1], Xi[:, 2])) ** 2).mean())
        return res + ic + _periodic(model, lo, hi, dev)

    train(model, loss)
    return model


def query(model, problem):
    """Projection of the trained network onto the evaluation points (timed separately, <= 5 s)."""
    ax = [problem.axes["t"], problem.axes["x"], problem.axes["y"]]
    return predict_grid(model, ax, fields=(0, 1), dtype=problem.dtype, chunk=1 << 20)
