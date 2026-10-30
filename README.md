# NeuralPDESpeedrun.py

A speedrun for **data-free** PDE solving. Solve one PDE problem as fast as possible on a single H100,
using only the governing equations (equations, BCs/ICs, domain), to

- **RelL2 ≤ 5e-3** and **L∞ ≤ 1e-2** against a closed-form or high-resolution classical reference,
  evaluated at fixed collocation points.

Architecture, ansatz, loss, optimizer, and discretization are all free. Hybrids with classical solvers
are allowed. The question: **can neural or hybrid methods come close to, or beat, traditional solvers?**

- Rules, tracks, status, and plan: [SPEEDRUN.md](SPEEDRUN.md)
- Research notes: [IDEAS.md](IDEAS.md)

## Quickstart

```bash
uv sync
uv run python -m speedrun.make_reference --track all            # build reference solutions + figures
uv run python -m speedrun.run --track track1                     # time + score the track's baseline
uv run python -m speedrun.run --track track1 --submission my_solver.py --out result.json
```

A submission is a python file with

```python
def solve(problem) -> torch.Tensor:   # shape problem.out_shape, values at problem.axes (tensor grid)
    ...
def warmup(problem) -> None:          # optional, untimed (compile / autotune only)
    ...
```

`problem` (`speedrun.Problem`) contains the statement, parameters, IC/BC/forcing/coefficient callables,
and the evaluation axes. It contains nothing about the solution.

## Tracks

| track | problem |
|---|---|
| 1 | 3rd-order ODE (NeuralPDE.jl) |
| 2A / 2B | 1D wave / damped wave (NeuralPDE.jl) |
| 3 | 2D high-contrast diffusion |
| 4A / 4B | 2D Helmholtz (heterogeneous) / advection-diffusion, Pe ~ 1e3 |
| 5A / 5B / 5C | Burgers 1D / Burgers 2D / Kuramoto-Sivashinsky (SNF-ROM, arXiv:2405.14890 §6.3–6.5) |
| 6 | Kovasznay flow |
| 7A / 7B | lid-driven cavity, Re = 100 / 1000 (regularized lid) |
| 8 | steady Stokes cavity |

Each `tracks/trackK/` has `problem.py`, `reference.py`, `baseline.py` (record #0, a classical solver),
`records.json`, and `figures/`.

No Julia anywhere. All solvers, reference and baseline alike, are PyTorch.
