"""Where does a submission actually compute? Counts every aten op of one warmup + solve + query by the device of its
tensors, so a run on a CUDA problem whose work silently happens on the CPU is reported as such.

Used by the harness (an extra, untimed `solve` after scoring); standalone:
    uv run python -m speedrun.devicecheck --track track3 [--submission f.py]
Work done outside torch (numpy, scipy, python loops over floats) is invisible here; it shows up as wall
time that the CUDA profiler does not account for, which `speedrun.run` does not try to attribute.
"""
from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

SMALL = 64            # tensors with fewer elements (scalars, tiny metadata) are ignored


class _Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.work = {"cuda": 0, "cpu": 0}
        self.ops = {"cuda": 0, "cpu": 0}
        self.cpu_ops: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        ts = [t for t in tree_flatten((args, kwargs, out))[0]
              if isinstance(t, torch.Tensor) and t.numel() >= SMALL and t.device.type in ("cuda", "cpu")]
        if ts:                                          # (meta tensors = shape inference, not compute)
            dev = "cpu" if any(t.device.type == "cpu" for t in ts) else "cuda"
            self.ops[dev] += 1
            self.work[dev] += max(t.numel() for t in ts)
            if dev == "cpu":
                name = str(func.overloadpacket.__name__)
                self.cpu_ops[name] = self.cpu_ops.get(name, 0) + 1
        return out


def check(run, problem) -> dict:
    """Run run(problem) (warmup + solve + query) once under op counting. `gpu_fraction` = share of tensor work
    on CUDA."""
    with _Count() as c:
        run(problem)
    total = sum(c.work.values())
    frac = c.work["cuda"] / total if total else 1.0
    top = sorted(c.cpu_ops.items(), key=lambda kv: -kv[1])[:8]
    return {"gpu_fraction": frac, "ops_cuda": c.ops["cuda"], "ops_cpu": c.ops["cpu"], "cpu_ops_top": dict(top)}


def classify(report: dict, gpu_name: str, cpu_name: str) -> str:
    """Hardware label: the GPU only if (essentially) all tensor work ran there."""
    f = report["gpu_fraction"]
    if f >= 0.99:
        return gpu_name
    if f <= 0.01:
        return cpu_name
    return f"CPU+GPU ({cpu_name} + {gpu_name}; {100 * f:.0f}% of tensor work on GPU)"


def main():
    import argparse

    from .run import _import_file, run_submission
    from .tracks import load_problem, resolve, track_dir
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--submission", default=None)
    a = ap.parse_args()
    t = resolve(a.track)
    mod = _import_file(Path(a.submission) if a.submission else track_dir(t) / "baseline.py")
    p = load_problem(t, device="cuda")
    print(t, check(lambda q: run_submission(mod, q), p))


if __name__ == "__main__":
    main()
