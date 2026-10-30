"""Official timing + scoring harness.

    uv run python -m speedrun.run --track 07                       # runs the track's baseline.py
    uv run python -m speedrun.run --track 07 \
        --submission tracks/track07_2D_HighContrastDiffusion/submissions/submission_PR0042

A submission is a directory `tracks/<track>/submissions/submission_PR<NNNN>/` (or a single .py file while
developing) whose `solve.py` defines

    def setup(problem) -> state                      # untimed: build / compile, no training     (OPTIONAL)
    def solve(problem, state, checkpoint) -> state   # timed: train / iterate, as long as it likes
    def query(problem, state) -> array               # timed separately, <= 5 s: the solution at problem.axes,
                                                     # shape problem.out_shape (torch, JAX or numpy)

The harness owns the clock and the budget. `solve` may loop forever; every so often (e.g. every 10-100
iterations) it calls `checkpoint(state)` with a state `query` can evaluate. Once the track's wall-clock budget
(problem.budget_s) is used up, the harness interrupts `solve` (checkpoint raises BudgetExceeded; a SIGALRM
ALARM_GRACE_S later is the backstop for long stretches without checkpoints) and queries the last checkpointed
state. `solve` may also return early with its final state. checkpoint() only keeps a reference: a checkpointed
state must not be modified in place afterwards (JAX arrays are immutable; snapshot torch tensors). One-shot
solvers may define `solve(problem, state)` without the checkpoint argument. If the final state is a tensor of
shape problem.out_shape, it is the output and `query` is not called (how the classical baselines work). Without
`setup`, `solve` receives state = None. The score time is solve + query; a query slower than QUERY_LIMIT_S
fails the run. `setup` must not train on the problem or compute any part of its solution.

Protocol (rules: README.md):
  1. seed everything (seed 0), build the Problem on the target device
  2. init CUDA libraries (problem-independent), state = setup(problem) if present (untimed)
  3. synchronize, clock `solve` (until it returns or the budget is used), synchronize; then clock `query` the same
     way, on the returned or last checkpointed state
  4. only THEN is the reference loaded and the returned array scored
  5. (untimed) device check: setup + solve with a DEVICE_CHECK_BUDGET_S budget + query once more under op counting; the
     reported hardware is the GPU only if essentially all tensor work ran on it (speedrun/devicecheck.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import inspect
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

# Load every CUDA kernel module at context creation. Under torch's default LAZY loading, the first
# launch of each kernel pays a one-time load (~0.1 s total on small tracks) inside the timed region.
# Read at CUDA context creation, so it only has to be set before the first CUDA call.
os.environ["CUDA_MODULE_LOADING"] = "EAGER"

import torch  # noqa: E402

from .metrics import score  # noqa: E402
from .tracks import ROOT, load_problem, load_reference, resolve, track_dir  # noqa: E402


QUERY_LIMIT_S = 5.0
DEVICE_CHECK_BUDGET_S = 10.0
ALARM_GRACE_S = 1.0     # the hard interrupt (SIGALRM) comes this long after the budget


class BudgetExceeded(BaseException):
    """Raised inside `solve` once the budget is used up. A BaseException, so `except Exception` in a submission
    does not swallow it."""


class Checkpoint:
    """The `checkpoint` callable handed to `solve`: checkpoint(state) makes `state` the current answer and raises
    BudgetExceeded once the budget is used up."""

    def __init__(self, budget_s: float):
        self.deadline = time.perf_counter() + budget_s
        self.state, self.count = None, 0

    def __call__(self, state):
        self.state, self.count = state, self.count + 1
        if time.perf_counter() >= self.deadline:
            raise BudgetExceeded


def _jax_arrays(tree):
    import jax
    return [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]


def _sync(device: str, tree=None):
    """Wait for all queued device work: torch's, and JAX's (asynchronous dispatch) if JAX is in use. JAX work is
    waited for through the arrays of `tree` if it holds any, else through every live JAX array."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    if "jax" in sys.modules:
        import jax
        arrays = _jax_arrays(tree) if tree is not None else []
        jax.block_until_ready(arrays or jax.live_arrays())


def _init_device_libs(device: str):
    """Problem-independent init of lazily loaded CUDA libraries (cuBLAS, cuSOLVER, cuFFT, ...),
    so their one-time setup cost is not charged to whichever submission touches them first."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return
    for dtype in (torch.float32, torch.float64):
        a = torch.eye(8, device=device, dtype=dtype) + 0.1 * torch.rand(8, 8, device=device, dtype=dtype)
        b = torch.ones(8, 1, device=device, dtype=dtype)
        _ = a @ b, torch.linalg.solve(a, b), torch.linalg.lu_factor(a), torch.linalg.cholesky(a @ a.mT)
        _ = torch.linalg.qr(a), torch.linalg.eigh(a @ a.mT), torch.linalg.svd(a)
        _ = torch.fft.rfft(b[:, 0]), torch.fft.fft2(a.to(torch.complex128 if dtype == torch.float64
                                                         else torch.complex64))
        _ = torch.linalg.solve(a.to(torch.complex128), b.to(torch.complex128))
    _sync(device)


def _import_file(path: Path):
    sys.dont_write_bytecode = True  # never run a stale cached submission
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def cpu_string() -> str:
    name = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"CPU ({name})"


def hardware_string(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return cpu_string()


def is_output(x, problem) -> bool:
    """Is x the output grid itself (classical solvers) rather than a solver state?"""
    return hasattr(x, "shape") and not isinstance(x, (dict, list, tuple)) and tuple(x.shape) == tuple(problem.out_shape)


def needs_query(mod, state, problem) -> bool:
    if is_output(state, problem):
        return False
    if not hasattr(mod, "query"):
        shape = tuple(state.shape) if hasattr(state, "shape") else type(state).__name__
        raise TypeError(f"the solve ended with {shape}, not an array of problem.out_shape = {problem.out_shape}, "
                        "and the submission defines no query(problem, state)")
    return True


def _takes_checkpoint(fn) -> bool:
    ps = list(inspect.signature(fn).parameters.values())
    return len(ps) >= 3 or any(p.kind == p.VAR_POSITIONAL for p in ps)


def _alarm(signum, frame):
    raise BudgetExceeded


def timed_solve(mod, problem, state, budget_s: float, device: str) -> tuple:
    """The timed part of a run: `solve` until it returns or the budget is used up. Returns (state, number of
    checkpoints, stop reason)."""
    ck = Checkpoint(budget_s)
    args = (problem, state, ck) if _takes_checkpoint(mod.solve) else (problem, state)
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s + ALARM_GRACE_S)
    try:
        state, stop = mod.solve(*args), "done"
    except BudgetExceeded:
        if ck.count == 0:
            raise RuntimeError(f"budget of {budget_s:g} s used up before the first checkpoint") from None
        state, stop = ck.state, "budget"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    _sync(device, state)
    return state, ck.count, stop


def run_submission(mod, problem, budget_s: float = DEVICE_CHECK_BUDGET_S):
    """setup -> solve (short budget) -> query, untimed (used by the device check)."""
    device = problem.device
    state = mod.setup(problem) if hasattr(mod, "setup") else None
    state, _, _ = timed_solve(mod, problem, state, budget_s, device)
    return mod.query(problem, state) if needs_query(mod, state, problem) else state


def to_torch(u) -> torch.Tensor:
    """The output as a float64 CPU tensor, whatever array library produced it."""
    if isinstance(u, torch.Tensor):
        return u.detach().to("cpu", torch.float64)
    import numpy as np
    return torch.from_numpy(np.asarray(u, dtype=np.float64))


def submission_file(path: Path) -> Path:
    """A submission directory (submission_PR<NNNN>/solve.py) or a single .py file."""
    return path / "solve.py" if path.is_dir() else path


def run(track: str, submission: str | None = None, device: str | None = None, seed: int = 0,
        out: str | None = None, quiet: bool = False, device_check: bool = True,
        budget_s: float | None = None) -> dict:
    """budget_s overrides the track's budget (development runs only; official runs use problem.budget_s)."""
    track = resolve(track)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sub_path = submission_file(Path(submission)) if submission else track_dir(track) / "baseline.py"
    sub_path = sub_path.resolve()

    torch.manual_seed(seed)
    problem = load_problem(track, device=device)
    mod = _import_file(sub_path)
    _init_device_libs(device)

    state = mod.setup(problem) if hasattr(mod, "setup") else None
    _sync(device, state)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    budget_s = problem.budget_s if budget_s is None else budget_s
    _sync(device)
    t0 = time.perf_counter()
    state, checkpoints, stop = timed_solve(mod, problem, state, budget_s, device)
    solve_s = time.perf_counter() - t0
    query_s = 0.0
    u = state
    if needs_query(mod, state, problem):
        t1 = time.perf_counter()
        u = mod.query(problem, state)
        _sync(device, u)
        query_s = time.perf_counter() - t1
    elapsed = solve_s + query_s
    del state

    u = to_torch(u)
    ref = load_reference(track)  # loaded only after the timed solve
    s = score(u, ref, problem.linf_mask(), problem.rel_l2_tol, problem.linf_tol)
    query_ok = query_s <= QUERY_LIMIT_S

    if device.startswith("cuda") and torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 2**30
        if "jax" in sys.modules:
            import jax
            mem_gb += jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2**30
        mem_kind = "cuda_peak_allocated_gb"
    else:
        mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB -> GiB (linux)
        mem_kind = "host_peak_rss_gb"

    hardware, dev_report = hardware_string(device), None
    if device_check and device.startswith("cuda") and torch.cuda.is_available():
        from .devicecheck import check, classify
        dev_report = check(lambda p: run_submission(mod, p), problem)
        hardware = classify(dev_report, hardware, cpu_string())

    result = {
        "track": track,
        "submission": str(sub_path.relative_to(ROOT)) if sub_path.is_relative_to(ROOT) else str(sub_path),
        "time_s": elapsed,
        "solve_time_s": solve_s,
        "query_time_s": query_s,
        "checkpoints": checkpoints,
        "stop": stop,
        "budget_s": budget_s,
        "rel_l2": s["rel_l2"],
        "linf": s["linf"],
        "accurate": s["passed"],
        "query_within_limit": query_ok,
        "passed": s["passed"] and query_ok,
        "tolerances": {"rel_l2": problem.rel_l2_tol, "linf": problem.linf_tol},
        "linf_margin": problem.linf_margin,
        "device": device,
        "hardware": hardware,
        "device_check": dev_report,
        mem_kind: mem_gb,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "seed": seed,
        "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if not quiet:
        status = "PASS" if result["passed"] else "FAIL"
        q = f" (solve {solve_s:.4f} + query {query_s:.4f}{'' if query_ok else ' > limit'})" if query_s else ""
        q += f" [{checkpoints} checkpoints, stopped: {stop}]" if checkpoints or stop != "done" else ""
        print(f"[{track}] {status}  time={elapsed:.4f}s{q}  relL2={s['rel_l2']:.3e} (<= {problem.rel_l2_tol:g})  "
              f"Linf={s['linf']:.3e} (<= {problem.linf_tol:g})  {mem_kind}={mem_gb:.3f}  on {result['hardware']}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True)
    ap.add_argument("--submission", default=None,
                    help="submission directory or .py file (default: the track's baseline.py)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write result json here")
    ap.add_argument("--no-device-check", action="store_true", help="skip the untimed op-device check")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="override the track's wall-clock budget (development only, not for records)")
    args = ap.parse_args()
    r = run(args.track, args.submission, args.device, args.seed, args.out, device_check=not args.no_device_check,
            budget_s=args.budget_s)
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
