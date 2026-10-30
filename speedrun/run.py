"""Official timing + scoring harness.

    uv run python -m speedrun.run --track 03                       # runs the track's baseline.py
    uv run python -m speedrun.run --track 03 \
        --submission tracks/track03_HighContrastDiffusion/submissions/submission_PR0042

A submission is a directory `tracks/<track>/submissions/submission_PR<NNNN>/` (or a single .py file while
developing) whose `solve.py` defines

    def warmup(problem) -> solver            # untimed: build / compile the solver object  (OPTIONAL)
    def solve(problem, solver) -> solver     # timed: train / iterate / solve               (REQUIRED)
    def query(problem, solver) -> Tensor     # timed separately, <= 5 s: evaluate at problem.axes,
                                             # shape problem.out_shape                      (OPTIONAL)

Without `warmup`, `solve` receives solver=None. Without `query`, `solve` returns the output tensor itself
(query time 0). Learning need not happen on the evaluation points; `query` is the projection onto them. The
score time is solve + query, and a query slower than QUERY_LIMIT_S fails the run. `warmup` must not train on
the problem or compute any part of its solution.

Protocol (rules: README.md):
  1. seed everything (seed 0), build the Problem on the target device
  2. init CUDA libraries (problem-independent), solver = warmup(problem) if present (untimed)
  3. synchronize, clock `solve(problem, solver)`, synchronize; then clock `query(problem, solver)` the same way
  4. only THEN is the reference loaded and the returned tensor scored
  5. (untimed) device check: one more warmup + solve + query under op counting; the reported hardware is the GPU only if
     essentially all tensor work ran on it (speedrun/devicecheck.py). The env var SPEEDRUN_DEVICE_CHECK=1 is
     set during this run, so long iterative solvers may shorten it (it is not scored).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import platform
import resource
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


def _sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def run_submission(mod, problem):
    """warmup -> solve -> query, untimed (used by the GPU check)."""
    solver = mod.warmup(problem) if hasattr(mod, "warmup") else None
    solver = mod.solve(problem, solver)
    return mod.query(problem, solver) if hasattr(mod, "query") else solver


def submission_file(path: Path) -> Path:
    """A submission directory (submission_PR<NNNN>/solve.py) or a single .py file."""
    return path / "solve.py" if path.is_dir() else path


def run(track: str, submission: str | None = None, device: str | None = None, seed: int = 0,
        out: str | None = None, quiet: bool = False, device_check: bool = True) -> dict:
    track = resolve(track)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sub_path = submission_file(Path(submission)) if submission else track_dir(track) / "baseline.py"
    sub_path = sub_path.resolve()

    torch.manual_seed(seed)
    problem = load_problem(track, device=device)
    mod = _import_file(sub_path)
    _init_device_libs(device)

    solver = mod.warmup(problem) if hasattr(mod, "warmup") else None
    _sync(device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _sync(device)
    t0 = time.perf_counter()
    solver = mod.solve(problem, solver)
    _sync(device)
    solve_s = time.perf_counter() - t0
    query_s = 0.0
    u = solver
    if hasattr(mod, "query"):
        t1 = time.perf_counter()
        u = mod.query(problem, solver)
        _sync(device)
        query_s = time.perf_counter() - t1
    elapsed = solve_s + query_s

    u = u.detach().to("cpu") if hasattr(u, "detach") else u
    ref = load_reference(track)  # loaded only after the timed solve
    s = score(u, ref, problem.linf_mask(), problem.rel_l2_tol, problem.linf_tol)
    query_ok = query_s <= QUERY_LIMIT_S

    if device.startswith("cuda") and torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 2**30
        mem_kind = "cuda_peak_allocated_gb"
    else:
        mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB -> GiB (linux)
        mem_kind = "host_peak_rss_gb"

    hardware, dev_report = hardware_string(device), None
    if device_check and device.startswith("cuda") and torch.cuda.is_available():
        from .devicecheck import check, classify
        os.environ["SPEEDRUN_DEVICE_CHECK"] = "1"   # submissions may shorten this untimed, unscored run
        try:
            dev_report = check(lambda p: run_submission(mod, p), problem)
        finally:
            del os.environ["SPEEDRUN_DEVICE_CHECK"]
        hardware = classify(dev_report, hardware, cpu_string())

    result = {
        "track": track,
        "submission": str(sub_path.relative_to(ROOT)) if sub_path.is_relative_to(ROOT) else str(sub_path),
        "time_s": elapsed,
        "solve_time_s": solve_s,
        "query_time_s": query_s,
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
    args = ap.parse_args()
    r = run(args.track, args.submission, args.device, args.seed, args.out, device_check=not args.no_device_check)
    sys.exit(0 if r["passed"] else 1)


if __name__ == "__main__":
    main()
