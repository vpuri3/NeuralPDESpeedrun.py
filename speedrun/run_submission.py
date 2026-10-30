"""Run one submission on every track it covers, one job per GPU at a time, then update records and figures.

    uv run python -m speedrun.run_submission --submission submission_PR0001 --gpus 0,1,2,3 \
        --name "PirateNet + SOAP" --kind neural --pr 1
    uv run python -m speedrun.run_submission --submission submission_PR0001 --tracks 03,05 --gpus 0

For each track with tracks/<track>/submissions/<submission>/solve.py this runs the official harness
(`python -m speedrun.run`), writing <submission dir>/result.json and run.log. With --name, the results are added
to each track's records.json, and the tracker plots, per-track READMEs and the main README table are regenerated.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

from .tracks import ROOT, TRACKS, resolve, track_dir


def tracks_with(submission: str, only: list[str] | None = None) -> list[str]:
    pool = [resolve(t) for t in only] if only else TRACKS
    return [t for t in pool if (track_dir(t) / "submissions" / submission / "solve.py").exists()]


def _run(track: str, submission: str, gpu: str, extra: list[str]) -> dict:
    d = track_dir(track) / "submissions" / submission
    cmd = [sys.executable, "-m", "speedrun.run", "--track", track, "--submission", str(d),
           "--out", str(d / "result.json"), *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    t0 = time.time()
    with open(d / "run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    return {"track": track, "gpu": gpu, "rc": rc, "wall_s": time.time() - t0, **res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True, help="directory name, e.g. submission_PR0001")
    ap.add_argument("--tracks", default=None, help="comma-separated track keys (default: all it covers)")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids; one job per GPU at a time")
    ap.add_argument("--name", default=None, help="record name; if given, results are added to records.json")
    ap.add_argument("--kind", default="neural", choices=["classical", "neural", "hybrid"])
    ap.add_argument("--author", default="maintainers")
    ap.add_argument("--pr", type=int, default=None)
    ap.add_argument("--no-device-check", action="store_true")
    a = ap.parse_args()

    todo = tracks_with(a.submission, a.tracks.split(",") if a.tracks else None)
    if not todo:
        sys.exit(f"no tracks have tracks/<track>/submissions/{a.submission}/solve.py")
    gpus = a.gpus.split(",")
    extra = ["--no-device-check"] if a.no_device_check else []
    print(f"{a.submission}: {len(todo)} tracks on GPUs {gpus}")
    jobs: queue.Queue = queue.Queue()
    for t in todo:
        jobs.put(t)
    results, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                t = jobs.get_nowait()
            except queue.Empty:
                return
            r = _run(t, a.submission, gpu, extra)
            with lock:
                results.append(r)
                ok = "PASS" if r.get("passed") else ("FAIL" if r.get("rc") in (0, 1) else f"ERROR rc={r['rc']}")
                rel = f"{r['rel_l2']:.2e}" if "rel_l2" in r else "-"
                tm = f"{r['time_s']:.1f}s" if "time_s" in r else "-"
                print(f"  [{t}] {ok}  time={tm}  relL2={rel}  (GPU {gpu})", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    if a.name:
        from .records import add_entry, plot_accuracy, plot_activity, update_main_readme, write_readme
        for r in sorted(results, key=lambda r: TRACKS.index(r["track"])):
            res = track_dir(r["track"]) / "submissions" / a.submission / "result.json"
            if res.exists():
                add_entry(r["track"], str(res), a.name, a.kind, a.author, a.pr)
                plot_accuracy(r["track"])
                plot_activity(r["track"])
                write_readme(r["track"])
        update_main_readme()
        print("records, figures and READMEs updated")


if __name__ == "__main__":
    main()
