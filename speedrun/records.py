"""Leaderboards: read tracks/<t>/records.json, draw tracks/<t>/figures/activity.png, write
tracks/<t>/README.md, and seed record #0.

    uv run python -m speedrun.records --track all                 # redraw every activity plot
    uv run python -m speedrun.records --track 03 --seed-baseline tracks/track03_HighContrastDiffusion/logs/baseline.json

A record entry is
    {"date": "YYYY-MM-DD", "author": ..., "name": ...,
     "submission": "tracks/<t>/baseline.py" | "tracks/<t>/submissions/submission_PR<NNNN>",
     "time_s": ..., "rel_l2": ..., "linf": ..., "hardware": ..., "kind": "classical" | "neural" | "hybrid",
     "pr": <PR number or null>}
`time_s` is the official (maintainer, H100 80GB) time. Entries timed elsewhere say so in `hardware`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .tracks import TRACKS, resolve, track_dir

KIND_STYLE = {"classical": ("tab:gray", "s"), "neural": ("tab:blue", "o"), "hybrid": ("tab:purple", "D")}


def records_path(track: str) -> Path:
    return track_dir(track) / "records.json"


def load_records(track: str) -> list[dict]:
    p = records_path(track)
    return json.loads(p.read_text()) if p.exists() else []


def save_records(track: str, recs: list[dict]) -> None:
    recs = sorted(recs, key=lambda r: (r["date"], -r["time_s"]))
    records_path(track).write_text(json.dumps(recs, indent=2) + "\n")


def seed_baseline(track: str, result_json: str, date: str | None = None, author: str = "maintainers") -> dict:
    """Make the harness result of the track's baseline record #0 (replacing an existing baseline entry)."""
    track = resolve(track)
    r = json.loads(Path(result_json).read_text())
    assert r["track"] == track and r["passed"], f"{result_json}: not a passing {track} result"
    entry = {"date": date or r["date"][:10], "author": author, "name": "record #0: classical baseline",
             "submission": r["submission"], "time_s": r["time_s"], "rel_l2": r["rel_l2"], "linf": r["linf"],
             "hardware": r["hardware"], "kind": "classical", "pr": None}
    recs = [e for e in load_records(track) if e.get("submission") != r["submission"]]
    save_records(track, recs + [entry])
    return entry


def plot_activity(track: str) -> Path | None:
    track = resolve(track)
    recs = load_records(track)
    if not recs:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [dt.date.fromisoformat(r["date"]) for r in recs]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    for kind, (c, m) in KIND_STYLE.items():
        pts = [(d, r["time_s"]) for d, r in zip(dates, recs) if r.get("kind", "classical") == kind]
        if pts:
            ax.scatter(*zip(*pts), color=c, marker=m, label=kind, zorder=3)
    # step line through the running record (best time so far)
    order = sorted(range(len(recs)), key=lambda i: dates[i])
    best, xs, ys = float("inf"), [], []
    for i in order:
        best = min(best, recs[i]["time_s"])
        xs.append(dates[i])
        ys.append(best)
    xs.append(max(dates[-1], dt.date.today()))
    ys.append(best)
    ax.step(xs, ys, where="post", color="k", lw=1.2, label="record")
    ax.set_yscale("log")
    ax.set_ylabel("time to tolerance [s]")
    ax.set_title(f"{track}: record {best:.4g} s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    out = track_dir(track) / "figures" / "activity.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, help="track id or 'all'")
    ap.add_argument("--seed-baseline", default=None, help="harness result json of the baseline -> record #0")
    a = ap.parse_args()
    for t in (TRACKS if a.track == "all" else [a.track]):
        if not (track_dir(t) / "problem.py").exists():
            print(f"[{t}] not implemented yet, skipped")
            continue
        if a.seed_baseline:
            print(seed_baseline(t, a.seed_baseline))
        out = plot_activity(t)
        print(f"[{t}] {out or 'no records'}  {write_readme(t)}")



def write_readme(track: str) -> Path:
    """tracks/<t>/README.md from the problem statement, the reference figures and the current record."""
    from .tracks import load_problem
    track = resolve(track)
    p = load_problem(track)
    d = track_dir(track)
    meta_p = d / "reference" / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    recs = load_records(track)
    bar = f"**Accuracy bar:** RelL2 ≤ {p.rel_l2_tol:g}, L∞ ≤ {p.linf_tol:g}"
    if p.linf_margin > 0:
        bar += f" (L∞ only at points ≥ {p.linf_margin:g} from the walls)"
    lines = [f"# {track}: {p.title}", "", "```text", p.statement.strip(), "```", "", bar + ".", ""]
    figs = sorted((d / "figures").glob("reference*.png")) + sorted((d / "figures").glob("reference*.gif"))
    lines += [f"![{f.stem}](figures/{f.name})" for f in figs] + [""]
    if meta:
        lines += ["## Reference", "", f"{meta.get('method', '')}", ""]
        lines += [f"- `{k}`: {v:.3e}" for k, v in meta.items() if k.startswith("check") and isinstance(v, float)]
        lines += [""]
    lines += ["## Records", ""]
    if recs:
        best = min(recs, key=lambda r: r["time_s"])
        lines += [f"Current record: **{best['time_s']:.4g} s** ({best['name']}, {best['author']}, {best['date']}, "
                  f"{best['hardware']}).", "",
                  "| date | author | name | kind | time [s] | RelL2 | L∞ | hardware |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['date']} | {r['author']} | {r['name']} | {r['kind']} | {r['time_s']:.4g} | {r['rel_l2']:.2e} "
                  f"| {r['linf']:.2e} | {r['hardware']} |" for r in recs]
        if (d / "figures" / "activity.png").exists():
            lines += ["", "![activity](figures/activity.png)"]
    else:
        lines += ["No records yet."]
    out = d / "README.md"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
