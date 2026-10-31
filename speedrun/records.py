"""Leaderboards: read tracks/<t>/records.json, draw tracks/<t>/figures/activity.png, write
tracks/<t>/README.md, and seed record #0.

    uv run python -m speedrun.records --track all                 # redraw every activity plot
    uv run python -m speedrun.records --track 07 --seed-baseline tracks/track07_2D_HighContrastDiffusion/logs/baseline.json

A record entry is
    {"date": "YYYY-MM-DD", "author": ..., "name": ...,
     "submission": "tracks/<t>/baseline.py" | "tracks/<t>/submissions/submission_PR<NNNN>",
     "time_s": ..., "rel_l2": ..., "rel_linf": ..., "hardware": ..., "kind": "classical" | "neural" | "hybrid",
     "pr": <PR number or null>}
`time_s` is the official (maintainer, H100 80GB) time. Entries timed elsewhere say so in `hardware`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .tracks import TRACKS, dim_label, key, name, resolve, track_dir

KIND_STYLE = {"classical": ("tab:gray", "s"), "neural": ("tab:blue", "o"), "hybrid": ("tab:purple", "D")}


def records_path(track: str) -> Path:
    return track_dir(track) / "records.json"


def load_records(track: str) -> list[dict]:
    p = records_path(track)
    return json.loads(p.read_text()) if p.exists() else []


def save_records(track: str, recs: list[dict]) -> None:
    recs = sorted(recs, key=lambda r: (r["date"], -r["time_s"]))
    records_path(track).write_text(json.dumps(recs, indent=2) + "\n")


def add_entry(track: str, result_json: str, name: str, kind: str, author: str = "maintainers",
              pr: int | None = None, date: str | None = None) -> dict:
    """Add a harness result to records.json (replacing an entry of the same submission). Entries that miss the
    accuracy bar are kept (passed = false): they are points on the accuracy-vs-time plot, not records."""
    track = resolve(track)
    r = json.loads(Path(result_json).read_text())
    assert r["track"] == track, f"{result_json} is a {r['track']} result, not {track}"
    if not r.get("impose_budget", True):
        raise ValueError(f"{result_json} ran without the budget (--no-impose-budget): not a record")
    entry = {"date": date or r["date"][:10], "author": author, "name": name, "submission": r["submission"],
             "time_s": r["time_s"], "solve_time_s": r.get("solve_time_s", r["time_s"]),
             "query_time_s": r.get("query_time_s", 0.0), "rel_l2": r["rel_l2"], "rel_linf": r["rel_linf"],
             "passed": r["passed"], "hardware": r["hardware"], "kind": kind, "pr": pr}
    if "stop" in r:
        entry.update(stop=r["stop"], budget_s=r["budget_s"])
    recs = [e for e in load_records(track) if e.get("submission") != r["submission"]]
    save_records(track, recs + [entry])
    return entry


def seed_baseline(track: str, result_json: str, date: str | None = None, author: str = "maintainers") -> dict:
    """Make the harness result of the track's baseline record #0."""
    r = json.loads(Path(result_json).read_text())
    assert r["passed"], f"{result_json}: the baseline does not pass"
    return add_entry(track, result_json, "record #0: classical baseline", "classical", author, date=date)


def passing(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r.get("passed", True)]


def best_submission(recs: list[dict]) -> dict | None:
    """Best non-baseline entry: until some submission meets the bar, the most accurate (lowest RelL2);
    afterwards, the fastest one that meets the bar."""
    subs = [r for r in recs if r.get("kind") != "classical"]
    if not subs:
        return None
    ok = passing(subs)
    return min(ok, key=lambda r: r["time_s"]) if ok else min(subs, key=lambda r: r["rel_l2"])


TABLE_BEGIN, TABLE_END = "<!-- tracks-table:begin -->", "<!-- tracks-table:end -->"
STATUS = {"track08_2D_Helmholtz": "*on hold*", "track14_2DT_CylinderFlow": "*TODO*"}


def tracks_table() -> str:
    """Main-README table: best submission per track (see best_submission), its time and accuracy."""
    from .tracks import load_problem
    lines = ["| track | best submission | time [s] | RelL2 | RelL∞ | bar met |", "|---|---|---|---|---|---|"]
    for t in TRACKS:
        label = f"{key(t)} {name(t)} ({dim_label(t)})"
        if t in STATUS or not (track_dir(t) / "problem.py").exists():
            lines.append(f"| {label} | {STATUS.get(t, '*TODO*')} | | | | |")
            continue
        b = best_submission(load_records(t))
        if b is None:
            lines.append(f"| {label} | — | | | | |")
            continue
        p = load_problem(t)
        link = f"[{b['name']}]({b['submission']})" if b.get("submission") else b["name"]
        lines.append(f"| {label} | {link} | {b['time_s']:.4g} | {b['rel_l2']:.2e} (≤ {p.rel_l2_tol:g}) | "
                     f"{b['rel_linf']:.2e} (≤ {p.rel_linf_tol:g}) | {'yes' if b.get('passed') else 'no'} |")
    return "\n".join(lines)


def update_main_readme(path: Path | None = None) -> Path:
    from .tracks import ROOT
    path = path or ROOT / "README.md"
    s = path.read_text()
    a, b = s.index(TABLE_BEGIN) + len(TABLE_BEGIN), s.index(TABLE_END)
    path.write_text(s[:a] + "\n" + tracks_table() + "\n" + s[b:])
    return path


def plot_accuracy(track: str) -> Path | None:
    """RelL2 vs wall-clock time, one point per submission, the track's bar as a dashed line."""
    from .tracks import load_problem
    track = resolve(track)
    recs = load_records(track)
    if not recs:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = load_problem(track)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for kind, (c, m) in KIND_STYLE.items():
        pts = [r for r in recs if r.get("kind", "classical") == kind]
        for r in pts:
            ax.scatter(r["time_s"], max(r["rel_l2"], 1e-16), color=c, marker=m, s=46, zorder=3,
                       facecolors=c if r.get("passed", True) else "none")
            ax.annotate(r["name"].replace("record #0: ", ""), (r["time_s"], max(r["rel_l2"], 1e-16)),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        if pts:
            ax.scatter([], [], color=c, marker=m, label=kind)
    ax.axhline(p.rel_l2_tol, color="k", ls="--", lw=1, label=f"bar RelL2 = {p.rel_l2_tol:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock time [s] (solve + query)")
    ax.set_ylabel("RelL2")
    ax.set_title(f"{key(track)} {name(track)} ({dim_label(track)})  (hollow = misses the bar)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    out = track_dir(track) / "figures" / "accuracy_vs_time.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_activity(track: str) -> Path | None:
    track = resolve(track)
    recs = passing(load_records(track))
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
    ax.set_title(f"{key(track)} {name(track)} ({dim_label(track)}): record {best:.4g} s")
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
    ap.add_argument("--add", default=None, help="harness result json of a submission to add")
    ap.add_argument("--name", default=None)
    ap.add_argument("--kind", default="neural", choices=list(KIND_STYLE))
    ap.add_argument("--author", default="maintainers")
    ap.add_argument("--pr", type=int, default=None)
    a = ap.parse_args()
    for t in (TRACKS if a.track == "all" else [a.track]):
        if not (track_dir(t) / "problem.py").exists():
            print(f"[{t}] not implemented yet, skipped")
            continue
        if a.seed_baseline:
            print(seed_baseline(t, a.seed_baseline))
        if a.add:
            print(add_entry(t, a.add, a.name or Path(a.add).parent.name, a.kind, a.author, a.pr))
        plot_accuracy(t)
        out = plot_activity(t)
        print(f"[{t}] {out or 'no records'}  {write_readme(t)}")
    if a.track == "all":
        print(f"[README] {update_main_readme()}")


def write_readme(track: str) -> Path:
    """tracks/<t>/README.md from the problem statement, the reference figures and the current record."""
    from .tracks import load_problem
    track = resolve(track)
    p = load_problem(track)
    d = track_dir(track)
    meta_p = d / "reference" / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    recs = load_records(track)
    bar = f"**Accuracy bar:** RelL2 ≤ {p.rel_l2_tol:g}, RelL∞ ≤ {p.rel_linf_tol:g}"
    if p.linf_margin > 0:
        bar += f" (RelL∞ only at points ≥ {p.linf_margin:g} from the walls)"
    bar += f". **Budget:** {p.budget_s:g} s of wall-clock solve time"
    lines = [f"# {key(track)} {name(track)} ({dim_label(track)}): {p.title}", "", "```text", p.statement.strip(), "```", "", bar + ".", ""]
    figs = sorted((d / "figures").glob("reference*.png")) + sorted((d / "figures").glob("reference*.gif"))
    lines += [f"![{f.stem}](figures/{f.name})" for f in figs] + [""]
    if meta:
        lines += ["## Reference", "", f"{meta.get('method', '')}", ""]
        lines += [f"- `{k}`: {v:.3e}" for k, v in meta.items() if k.startswith("check") and isinstance(v, float)]
        lines += [""]
    lines += ["## Records", ""]
    if recs:
        best = min(passing(recs) or recs, key=lambda r: r["time_s"])
        lines += [f"Current record: **{best['time_s']:.4g} s** ({best['name']}, {best['author']}, {best['date']}, "
                  f"{best['hardware']}).", "",
                  "| date | author | name | kind | time [s] | RelL2 | RelL∞ | passes | hardware |",
                  "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['date']} | {r['author']} | {r['name']} | {r['kind']} | {r['time_s']:.4g} | {r['rel_l2']:.2e} "
                  f"| {r['rel_linf']:.2e} | {'yes' if r.get('passed', True) else 'no'} | {r['hardware']} |" for r in recs]
        for fig in ("accuracy_vs_time", "activity"):
            if (d / "figures" / f"{fig}.png").exists():
                lines += ["", f"![{fig}](figures/{fig}.png)"]
    else:
        lines += ["No records yet."]
    out = d / "README.md"
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    main()
