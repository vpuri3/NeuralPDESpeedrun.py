"""Inspect one PR0005 training run after the fact (never during a scored solve).

    uv run python -m speedrun.submissions.submission_PR0005.inspect_run run/tracks/PR0005/<track>_<name>_<datetime> \
        [--result <harness result.json>] [--n-spectra 10] [--replot]

Reads the run's history.npz / config.json (written by pinn.RunLog), rebuilds the network from the track's
solve.py, and adds to the run directory:
  - errors.npz: RelL2 / RelL∞ against the reference at every weight snapshot (the reference is loaded here, after
    the run, as the harness does after the clock stops);
  - spectra.npz: singular values of J. Diagnostic-mode runs: those recorded during the run (log-spaced); otherwise
    computed here at n_spectra log-spaced snapshots, on the run's first draw of points;
  - solve_check.npz: at evenly spaced snapshots, the QR LM step (this submission) and submission 4's Cholesky step
    against a reference least-squares solve of [J; √μI] d = [F; 0] (relative errors of d and of J·d), and the
    singular values of J (SVD), R0 (QR), fl(JᵀJ) and [J; √μI];
  - timing.json: one LM iteration's components at the final weights (F, J, one per-trial stacked QR solve;
    for comparison submission 4's JᵀJ and Cholesky solve);
  - summary.json and inspect.png (all panels against the LM iteration count, linear; log y wherever a quantity
    spans decades; one panel of time vs iteration);
  - result.json: a copy of the harness result, if given.

Conditioning. J (rows x parameters) is rank-deficient here, so σmin(J) sits at round-off and cond(J), and
cond(JᵀJ) = cond(J)², only say "singular" (cond(JᵀJ) restricted to the numerical range is no better: it is
fixed by the rank threshold σmax · max(rows, parameters) · eps itself, ~1e23). What LM factors is JᵀJ + μI, with
eigenvalues σᵢ² + μ: cond(JᵀJ + μI) = (σmax² + μ) / (σmin² + μ) ≈ σmax² / μ. LM's filter factor σᵢ² / (σᵢ² + μ)
passes directions with σᵢ ≫ √μ and freezes those with σᵢ ≪ √μ, so the count of σᵢ > √μ ("active directions") is
shown next to the numerical rank.

Why the solves still work at cond(JᵀJ + μI) ~ 1e15. Cholesky is backward stable: the computed step solves
(JᵀJ + μI + E) d = JᵀF with ‖E‖ ~ ε σmax². Its error in d is concentrated in directions with σᵢ² ≲ μ (bounded by
ε κ, the textbook number), but those enter the residual change J·d multiplied by σᵢ, so the relative error of J·d
is ≲ ε σmax / (2√μ) ≈ ε √κ. LM only needs J·d (the predicted decrease and the accept test), and the accept test is
exact anyway. The solve_check panel measures both errors in every run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]                                                                 # sequential blue, light -> dark
INK, INK2, GRID, SURF, BAR = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"


def load_solve(track: str):
    path = REPO / "tracks" / track / "submissions" / "submission_PR0005" / "solve.py"
    spec = importlib.util.spec_from_file_location(f"pr0004_solve_{track}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timeit(f, *a, reps=5):
    import jax
    jax.block_until_ready(f(*a))
    t = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(f(*a))
    return (time.perf_counter() - t) / reps


def iteration_times(h):
    """Wall (or training-clock) time of every iteration, interpolated within each device chunk."""
    n = len(h["iter"])
    return np.interp(np.arange(1, n + 1), np.r_[0, h["chunk_it"]], np.r_[0.0, h["chunk_t"]])


def analyse(run: Path, n_spectra: int, device: str = "cuda"):
    conf = json.loads((run / "config.json").read_text())
    h = dict(np.load(run / "history.npz"))
    os.environ["PR0005_ARCH"] = conf["arch"]
    os.environ["PR0005_RUN_DIR_WRITTEN"] = "inspect"          # never write a new run directory from here
    os.environ.pop("PR0005_DIAG", None)
    import jax
    import jax.numpy as jnp
    import torch

    from speedrun.metrics import rel_l2, rel_linf
    from speedrun.tracks import load_problem, load_reference

    problem = load_problem(conf["track"], device=device)
    model = load_solve(conf["track"]).setup(problem)
    ref = load_reference(conf["track"])
    mask = problem.linf_mask()
    cols = {c: i for i, c in enumerate(conf["iter_cols"])}
    mu_it = h["iter"][:, cols["mu"]]

    W, tp = h["probe_w"], h["probe_t"]
    l2, linf = np.empty(len(W)), np.empty(len(W))
    for i, w in enumerate(W):
        u = torch.from_numpy(np.asarray(model._pred_c(jnp.asarray(w)), np.float64))
        l2[i], linf[i] = rel_l2(u, ref), rel_linf(u, ref, mask)
    np.savez(run / "errors.npz", t=tp, it=h["probe_it"], rel_l2=l2, rel_linf=linf)

    Zs = model.Zs0
    J_c = jax.jit(lambda w: model.J(w, Zs))
    eps = np.finfo(np.float64).eps
    if conf.get("diagnostic_mode"):
        ts, sv, si = h["diag_spec_t"], np.asarray(h["diag_spec"]), h["diag_spec_it"]
        if len(ts) > n_spectra:
            keep = sorted({int(np.argmin(np.abs(si - i))) for i in np.linspace(si[0], si[-1], n_spectra)})
            ts, sv, si = ts[keep], sv[keep], si[keep]
        cond = {"t": h["diag_t"], "it": h["diag_it"], "mu": h["diag_mu"], "smax": h["diag_sv_max"], "smin": h["diag_sv_min"],
                "rank": h["diag_rank"], "active": h["diag_n_active"]}
    else:
        pi_ = h["probe_it"]
        picks = sorted({int(np.argmin(np.abs(pi_ - i))) for i in np.linspace(pi_[0], pi_[-1], n_spectra)})
        ts, si = tp[picks], pi_[picks]
        sv = np.array([np.asarray(jnp.linalg.svd(J_c(jnp.asarray(W[i])), compute_uv=False)) for i in picks])
        rank = (sv > sv[:, :1] * max(conf["rows"], conf["n_params"]) * eps).sum(1)
        mu_s = mu_it[np.clip(h["probe_it"][picks] - 1, 0, len(mu_it) - 1)]
        cond = {"t": ts, "it": si, "mu": mu_s, "smax": sv[:, 0], "smin": sv[:, -1], "rank": rank,
                "active": (sv > np.sqrt(mu_s)[:, None]).sum(1)}
    np.savez(run / "spectra.npz", t=ts, it=si, sv=sv)

    # the linear solve, checked: at evenly spaced snapshots, the QR step (submission 5) and the Cholesky step of
    # submission 4 (JᵀJ formed in float64) against a reference least-squares solve of [J; √μ I] d = [F; 0] (SVD);
    # singular values of J (SVD), of R0 (from J = Q0 R0), of fl(JᵀJ) (square roots), and of [J; √μ I] (= R1)
    from speedrun.submissions.submission_PR0005.pinn import cholesky_step, qr_step
    F_c = jax.jit(lambda w: model.F(w, Zs))
    nchk = min(6, len(W))
    pit = h["probe_it"]
    chk_i = sorted({int(np.argmin(np.abs(pit - i))) for i in np.linspace(pit[0], pit[-1], nchk)})
    keys = ("t", "it", "mu", "kappa", "err_d_qr", "err_Jd_qr", "err_d_chol", "err_Jd_chol", "pred_ratio_qr",
            "sv_J", "sv_R0", "sv_JtJ_sqrt", "sv_aug")
    chk = {k: [] for k in keys}
    for i in chk_i:
        w_ = jnp.asarray(W[i]); mu_ = float(mu_it[min(max(int(h["probe_it"][i]) - 1, 0), len(mu_it) - 1)])
        J_, F_ = J_c(w_), F_c(w_)
        dq = qr_step(J_, F_, mu_)
        R0 = jnp.linalg.qr(J_, mode="r")
        dc = cholesky_step(J_.T @ J_, J_.T @ F_, mu_)
        M = jnp.concatenate([J_, np.sqrt(mu_) * jnp.eye(J_.shape[1])])
        dr = jnp.linalg.lstsq(M, jnp.concatenate([F_, jnp.zeros(J_.shape[1])]), rcond=None)[0]
        svd = lambda X: np.asarray(jnp.linalg.svd(X, compute_uv=False))
        sJ, sR0, sA = svd(J_), svd(R0), svd(J_.T @ J_)
        sAug = np.sqrt(sJ ** 2 + mu_)
        ed = lambda d: float(jnp.linalg.norm(d - dr) / jnp.linalg.norm(dr))
        eJ = lambda d: float(jnp.linalg.norm(J_ @ (d - dr)) / jnp.linalg.norm(J_ @ dr))
        pred = lambda d: float(F_ @ F_ - jnp.sum((F_ - J_ @ d) ** 2))
        for k, v in zip(keys, (tp[i], int(pit[i]), mu_, float((sJ[0] ** 2 + mu_) / (sJ[-1] ** 2 + mu_)), ed(dq), eJ(dq),
                               ed(dc), eJ(dc), pred(dq) / pred(dr), sJ, sR0, np.sqrt(np.maximum(sA, 0)), sAug)):
            chk[k].append(v)
    chk = {k: np.asarray(v) for k, v in chk.items()}
    np.savez(run / "solve_check.npz", **chk)

    w = jnp.asarray(W[-1])
    J, F = J_c(w), F_c(w)
    A_c = jax.jit(lambda J: J.T @ J)
    A, g = A_c(J), J.T @ F
    step_c, chol_c = jax.jit(qr_step), jax.jit(cholesky_step)
    timing = {"F (residual)": timeit(F_c, w), "J (Jacobian)": timeit(J_c, w),
              "QR of [J F; √μI 0] + back subst. (per trial)": timeit(step_c, J, F, 1e-4),
              "JᵀJ (submission 4)": timeit(A_c, J), "Cholesky + solve (submission 4)": timeit(chol_c, A, g, 1e-4)}
    (run / "timing.json").write_text(json.dumps({k: 1e3 * v for k, v in timing.items()}, indent=1))

    it = h["iter"]
    ok = (l2 <= problem.rel_l2_tol) & (linf <= problem.rel_linf_tol)
    c_damped = (cond["smax"] ** 2 + cond["mu"]) / (cond["smin"] ** 2 + cond["mu"])
    summary = {
        "track": conf["track"], "name": conf["name"], "arch": conf["arch"], "n_params": conf["n_params"],
        "rows": conf["rows"], "resample_every": conf["config"]["resample_every"],
        "mu_patience": conf["config"]["mu_patience"], "stop": conf["stop"],
        "iterations": int(len(it)), "solve_time_s": float(h["chunk_t"][-1]),
        "ms_per_iteration": 1e3 * float(h["chunk_t"][-1]) / max(len(it), 1),
        "trials_per_iteration": float(np.nanmean(it[:, cols["trials"]])),
        "first_trial_rejected_fraction": float(np.mean(it[:, cols["trials"]] > 1)),
        "final": {"loss": float(it[-1, cols["loss"]]), "mu": float(mu_it[-1]), "rel_l2": float(l2[-1]),
                  "rel_linf": float(linf[-1])},
        "best_rel_l2": {"value": float(l2.min()), "t_s": float(tp[l2.argmin()]), "it": int(h["probe_it"][l2.argmin()])},
        "bar_first_met": ({"t_s": float(tp[ok.argmax()]), "it": int(h["probe_it"][ok.argmax()])} if ok.any() else None),
        "final_cond_JtJ_plus_muI": float(c_damped[-1]),
        "final_numerical_rank": int(cond["rank"][-1]),
        "final_active_directions": int(cond["active"][-1]),
        "solver": conf["config"].get("solver", "cholesky"),
        "solve_check": {f"it {int(t)}": {"mu": float(m_), "cond_JtJ_plus_muI": float(k_),
                                         "cond_J_sqrtmu_stacked": float(np.sqrt(k_)),
                                         "rel_err_d_qr": float(a), "rel_err_Jd_qr": float(b),
                                         "rel_err_d_cholesky": float(e1), "rel_err_Jd_cholesky": float(e2),
                                         "pred_ratio_qr_over_ref": float(r_)}
                        for t, m_, k_, a, b, e1, e2, r_ in zip(chk["it"], chk["mu"], chk["kappa"], chk["err_d_qr"],
                                                               chk["err_Jd_qr"], chk["err_d_chol"], chk["err_Jd_chol"],
                                                               chk["pred_ratio_qr"])},
        "timing_ms": {k: 1e3 * v for k, v in timing.items()},
        "bar": {"rel_l2": problem.rel_l2_tol, "rel_linf": problem.rel_linf_tol},
        "diagnostic_mode": bool(conf.get("diagnostic_mode")), "diagnostics_time_s": conf.get("diagnostics_time_s"),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    return conf, h, cols, (l2, linf), (si, sv), cond, chk, timing, summary


def moving(x, n=100):
    if len(x) < n:
        return x
    c = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    return np.r_[c[1:n] / np.arange(1, n), (c[n:] - c[:-n]) / n]


def plot(run: Path, conf, h, cols, err, spec, cond, chk, timing, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    diag = bool(conf.get("diagnostic_mode"))
    it = h["iter"]
    xi = np.arange(1, len(it) + 1)                 # every iteration
    ip, pr = h["probe_it"], h["probe_probe"]       # weight snapshots
    iq, pq = (h["diag_it"], h["diag_probe"]) if diag else (ip, pr)
    names = conf["groups"]
    ng = len(names)
    l2, linf = err
    ibar = summary["bar_first_met"]["it"] if summary["bar_first_met"] else None
    every = "every 50 iterations" if diag else "at snapshots"

    f, axs = plt.subplots(7, 3, figsize=(15, 27), facecolor=SURF)
    ax = axs.ravel()

    def style(a, title, ylog=True, xlabel="LM iteration", bar=True):
        """Linear x (iterations, unless given); log y where the quantity spans decades."""
        a.set_title(title, color=INK, fontsize=10, loc="left")
        a.set_facecolor(SURF)
        a.set_xlabel(xlabel, color=INK2, fontsize=8.5)
        if ylog:
            a.set_yscale("log")
            lo, hi = a.get_ylim()
            if lo > 0 and hi / lo > 10:                   # minor labels only on sub-decade axes
                a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.grid(True, which="major", color=GRID, lw=0.6)
        a.tick_params(colors=INK2, labelsize=8)
        for s_ in a.spines.values():
            s_.set_color(GRID)
        if ibar and bar:
            a.axvline(ibar, color=BAR, lw=1, ls=(0, (4, 3)))

    def legend(a):
        a.legend(fontsize=7.5, frameon=False, labelcolor=INK2)

    one = dict(color=SERIES[0], lw=1.2)
    thin = dict(color=SERIES[0], lw=0.6, rasterized=True)
    re = conf["config"]["resample_every"]
    ax[0].plot(xi, it[:, cols["loss"]], **thin)
    style(ax[0], "loss per iteration, on the current points" + (f" (redrawn every {re})" if re else ""))
    for g in range(ng):
        ls = (0, (4, 2)) if g == 1 else "solid"
        ax[1].plot(iq, pq[:, 2 * g], color=SERIES[g % 6], lw=1.4, ls=ls, label=names[g])
        ax[2].plot(iq, pq[:, 2 * g + 1], color=SERIES[g % 6], lw=1.4, ls=ls, label=names[g])
    style(ax[1], "residual RMS per group (unscaled)"); legend(ax[1])
    style(ax[2], "residual max |r| per group (unscaled)"); legend(ax[2])

    ax[3].plot(ip, l2, color=SERIES[0], lw=1.4, label="RelL2")
    ax[3].plot(ip, linf, color=SERIES[1], lw=1.4, label="RelL∞")
    for v, c in ((summary["bar"]["rel_l2"], SERIES[0]), (summary["bar"]["rel_linf"], SERIES[1])):
        ax[3].axhline(v, color=c, lw=0.8, ls=(0, (1, 2)))
    style(ax[3], "error vs reference (dotted: bar; dashed: bar first met)"); legend(ax[3])

    ax[4].plot(xi, it[:, cols["mu"]], **one); style(ax[4], "damping μ, every iteration")
    ax[5].plot(xi, np.clip(it[:, cols["rho"]], 1e-4, None), **thin)
    style(ax[5], "gain ratio ρ = actual / predicted decrease, every iteration")
    ax[6].plot(xi, moving(it[:, cols["trials"]]), **one)
    style(ax[6], "trials per iteration (mean over 100; 1 = no rejection)", ylog=False)
    ax[7].plot(xi, it[:, cols["step"]], **thin); style(ax[7], "step size ‖d‖, every iteration")
    ax[8].plot(iq, pq[:, 2 * ng], **one); style(ax[8], "gradient ‖JᵀF‖ (current points)")
    ax[9].plot(iq, pq[:, 2 * ng + 1], **one); style(ax[9], "weights ‖w‖")

    si, sv = spec
    cs = [RAMP[int(round(k))] for k in np.linspace(0, len(RAMP) - 1, len(si))]
    for k, (ii, s_) in enumerate(zip(si, sv)):
        ax[10].plot(np.arange(1, len(s_) + 1), s_, color=cs[k], lw=1.2, label=f"iteration {int(ii)}")
    style(ax[10], "singular values of J (light → dark: later)", xlabel="index", bar=False); legend(ax[10])

    mk = {} if diag else dict(marker="o", ms=4)
    ic, mu, smax, smin = cond["it"], cond["mu"], cond["smax"], cond["smin"]
    kap = (smax ** 2 + mu) / (smin ** 2 + mu)
    ax[11].plot(ic, np.sqrt(kap), color=SERIES[0], lw=1.4, label="cond([J; √μI]): what the QR step solves", **mk)
    ax[11].plot(ic, kap, color=SERIES[1], lw=1.4, ls=(0, (4, 2)), label="cond(JᵀJ + μI) = its square (normal equations)", **mk)
    style(ax[11], f"condition numbers ({every})"); legend(ax[11])
    ax[12].plot(ic, smax, color=SERIES[0], lw=1.4, label="σmax(J)", **mk)
    ax[12].plot(ic, np.sqrt(mu), color=SERIES[2], lw=1.4, label="√μ (damping scale)", **mk)
    ax[12].plot(ic, smin, color=SERIES[1], lw=1.4, label="σmin(J) (round-off: J is singular)", **mk)
    style(ax[12], f"singular values vs damping ({every})"); legend(ax[12])
    ax[13].plot(ic, cond["rank"], color=SERIES[0], lw=1.4, label="numerical rank", **mk)
    ax[13].plot(ic, cond["active"], color=SERIES[1], lw=1.4, label="active: σ > √μ", **mk)
    ax[13].axhline(conf["n_params"], color=BAR, lw=0.8, ls=(0, (1, 2)))
    style(ax[13], f"directions of J (dotted: {conf['n_params']} parameters)", ylog=False); legend(ax[13])

    ax[14].plot(np.r_[0, h["chunk_it"]], np.r_[0.0, h["chunk_t"]], **one)
    style(ax[14], "time vs iteration" + (" (training clock; diagnostics excluded)" if diag else " (wall clock)"),
          ylog=False)
    ax[14].set_ylabel("s", color=INK2, fontsize=8.5)
    keys = list(timing)
    ms = [1e3 * timing[k] for k in keys]
    ax[15].barh(keys, ms, color=SERIES[0], height=0.5)
    for y, v in enumerate(ms):
        ax[15].text(v, y, f" {v:.3g} ms", va="center", color=INK2, fontsize=8)
    style(ax[15], "one LM iteration: component times (final weights)", ylog=False, xlabel="ms", bar=False)
    ax[15].set_xlim(0, max(ms) * 1.35)

    eps = np.finfo(np.float64).eps
    last = -1
    idx = np.arange(1, len(chk["sv_J"][last]) + 1)
    ax[16].plot(idx, chk["sv_J"][last], color=SERIES[0], lw=1.6, label="σ(J), SVD of J")
    ax[16].plot(idx, np.sort(chk["sv_R0"][last])[::-1], color=SERIES[2], lw=1.2, ls=(0, (4, 2)), label="σ(R), from the R-only QR of J")
    ax[16].plot(idx, chk["sv_JtJ_sqrt"][last], color=SERIES[1], lw=1.2, ls=(0, (1, 1.5)), label="√σ(fl(JᵀJ)): what the normal equations see")
    ax[16].axhline(eps * chk["sv_J"][last][0], color=BAR, lw=0.8, ls=(0, (1, 2)))
    ax[16].axhline(np.sqrt(eps) * chk["sv_J"][last][0], color=BAR, lw=0.8, ls=(0, (4, 2)))
    ax[16].text(1, eps * chk["sv_J"][last][0] * 3, "ε·σmax (QR floor)", color=INK2, fontsize=7.5)
    ax[16].text(1, np.sqrt(eps) * chk["sv_J"][last][0] * 3, "√ε·σmax (normal-equations floor)", color=INK2, fontsize=7.5)
    style(ax[16], f"small singular values: SVD vs QR vs JᵀJ (iteration {int(chk['it'][last])})", xlabel="index", bar=False)
    legend(ax[16])
    cs = [RAMP[int(round(k))] for k in np.linspace(0, len(RAMP) - 1, len(chk["it"]))]
    for k, (ii, e, m_) in enumerate(zip(chk["it"], chk["sv_aug"], chk["mu"])):
        ax[17].plot(np.arange(1, len(e) + 1), e, color=cs[k], lw=1.2, label=f"iteration {int(ii)}, √μ = {np.sqrt(m_):.0e}")
    style(ax[17], "singular values of [J; √μI] = σ(R1) (floor √μ)", xlabel="index", bar=False); legend(ax[17])
    ax[18].plot(chk["it"], chk["err_d_qr"], color=SERIES[0], lw=1.6, marker="o", ms=4, label="QR: rel. error of d")
    ax[18].plot(chk["it"], chk["err_Jd_qr"], color=SERIES[0], lw=1.2, ls=(0, (4, 2)), marker="s", ms=3, label="QR: rel. error of J·d")
    ax[18].plot(chk["it"], chk["err_d_chol"], color=SERIES[1], lw=1.6, marker="o", ms=4, label="Cholesky (sub. 4): d")
    ax[18].plot(chk["it"], chk["err_Jd_chol"], color=SERIES[1], lw=1.2, ls=(0, (4, 2)), marker="s", ms=3, label="Cholesky (sub. 4): J·d")
    ax[18].plot(chk["it"], eps * np.sqrt(chk["kappa"]), color=SERIES[2], lw=1, ls=(0, (1, 1.5)), label="ε·√κ")
    ax[18].plot(chk["it"], eps * chk["kappa"], color=BAR, lw=1, ls=(0, (1, 1.5)), label="ε·κ")
    style(ax[18], "step accuracy vs SVD reference solve of [J; √μI]"); legend(ax[18])

    s = summary
    bm = s["bar_first_met"]
    text = [s["track"], f"{s['arch']}  ·  {s['n_params']} parameters  ·  {s['rows']} rows",
            f"points redrawn every {s['resample_every']} iterations" if s["resample_every"] else "points fixed",
            f"μ ÷10 after {s['mu_patience']} first-trial acceptances",
            f"stopped by: {s['stop']}", f"iterations: {s['iterations']}  ({s['ms_per_iteration']:.3g} ms each)",
            f"trials / iteration: {s['trials_per_iteration']:.2f}  "
            f"(first trial rejected: {100 * s['first_trial_rejected_fraction']:.0f}%)", "",
            f"final loss {s['final']['loss']:.2e},  μ {s['final']['mu']:.0e}",
            f"final RelL2 {100 * s['final']['rel_l2']:.3g}%,  RelL∞ {100 * s['final']['rel_linf']:.3g}%",
            f"best RelL2 {100 * s['best_rel_l2']['value']:.3g}% at {s['best_rel_l2']['t_s']:.3g} s (iteration {s['best_rel_l2']['it']})",
            f"bar first met: {f'{bm['t_s']:.3g} s (iteration {bm['it']})' if bm else 'never'}",
            f"solver: {s['solver']}",
            f"final cond([J; √μI]) {np.sqrt(s['final_cond_JtJ_plus_muI']):.1e} (normal eqs: {s['final_cond_JtJ_plus_muI']:.1e})",
            f"final rank {s['final_numerical_rank']}, active (σ > √μ) {s['final_active_directions']}"]
    if diag:
        text += ["", f"diagnostic mode: {s['diagnostics_time_s']:.3g} s of diagnostics kept off the clock"]
    for a in (ax[19], ax[20]):
        a.axis("off")
    ax[19].text(0, 1, "\n".join(text), va="top", ha="left", color=INK, fontsize=9.5, family="monospace")
    f.suptitle(f"{run.name}  —  shallow PINN + Levenberg-Marquardt (PR0005)", color=INK, fontsize=12, x=0.01,
               ha="left", y=0.995)
    f.tight_layout(rect=(0, 0, 1, 0.99))
    f.savefig(run / "inspect.png", dpi=110, facecolor=SURF)
    plt.close(f)


def load_saved(run: Path):
    """Everything plot() needs, from the files analyse() wrote (no GPU, no model). Older run directories lack the
    iteration numbers of the spectra and solve checks; those are recovered from the snapshot times."""
    conf = json.loads((run / "config.json").read_text())
    h = dict(np.load(run / "history.npz"))
    summary = json.loads((run / "summary.json").read_text())
    timing = {k: v / 1e3 for k, v in json.loads((run / "timing.json").read_text()).items()}
    cols = {c: i for i, c in enumerate(conf["iter_cols"])}
    e = np.load(run / "errors.npz")
    t2it = lambda t: np.rint(np.interp(t, h["probe_t"], h["probe_it"])).astype(int)
    sp = dict(np.load(run / "spectra.npz"))
    si = sp["it"] if "it" in sp else t2it(sp["t"])
    chk = dict(np.load(run / "solve_check.npz"))
    chk.setdefault("it", t2it(chk["t"]))
    mu_it = h["iter"][:, cols["mu"]]
    if conf.get("diagnostic_mode"):
        cond = {"t": h["diag_t"], "it": h["diag_it"], "mu": h["diag_mu"], "smax": h["diag_sv_max"],
                "smin": h["diag_sv_min"], "rank": h["diag_rank"], "active": h["diag_n_active"]}
    else:
        sv = sp["sv"]
        mu_s = mu_it[np.clip(si - 1, 0, len(mu_it) - 1)]
        rank = (sv > sv[:, :1] * max(conf["rows"], conf["n_params"]) * np.finfo(np.float64).eps).sum(1)
        cond = {"t": sp["t"], "it": si, "mu": mu_s, "smax": sv[:, 0], "smin": sv[:, -1], "rank": rank,
                "active": (sv > np.sqrt(mu_s)[:, None]).sum(1)}
    b = summary["best_rel_l2"]
    b.setdefault("it", int(e["it"][int(np.argmin(e["rel_l2"]))]))
    return conf, h, cols, (e["rel_l2"], e["rel_linf"]), (si, sp["sv"]), cond, chk, timing, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--result", type=Path, default=None)
    ap.add_argument("--n-spectra", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replot", action="store_true", help="only redraw inspect.png from the saved files (CPU)")
    a = ap.parse_args()
    if a.result:
        shutil.copy(a.result, a.run / "result.json")
    if a.replot:
        plot(a.run, *load_saved(a.run))
        return 0
    out = analyse(a.run, a.n_spectra, a.device)
    plot(a.run, *out)
    print(json.dumps(out[-1], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
