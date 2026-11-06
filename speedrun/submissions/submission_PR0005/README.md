# submission_PR0005: shallow PINN + Levenberg-Marquardt with QR steps (PR #5)

Submission 4 ([`speedrun/submissions/submission_PR0004`](../submission_PR0004/README.md): tanh MLP, residual points
redrawn every 50 iterations, conditions as residual rows, LM with μI damping and the patience rule, float64) with
one change: **the LM step is computed from a QR factorization instead of the normal equations**
(`IDEAS.md` §5 item 8). Code: [`pinn.py`](pinn.py) (`Config.solver`, `qr_step`); inspection:
[`inspect_run.py`](inspect_run.py).

## The step

Both submissions compute the same step, d = argmin ‖[J; √μ I] d − [r; 0]‖, i.e. (JᵀJ + μI) d = Jᵀr.

- Submission 4 forms JᵀJ and factors JᵀJ + μI by Cholesky: condition κ = (σmax² + μ)/(σmin² + μ), relative error
  of d up to ε·κ.
- Submission 5 never forms JᵀJ: per trial μ, one R-only Householder QR of the stacked system
  [[J, r], [√μ I, 0]] (r rides along as the last column, so Q is never formed), then back substitution R d = c.
  The system solved has condition cond([J; √μ I]) = √κ: relative error of d up to ε·√κ.

On synthetic least-squares problems with κ = 1e14, the QR step's error is 2e-9 and the Cholesky step's 3e-3.

Cost on the GPU (1D_Steady_AdvectionDiffusion, 32×4: J is 10002 × 3265, float64, H100): JᵀJ 4.8 ms + Cholesky
2.4 ms per trial, against 38.6 ms per trial for the stacked QR (Householder QR runs far below GEMM throughput).
Two alternatives were slower: a QR of J once per iteration (35 ms) plus a QR of [R0; √μI] per trial (30 ms), and an
SVD of R0 once per iteration (1.4 s) with O(m²) trials. With ~1.1 trials per iteration, an LM iteration costs
~42 ms instead of ~8 ms.

`Config(solver="cholesky")` keeps submission 4's step for comparison.

## Inspection

`inspect_run.py` (submission 4's, extended): the condition number of the system the QR step solves,
cond([J; √μI]), next to the normal equations' cond(JᵀJ + μI); the small singular values of J as seen by an SVD of J,
by the R of a QR of J, and by the normal equations (√σ(fl(JᵀJ)), floor √ε·σmax); the singular values of
[J; √μI] (floor √μ); and, at snapshots, the QR and the Cholesky steps against an SVD-based reference solve (relative
errors of d and of J·d, with the ε·√κ and ε·κ bounds). Timings list the stacked QR and, for comparison,
submission 4's JᵀJ and Cholesky.

## Results

**1D_Steady_AdvectionDiffusion** (32×4, points redrawn every 50 iterations, 1000 s budget, H100), against submission
4's identical setup with the Cholesky step:

| | submission 4 (Cholesky) | submission 5 (QR) |
|---|---|---|
| RelL2 / RelL∞ (scored) | 2.7e-11 / 3.5e-11 | 5.8e-10 / 6.5e-10 |
| bar first met | 0.96 s (iteration 113) | 3.43 s (iteration 78) |
| iterations in 1000 s (ms each) | 129k (7.7) | 23.9k (41.9) |
| final μ; cond(JᵀJ + μI); cond([J; √μI]) | 1e-7; 4.2e13; — | 1e-5; 4.3e11; 6.5e5 |
| step error vs SVD reference, d (late snapshots) | — | 1e-12 (Cholesky on the same J: 6e-5) |
| step error, J·d | — | 2e-11 (Cholesky: 4e-7) |

The QR step is accurate to ~1e-12 where the normal equations give 6e-5, and R from the QR resolves J's singular values
down to ~1e-16·σmax where JᵀJ stops at ~√ε·σmax ≈ 1e-5 (the "small singular values" panel). On this track that does
not change the outcome: both pass by a wide margin, and at 5.4× the cost per iteration the QR run does fewer
iterations and ends less accurate. The more accurate step pays off only where the normal equations' error matters
(larger κ, or float32: `IDEAS.md` §5 item 8). No stopping rule yet: the run is scored at ~1000 s; the record on this
track is submission 2 (37.5 s; 37.8 s re-timed on the same GPU, `previous.json`).
