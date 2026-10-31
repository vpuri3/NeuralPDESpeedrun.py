# track15_2DT_GreyScott / submission_PR0002: PINN + Gauss-Newton (D-NGD)

Method: see [speedrun/submissions/submission_PR0002/README.md](../../../../speedrun/submissions/submission_PR0002/README.md).
The residual and the per-track settings are in `solve.py`. What changed vs. the previous record: Gauss-Newton
(natural-gradient) training of sine MLPs in float64 (time-windowed with a hard initial condition on
time-dependent tracks), instead of SOAP on one PirateNet.
