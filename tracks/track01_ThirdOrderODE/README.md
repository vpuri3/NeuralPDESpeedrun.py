# track01_ThirdOrderODE: 3rd-order ODE  u''' = cos(pi x)

```text
Third-order boundary value problem on x in [0, 1]:

    u'''(x) = cos(pi x)
    u(0) = 0,   u(1) = cos(pi) = -1,   u'(1) = 1

Output: u at the 1001 uniformly spaced points x = linspace(0, 1, 1001).
```

**Accuracy bar:** RelL2 ≤ 0.0001, L∞ ≤ 0.001.

![reference](figures/reference.png)

## Reference

closed form

- `check_chebyshev_N32_linf`: 2.976e-11

## Records

Current record: **0.001486 s** (record #0: classical baseline, maintainers, 2026-10-30, NVIDIA GeForce RTX 2080 Ti).

| date | author | name | kind | time [s] | RelL2 | L∞ | passes | hardware |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | maintainers | record #0: classical baseline | classical | 0.001486 | 3.35e-13 | 6.22e-13 | yes | NVIDIA GeForce RTX 2080 Ti |

![accuracy_vs_time](figures/accuracy_vs_time.png)

![activity](figures/activity.png)
