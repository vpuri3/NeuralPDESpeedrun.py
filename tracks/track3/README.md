# track3: 2D high-contrast diffusion (contrast 1e3)

```text
Steady diffusion with a discontinuous, high-contrast coefficient on the unit square [0, 1]^2:

    -div( a(x, y) grad u ) = f,        u = 0 on the boundary
    a = 1000 inside 9 square inclusions [s_i, e_i] x [s_j, e_j],  a = 1 elsewhere,
        (s, e) in {(2/16, 4/16), (7/16, 9/16), (12/16, 14/16)}
    f = 18

(Flux continuity a grad u . n across the inclusion edges; u and the normal flux are continuous.)

Output: u at the 256 x 256 cell centers x_i = y_i = (i + 1/2) / 256; shape (1, 256, 256).
```

![reference](figures/reference.png)

## Reference

cell-centered FV + MG-PCG, N=2304 (175 PCG its)

- `check_N768_vs_N2304_linf`: 8.879e-04
- `check_N768_vs_N2304_rel_l2`: 2.101e-04

## Records

Current record: **0.172 s** (record #0: classical baseline, maintainers, 2026-10-30, NVIDIA GeForce RTX 2080 Ti).

| date | author | name | kind | time [s] | RelL2 | L∞ | hardware |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | maintainers | record #0: classical baseline | classical | 0.172 | 1.13e-03 | 4.58e-03 | NVIDIA GeForce RTX 2080 Ti |

![activity](figures/activity.png)
