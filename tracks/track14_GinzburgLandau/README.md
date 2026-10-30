# track14_GinzburgLandau: Ginzburg-Landau

```text
Complex Ginzburg-Landau equation, written for A = u + i v, on (x, y) in [-1, 1]^2 (periodic), t in [0, 1]:

    u_t = eps Lap u + k ( u - (u^2 + v^2) u + 1.5 (u^2 + v^2) v )
    v_t = eps Lap v + k ( v - (u^2 + v^2) v - 1.5 (u^2 + v^2) u ),        eps = 0.004, k = 10

Initial condition: (u, v)(0, x, y) given as samples on the output grid (problem.fns["ic"]; problem.fns["u0"],
["v0"] are their trigonometric interpolants).

Output: (u, v) at t_k = 0.01 k, k = 1..100, on the 200 x 200 grid x_i, y_j = -1 + 2 i / 199;
shape (2, 100, 200, 200).
```

**Accuracy bar:** RelL2 ≤ 0.01, L∞ ≤ 0.05.

![reference](figures/reference.png)
![reference](figures/reference.gif)

## Reference

published data (jaxpi)


## Records

No records yet.
