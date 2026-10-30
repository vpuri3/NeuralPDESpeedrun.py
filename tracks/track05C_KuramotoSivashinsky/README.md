# track05C_KuramotoSivashinsky: Kuramoto-Sivashinsky (chaotic)

```text
Kuramoto-Sivashinsky equation on x in [0, 2 pi] (periodic), t in [0, 1], in the chaotic regime:

    u_t + a u u_x + b u_xx + c u_xxxx = 0,      a = 100/16,  b = 100/16^2,  c = 100/16^4

Initial condition: u(0, x) is given as samples on the output x grid (problem.fns["ic"]; problem.fns["u0"] is
its trigonometric interpolant). Small errors grow exponentially; the bar reflects that.

Output: u at t_k = k / 250, k = 1..250, and x_j = 2 pi j / 511, j = 0..511; shape (1, 250, 512).
```

**Accuracy bar:** RelL2 ≤ 0.05, L∞ ≤ 0.5.

![reference](figures/reference.png)
![reference](figures/reference.gif)

## Reference

published data (jaxpi)


## Records

No records yet.
