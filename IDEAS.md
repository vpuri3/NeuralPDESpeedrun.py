# IDEAS.md — data-free neural PDE solvers

Working notes on solving PDEs with neural/explicit ansätze **without ever seeing a solution**: true
solvers, not surrogates. The speedrun (README.md, PLAN.md) is the testbed.

## 0. Framing

Two different problems hide under "neural PDE solver":

1. **Per-instance solve.** Pick an ansatz, minimize a PDE residual/energy for this one problem (PINNs,
   Neural Galerkin, explicit fields). This competes head-on with FEM/spectral on a single instance.
2. **Amortized solve.** Learn a *solver* (an iterative update, a preconditioner, a coarse correction)
   over a family of problems, trained only on equations. This is where looped transformers / FLARE fit.

The speedrun scores (1) directly. (2) enters through hybrids and, later, a possible amortized track.

## 1. Why PINNs fail: conditioning first, then representation

- The squared residual loss squares the operator's condition number. For a Laplacian at resolution h,
  κ ~ h⁻², so the loss sees κ² ~ h⁻⁴. Rathore et al. show the PINN loss is ill-conditioned *because of the
  differential operator*, and quasi-Newton methods improve conditioning by ≥1000×.
  [Challenges in training PINNs](https://arxiv.org/html/2402.01868v1)
- With function-space preconditioning (natural gradient / Gauss–Newton / energy NG), PINNs reach near
  single-precision accuracy. The remaining blocker is the cubic cost of those methods at scale.
  [Müller & Zeinhofer et al., scalable NG](https://arxiv.org/pdf/2505.21404v1.pdf),
  [Curvature-aware optimization for high-accuracy PINNs (2026)](https://www.alphaxiv.org/abs/2604.05230.md),
  [DSGNAR, doubly-sketched Gauss–Newton (2026)](https://papers.cool/arxiv/2607.02194),
  [Randomized Nyström-preconditioned NGD](https://simai2025.cimne.com/event/contribution/05c9f932-32fc-11f0-a5f6-000c29ddfc0c)
- SOAP (quasi-second-order) gives state-of-the-art PINN results, including turbulent flows up to Re = 10⁴.
  [Gradient alignment in PINNs (NeurIPS 2025)](https://arxiv.org/html/2502.00604v2)
- Neural Galerkin (Bruna, Peherstorfer, Vanden-Eijnden; Berman & Peherstorfer): sequential-in-time
  Dirac–Frenkel training. Errors accumulate and amplify over time steps, and randomized sparse parameter
  updates help.
  [NG with active learning](https://pdf.arxiv.org/pdf/2203.01360),
  [Randomized sparse NG](https://pdf.arxiv.org/pdf/2310.04867)

**Takeaway.** "One global network for all of space" is a real issue (spectral bias, no locality), but
it's second to conditioning. A local representation alone doesn't fix κ². It needs a preconditioner.

## 2. Local / explicit ansätze (Instant-NGP, Gaussians)

Prior work:
- Hash-encoded PINNs (Huang & Alkhalifah 2023): about 10× faster than vanilla PINNs, but hash-grid
  derivatives are discontinuous, so they fall back to **finite differences** for the residual.
  [arXiv:2302.13397](https://www.arxiv.org/abs/2302.13397)
- Hermite-NGP (2026): stores values plus mixed derivatives at the grid vertices, giving analytic gradients
  and Hessians by Hermite interpolation, plus a multigrid-V-cycle-style coarse-to-fine curriculum.
  [arXiv:2605.24774](https://arxiv.org/pdf/2605.24774)
- Hash-grid neural simulation (AAAI) also uses FD inside the grid.
  [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28349/28684)
- Gaussian representations for PDEs/fluids: [Gaussian fluids (2024)](https://arxiv.org/abs/2405.18133v1),
  [Physics-informed Gaussian splatting thesis (TU Delft)](https://repository.tudelft.nl/record/uuid:7069ec01-9b5e-4957-9af4-850c4cf32d20),
  [FFGS operator learning (2026)](https://www.alphaxiv.org/abs/2608.24049).

**Gap: nobody has exploited the sparsity that locality gives the Jacobian/Hessian on PDE problems.**
Graphics already does this for image fitting:
[3DGS-LM (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.pdf),
[sparse-Jacobian LM for splats](https://arxiv.org/html/2504.12905v1).

### Proposal A: multilevel explicit ansatz + sparse Gauss–Newton
- **Representation.** Dense (unhashed) multiresolution grid of B-spline/Hermite coefficients, plus an
  optional tiny shared MLP decoder. No hashing, because hash collisions destroy locality.
- **Derivatives.** Analytic: B-spline/Hermite basis derivatives, forward-mode through the decoder.
  **Never FD.**
- **Structure of JᵀJ.**
  - No decoder and a linear PDE: residual linear in the parameters → one sparse least-squares solve
    (= least-squares collocation with a multilevel spline basis). This is the honest classical baseline.
  - With decoder: arrowhead (sparse block over grid features + dense border over the MLP weights) →
    Schur complement onto the small MLP block, PCG on the sparse block.
  - Gaussians: positions/scales give r-adaptivity, also with local Jacobian blocks. Mesh-tangling-like
    degeneracies need regularization.
- **Preconditioning.** The multiresolution levels already form a hierarchical basis, giving
  BPX/additive-Schwarz preconditioning for the inner CG almost for free. This is what fixes κ².
- **Where it could beat FEM.** Transport-dominated problems with moving features (nonlinear
  approximation, Kolmogorov n-width; this is the NeuralROMs/SNF-ROM argument), adaptivity, and high
  dimensions. On smooth 2D elliptic problems FEM+AMG is very hard to beat.

## 3. Transformers that predict jointly: computing the residual

For self-attention over the query points, u_i = F_θ(x_i; x_1..x_N) depends on the other points as well
as on θ. Two consequences:
1. Autodiff ∂u_i/∂x_i includes paths through other tokens' keys/values. That is **not** the spatial
   derivative of a field.
2. Resampling collocation points changes the function itself, so the residual is a moving target.

**Fix: separate context from query.** In FLARE, latents Z^l = Attn(latent queries, tokens^{l-1}) and
token updates h_i^l = f^l(h_i^{l-1}, Z^l). If the latent stack {Z^l} is computed once from a *fixed*
context (problem description: coefficients, geometry, forcing; or a fixed point set) and treated as
constant with respect to coordinates (not θ), every query point follows its own pointwise path. That
gives a well-defined field u(x; Z, θ) with exact, cheap derivatives (forward/Taylor-mode on the query
path only). The latents carry the global, joint information, like global DOFs.

Alternative: output nodal values on a mesh and use an assembled FE weak-form residual r = f − A(U).
Joint prediction is then harmless, and this is a proper Galerkin discretization (not FD on a neural field).

## 4. Data-free training of FLARE / looped transformers (amortized solvers)

Looped transformer as a learned iterative method: **u_{k+1} = u_k + T_θ(u_k, r_k, problem)**,
r_k = f − A(u_k).
- **Correct by construction.** If the update is driven by the residual, its fixed point is the exact
  discrete solution. Accuracy is limited by iteration count, not network approximation error (unlike
  surrogates). More loops give lower residual: test-time compute scaling with a verifiable target.
- **FLARE ↔ multigrid.** Local operator = smoother. FLARE's low-rank latent routing = coarse-space
  correction (global communication at linear cost, like a learned two-level/Nyström coarse space). One
  loop ≈ one V-cycle.
- **Data-free training signals.**
  1. **Residual over a problem sampler.** Sample coefficients/forcing/geometry; minimize ‖r_K‖ after K
     loops (energy norm for SPD). Use random K and later-loop weighting so the network learns to keep
     contracting. Random white-noise RHS covers all frequencies
     ([MGCNN](https://arxiv.org/pdf/2312.11093)).
  2. **Manufactured solutions.** Sample u* from a broad random field and compute f = L(u*) exactly. That
     gives unlimited supervised pairs with no solver. For iterative solvers the true error e_k = u* − u_k is
     known at every loop, so the update can be trained on a *well-conditioned error loss* instead of the
     κ²-squared residual. Caveat: the induced f distribution is synthetic, so the u* spectrum must cover
     what you will solve. Works for nonlinear L too.
  3. **Self-distillation through test-time compute.** More loops (lower, checkable residual) teach fewer
     loops.
- **Prior art to beat:** [UGrid (ICML 2024)](https://proceedings.mlr.press/v235/han24a.html) (provable
  convergence, unsupervised residual loss), [Learning to optimize multigrid](https://ar5iv.org/html/1902.10248),
  [Hsieh et al. 2019](https://ar5iv.labs.arxiv.org/html/1906.01200),
  [NPSolver (2026)](https://sotaverified.org/papers/260525786),
  [NLAFormer: transformers express CG](https://arxiv.org/pdf/2508.19557),
  [Looped transformers learn learning algorithms](https://arxiv.org/pdf/2311.12424).
- **The metric that matters.** Mesh-independent convergence: iterations to tolerance stay flat as
  h → 0. Report error vs. wall-clock against PETSc/hypre AMG on the same GPU.
- **Targets where classical solvers struggle.** High-frequency Helmholtz (indefinite, AMG fails),
  convection-dominated flow, implicit time steps inside Newton for Navier–Stokes.

## 5. Research program (ordered)

1. **Speedrun harness** (this repo) with honest classical baselines.
2. **First neural records.** PINN + Adam/L-BFGS, then SOAP/NG on easy tracks (1, 2A, 6, 7A), to quantify
   the gap.
3. **Proposal A** (multilevel explicit ansatz + sparse GN + level-preconditioned CG) on tracks 3, 4A, 4B.
4. **Hybrid in-run learning.** Classical smoother + learned coarse correction trained during the solve
   from residuals only (tracks 3, 4A).
5. **Amortized FLARE/looped solver** trained data-free (residual + manufactured solutions), plugged in as
   the coarse correction. Measure mesh-independence and loop-count scaling at million-scale meshes.
6. **Transport/moving features** (5A, 5B, 5C): adaptive Gaussian / neural-field Galerkin in time
   (NeuralROMs line) with sparse GN per step.
