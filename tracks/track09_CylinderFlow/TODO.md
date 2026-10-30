# track09_CylinderFlow: flow past a cylinder (TODO, not set up)

Notes for setting this track up later. Nothing below is decided.

- Physics: 2D incompressible Navier-Stokes around a circular cylinder in a channel. Candidates:
  - steady, Re = 20 (Schäfer & Turek 1996 benchmark 2D-1: drag/lift coefficients, recirculation length);
  - unsteady, Re = 100 (benchmark 2D-2/2D-3: periodic vortex shedding; Strouhal number, max drag/lift).
- Geometry: channel [0, 2.2] x [0, 0.41], cylinder of diameter 0.1 centered at (0.2, 0.2) (Schäfer-Turek), or
  a larger domain with a centered cylinder.
- BCs: parabolic inflow, no-slip walls and cylinder, outflow (do-nothing / zero-stress).
- Scoring: field values at fixed points (what every other track scores) and/or integral quantities
  (C_D, C_L, St). Point values of a shedding wake are phase-sensitive; integral quantities may be the
  better bar for the unsteady case.
- Reference: needs a body-fitted solver, e.g. a multi-element version of `speedrun/numerics/sem.py`
  (Gordon-Hall curved elements around the cylinder) or published benchmark values.
- Why it is interesting: curved geometry plus a wake makes it a candidate for mesh-free / hybrid methods.
