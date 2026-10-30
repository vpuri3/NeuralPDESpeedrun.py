"""Record #0 (classical baseline): Fourier pseudo-spectral vorticity-streamfunction form, N = 32,
2/3 dealiasing, ETDRK4 with dt = 0.05, Fourier interpolation to the output grid."""
import torch

from speedrun.numerics.periodic import navier_stokes_2d

N = 32
SUBSTEPS = 10


def fourier_solve(p, N, substeps=SUBSTEPS):
    L = p.domain["x"][1]
    g = L * torch.arange(N, dtype=p.dtype, device=p.device) / N
    X, Y = torch.meshgrid(g, g, indexing="ij")
    out = navier_stokes_2d(p.fns["u0"](X, Y), p.fns["v0"](X, Y), L, N, p.params["nu"], p.axes["t"], substeps,
                           dtype=p.dtype, device=p.device)                      # (nt, 2, N, N)
    M = p.axes["x"].numel()                                                  # zero-pad spectrum to M x M
    uh = torch.fft.fft2(out)
    U = torch.zeros(*uh.shape[:2], M, M, dtype=uh.dtype, device=uh.device)
    h = N // 2
    for a in (slice(0, h), slice(-h, None)):
        for b in (slice(0, h), slice(-h, None)):
            U[..., a, b] = uh[..., a, b]
    return (torch.fft.ifft2(U).real * (M / N) ** 2).permute(1, 0, 2, 3).contiguous()


def solve(problem):
    return fourier_solve(problem, N)
