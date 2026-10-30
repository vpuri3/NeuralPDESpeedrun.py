"""Matrix-free Krylov solvers (PCG, right-preconditioned restarted GMRES) on flat tensors."""
from __future__ import annotations

import math

import torch


def _dot(a, b):
    return (a.conj() * b).sum() if a.is_complex() else (a * b).sum()


def pcg(A, b, M=None, x0=None, rtol=1e-10, maxiter=1000):
    """Preconditioned conjugate gradients for SPD A. A, M: callables on tensors shaped like b."""
    M = M or (lambda r: r)
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A(x)
    z = M(r)
    p = z.clone()
    rz = _dot(r, z)
    bn = float(torch.linalg.norm(b))
    hist = []
    for it in range(maxiter):
        Ap = A(p)
        alpha = rz / _dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rn = float(torch.linalg.norm(r)) / bn
        hist.append(rn)
        if rn < rtol:
            break
        z = M(r)
        rz_new = _dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, hist


def gmres(A, b, M=None, x0=None, rtol=1e-10, restart=60, maxiter=2000):
    """Right-preconditioned restarted GMRES:  A M y = b,  x = M y.  Complex arithmetic."""
    M = M or (lambda r: r)
    cdt = torch.complex128
    b = b.to(cdt)
    x = torch.zeros_like(b) if x0 is None else x0.to(cdt).clone()
    bn = float(torch.linalg.norm(b))
    hist = []
    total = 0
    while total < maxiter:
        r = b - A(x)
        beta = float(torch.linalg.norm(r))
        if beta / bn < rtol:
            break
        V = [r / beta]
        H = torch.zeros(restart + 1, restart, dtype=cdt, device=b.device)
        cs = torch.zeros(restart, dtype=cdt, device=b.device)
        sn = torch.zeros(restart, dtype=cdt, device=b.device)
        g = torch.zeros(restart + 1, dtype=cdt, device=b.device)
        g[0] = beta
        k_used = 0
        for k in range(restart):
            w = A(M(V[k]))
            for i in range(k + 1):                    # modified Gram-Schmidt
                H[i, k] = _dot(V[i], w)
                w = w - H[i, k] * V[i]
            H[k + 1, k] = torch.linalg.norm(w)
            V.append(w / H[k + 1, k])
            for i in range(k):                        # apply previous Givens rotations
                t = cs[i].conj() * H[i, k] + sn[i].conj() * H[i + 1, k]
                H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                H[i, k] = t
            a, bb = H[k, k], H[k + 1, k]
            den = math.sqrt(abs(complex(a)) ** 2 + abs(complex(bb)) ** 2)
            cs[k] = a / den
            sn[k] = bb / den
            H[k, k] = cs[k].conj() * a + sn[k].conj() * bb
            H[k + 1, k] = 0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k].conj() * g[k]
            k_used = k + 1
            total += 1
            res = abs(complex(g[k + 1])) / bn
            hist.append(res)
            if res < rtol or total >= maxiter:
                break
        # back substitution
        y = torch.zeros(k_used, dtype=cdt, device=b.device)
        for i in range(k_used - 1, -1, -1):
            s = g[i] - (H[i, i + 1:k_used] * y[i + 1:k_used]).sum()
            y[i] = s / H[i, i]
        dx = sum(y[i] * V[i] for i in range(k_used))
        x = x + M(dx)
        if hist and hist[-1] < rtol:
            break
    return x, hist
