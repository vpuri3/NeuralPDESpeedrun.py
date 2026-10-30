"""Classical numerical building blocks (all PyTorch) used by reference solvers and baselines."""
from .cheb import bary_matrix, cheb, clenshaw_curtis_weights  # noqa: F401
from .fourier import (dealias_mask, dst, dst2, etdrk4_coefficients, etdrk4_step, idst,  # noqa: F401
                      idst2, wavenumbers)
from .krylov import gmres, pcg  # noqa: F401
