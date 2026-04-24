"""Codegenerable Cholesky factorization for tinygrad.

Entrypoint:  cholesky(A: Tensor) -> Tensor  (lower-triangular factor L such that A = L Lᵀ)

Internally dispatches to the fastest variant that matches the target device and
the matrix size. All variants are single custom kernels — no .realize() in the
compute, no python-level kernel dispatch loops.
"""
from __future__ import annotations
from tinygrad import Tensor

from .cholesky_v1 import cholesky as _v1
from .cholesky_v2 import cholesky as _v2  # noqa: F401  (kept for bench/comparison)
from .cholesky_v3 import cholesky as _v3
from .cholesky_v4 import cholesky as _v4
from .cholesky_v5 import cholesky as _v5


def _has_local(device: str) -> bool:
  # Rough heuristic: GPU/accel devices support __shared__ / LDS + threads.
  root = device.split(":", 1)[0].upper()
  return root in {"NV", "CUDA", "AMD", "METAL", "HIP", "CL", "GPU", "QCOM", "WEBGPU"}


def cholesky(A: Tensor) -> Tensor:
  """Compute the lower-triangular Cholesky factor L of a SPD matrix A."""
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "square matrix required"
  N = A.shape[0]
  dev = A.device if isinstance(A.device, str) else A.device[0]
  if _has_local(dev) and N >= 16:
    # v5 is the fastest GPU variant — 2D thread tile, right-looking.
    # Fall back to v1 for tiny matrices where tile/barrier overhead dominates.
    return _v5(A)
  # CPU, PYTHON, or tiny N: v1's simple triple-loop is fine.
  return _v1(A)
