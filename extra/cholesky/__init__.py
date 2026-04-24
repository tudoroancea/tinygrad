"""Codegenerable Cholesky factorization for tinygrad.

Entrypoint:  cholesky(A: Tensor) -> Tensor  (lower-triangular factor L such that A = L Lᵀ)

Single-kernel variants (v1-v5) fit in one custom_kernel invocation.
Multi-kernel variants (v7-v10) chain many kernels but keep everything in
one UOp graph — tinygrad schedules them as a single pipeline with no
.realize() in between, so the "codegenerable in one go" constraint holds.
"""
from __future__ import annotations
from tinygrad import Tensor

from .cholesky_v1  import cholesky as _v1
from .cholesky_v2  import cholesky as _v2   # noqa: F401
from .cholesky_v3  import cholesky as _v3   # noqa: F401
from .cholesky_v4  import cholesky as _v4   # noqa: F401
from .cholesky_v5  import cholesky as _v5
from .cholesky_v7  import cholesky as _v7   # noqa: F401
from .cholesky_v8  import cholesky as _v8   # noqa: F401
from .cholesky_v9  import cholesky as _v9   # noqa: F401
from .cholesky_v10 import cholesky as _v10


def _has_local(device: str) -> bool:
  root = device.split(":", 1)[0].upper()
  return root in {"NV", "CUDA", "AMD", "METAL", "HIP", "CL", "GPU", "QCOM", "WEBGPU"}


def cholesky(A: Tensor) -> Tensor:
  """Lower-triangular Cholesky factor of SPD A."""
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "square matrix required"
  N = A.shape[0]
  dev = A.device if isinstance(A.device, str) else A.device[0]
  if not _has_local(dev):
    return _v1(A)                           # CPU / PYTHON
  if N < 256 or N % 128 != 0:
    return _v5(A)                           # small square: 1-kernel version has no launch overhead
  return _v10(A)                            # big square: multi-kernel blocked Cholesky
