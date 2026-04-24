"""Cholesky v9: blocked + "inverse-and-multiply" TRSM.

v7's M×b TRSM is the bottleneck because forward substitution on a wide
right-hand side is hard to vectorise in our simple custom kernel. But:

    X @ L_IIᵀ = T    ⇔    X = T @ (L_II⁻ᵀ)

If we compute `L_II⁻ᵀ` (a b×b upper triangular) once per block step, the
rest of the work is a single large matmul that tinygrad's tiled GEMM
handles at 2+ TFLOPS. The tiny `b × b` TRSM against an identity matrix is
O(b³/2), tiny next to the M × b² main cost.

This collapses the two hot custom kernels (big TRSM → small TRSM vs I + GEMM).
"""
from __future__ import annotations
import os
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

try: from cholesky_v5 import cholesky as _small_chol
except ImportError: from .cholesky_v5 import cholesky as _small_chol

B        = int(os.environ.get("CHOL_B", 256))
TRSM_TI  = int(os.environ.get("CHOL_TRSM_T", 256))


def small_trsm_kernel(X: UOp, T_in: UOp, L_tri: UOp) -> UOp:
  """Solve X @ L_triᵀ = T_in, L_tri lower triangular.  (Same as v7/v8.)"""
  M, b = X.shape[0], X.shape[1]
  dtype = X.dtype.base
  zero = UOp.const(dtype, 0.0)

  t_i = TRSM_TI
  while M % t_i != 0 and t_i > 1: t_i //= 2
  tid      = UOp.special(t_i, "lidx0")
  i_chunks = M // t_i
  i_outer  = UOp.range(i_chunks, 0, AxisType.REDUCE)
  i_my     = i_outer * t_i + tid

  x_reg = UOp.placeholder((b,), dtype, 0, addrspace=AddrSpace.REG)
  k = UOp.range(b, 1, AxisType.REDUCE)
  l = UOp.range(b, 2, AxisType.REDUCE)
  sum_reg = UOp.placeholder((1,), dtype, 1, addrspace=AddrSpace.REG)
  sum_reg = sum_reg.after(i_outer, k)[0].set(zero)
  contrib = (l < k).where(L_tri[k, l] * x_reg.after(i_outer, k)[l], zero)
  sum_reg = sum_reg[0].set(sum_reg.after(l)[0] + contrib, end=l)

  x_val    = (T_in[i_my, k] - sum_reg[0]) / L_tri[k, k]
  x_reg_w  = x_reg.after(i_outer, k)[k].store(x_val)
  x_out_w  = X[i_my, k].store(x_val)
  ended    = UOp.group(x_reg_w, x_out_w).end(k).end(i_outer)
  return ended.sink(arg=KernelInfo(name=f"trsm_sub_{M}x{b}", opts_to_apply=()))


def _trsm(T_in: Tensor, L_tri: Tensor) -> Tensor:
  X = Tensor.empty(T_in.shape[0], T_in.shape[1], dtype=T_in.dtype, device=T_in.device)
  return Tensor.custom_kernel(X, T_in, L_tri, fxn=small_trsm_kernel)[0]


def _inv_upper_T(L_tri: Tensor) -> Tensor:
  """Return L_tri⁻ᵀ (upper-triangular inverse transpose) by solving X @ L_triᵀ = I."""
  b = L_tri.shape[0]
  I = Tensor.eye(b, dtype=L_tri.dtype, device=L_tri.device)
  return _trsm(I, L_tri)


def blocked_cholesky(A: Tensor, block: int | None = None) -> Tensor:
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1]
  N = A.shape[0]
  b = block if block is not None else B
  if N % b != 0 or N <= b:
    return _small_chol(A)
  NB = N // b

  dtype, device = A.dtype, A.device
  L = Tensor.zeros(N, N, dtype=dtype, device=device)

  for I in range(NB):
    I_start, I_end = I * b, (I + 1) * b

    A_rest = A[I_start:, I_start:I_end]
    if I > 0:
      L_rest_prior  = L[I_start:, :I_start]
      L_row_I_prior = L[I_start:I_end, :I_start]
      S = A_rest - L_rest_prior @ L_row_I_prior.transpose(-1, -2)
    else:
      S = A_rest

    L_II = _small_chol(S[:b, :])

    parts = [L_II]
    if I < NB - 1:
      L_II_inv_T = _inv_upper_T(L_II)          # b × b (O(b³/2) tiny TRSM on identity)
      L_off = S[b:, :] @ L_II_inv_T             # big fast tinygrad matmul
      parts.append(L_off)

    top = Tensor.zeros(I_start, b, dtype=dtype, device=device)
    col_I = Tensor.cat(top, *parts, dim=0) if I > 0 else Tensor.cat(*parts, dim=0)

    lr = []
    if I > 0:            lr.append(Tensor.zeros(N, I_start, dtype=dtype, device=device))
    lr.append(col_I)
    if I < NB - 1:       lr.append(Tensor.zeros(N, (NB - I - 1) * b, dtype=dtype, device=device))
    col_full = Tensor.cat(*lr, dim=1) if len(lr) > 1 else lr[0]

    L = L + col_full

  return L


cholesky = blocked_cholesky


if __name__ == "__main__":
  import numpy as np
  np.random.seed(0)
  for N in [64, 128, 256, 512, 1024, 2048]:
    X = np.random.randn(N, N).astype(np.float32)
    A_np = X @ X.T + N * np.eye(N, dtype=np.float32)
    A = Tensor(A_np)
    L = cholesky(A).numpy()
    ref = np.linalg.cholesky(A_np)
    err = float(np.abs(L - ref).max())
    print(f"[{os.environ.get('DEV', A.device)}]  B={B}  N={N:4d}  max_err={err:.3e}")
