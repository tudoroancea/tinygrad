"""Cholesky v11: v10 + pad-based L assembly (no explicit zero tensors).

v10 emits ~240 tiny elementwise E_ kernels per factorisation just to build
the zero-padding tensors it concatenates into L at every block step. v11
replaces the cat-of-zeros with `Tensor.pad`, which is a pure movement op
in tinygrad and doesn't materialise the zero halo. Same algorithm
(right-looking blocked + recursive L⁻ᵀ), same correctness, fewer kernels.
"""
from __future__ import annotations
import os
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

try: from cholesky_v5 import cholesky as _small_chol
except ImportError: from .cholesky_v5 import cholesky as _small_chol

B        = int(os.environ.get("CHOL_B",       64))
BASE_B   = int(os.environ.get("CHOL_BASE_B",  32))
TRSM_TI  = int(os.environ.get("CHOL_TRSM_T",  256))


def small_trsm_kernel(X: UOp, T_in: UOp, L_tri: UOp) -> UOp:
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
  return ended.sink(arg=KernelInfo(name=f"trsm_base_{M}x{b}", opts_to_apply=()))


def _trsm(T_in: Tensor, L_tri: Tensor) -> Tensor:
  X = Tensor.empty(T_in.shape[0], T_in.shape[1], dtype=T_in.dtype, device=T_in.device)
  return Tensor.custom_kernel(X, T_in, L_tri, fxn=small_trsm_kernel)[0]


def _inv_T(L: Tensor) -> Tensor:
  b = L.shape[0]
  if b <= BASE_B:
    I = Tensor.eye(b, dtype=L.dtype, device=L.device)
    return _trsm(I, L)
  m = b // 2
  A = L[:m, :m]
  Bblk = L[m:, :m]
  C = L[m:, m:]
  A_inv_T = _inv_T(A)
  C_inv_T = _inv_T(C)
  off_diag = -A_inv_T @ Bblk.transpose(-1, -2) @ C_inv_T
  # Assemble via pad + add (no explicit zero tensors materialise).
  # Top row block: [A_inv_T, off_diag] padded to (b, b) with zeros below.
  top = A_inv_T.pad(((0, m), (0, m))) + off_diag.pad(((0, m), (m, 0)))
  bot = C_inv_T.pad(((m, 0), (m, 0)))
  return top + bot


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
      L_II_inv_T = _inv_T(L_II)
      L_off = S[b:, :] @ L_II_inv_T
      parts.append(L_off)

    # Build the full N x b block-column with the computed pieces, then pad to
    # (N, N) via `.pad` (no explicit zero tensors — pad is a movement op).
    if I < NB - 1:
      col_I = Tensor.cat(L_II, L_off, dim=0)                     # (N - I_start, b)
    else:
      col_I = L_II                                                # (b, b)
    # Pad top with I_start zero rows (leaves bottom unchanged).
    col_I = col_I.pad(((I_start, 0), (0, 0)))                    # (N, b)
    # Pad left/right to full N columns.
    col_full = col_I.pad(((0, 0), (I_start, (NB - I - 1) * b)))  # (N, N)

    L = L + col_full

  return L


cholesky = blocked_cholesky


if __name__ == "__main__":
  import numpy as np
  np.random.seed(0)
  for N in [64, 128, 256, 512, 1024, 2048, 4096]:
    X = np.random.randn(N, N).astype(np.float32)
    A_np = X @ X.T + N * np.eye(N, dtype=np.float32)
    A = Tensor(A_np)
    L = cholesky(A).numpy()
    ref = np.linalg.cholesky(A_np)
    err = float(np.abs(L - ref).max())
    print(f"[{os.environ.get('DEV', A.device)}]  B={B} base={BASE_B}  N={N:4d}  max_err={err:.3e}")
