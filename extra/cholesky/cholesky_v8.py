"""Cholesky v8: v7 + blocked TRSM.

v7's hot path is the triangular solve `X @ L_II.T = T` — for b=256 that
custom kernel hits ~55 GFLOPS because the inner `l < k` mask halves the
effective work and forces scalar writes. We split the solve into a chain
of small (b_sub = 32) TRSMs separated by tinygrad matmuls, so the bulk
of the work gets done by tinygrad's tiled GEMM (2+ TFLOPS) instead of the
custom forward-sub kernel.

Algorithm (block TRSM, solving X @ L.T = T for L lower triangular b×b):
  for k in 0..NB_sub-1:
    X_k  = small_trsm( T_cur[:, k*b_sub:(k+1)*b_sub],
                       L[k*b_sub:(k+1)*b_sub, k*b_sub:(k+1)*b_sub] )
    T_cur[:, (k+1)*b_sub:]  -=  X_k @ L[(k+1)*b_sub:, k*b_sub:(k+1)*b_sub].T
  return concat(X_0 .. X_{NB_sub-1})

Everything else (outer block-row loop, small Cholesky on the diagonal
block, trailing GEMM update) is identical to v7.
"""
from __future__ import annotations
import os
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

try: from cholesky_v5 import cholesky as _small_chol
except ImportError: from .cholesky_v5 import cholesky as _small_chol

B       = int(os.environ.get("CHOL_B",       128))
B_SUB   = int(os.environ.get("CHOL_BSUB",    32))
TRSM_TI = int(os.environ.get("CHOL_TRSM_T",  256))


def small_trsm_kernel(X: UOp, T_in: UOp, L_tri: UOp) -> UOp:
  """Solve X @ L_triᵀ = T_in.  L_tri is a small (b_sub × b_sub) lower-triangular block."""
  M, b = X.shape[0], X.shape[1]
  assert T_in.shape == (M, b) and L_tri.shape == (b, b)
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


def _small_trsm(T_in: Tensor, L_tri: Tensor) -> Tensor:
  X = Tensor.empty(T_in.shape[0], T_in.shape[1], dtype=T_in.dtype, device=T_in.device)
  return Tensor.custom_kernel(X, T_in, L_tri, fxn=small_trsm_kernel)[0]


def _block_trsm(T_in: Tensor, L_tri: Tensor, b_sub: int) -> Tensor:
  """Solve X @ L_triᵀ = T_in, b_sub-blocked so most work is GEMMs."""
  M, b = T_in.shape
  assert L_tri.shape == (b, b)
  if b <= b_sub:
    return _small_trsm(T_in, L_tri)
  assert b % b_sub == 0, f"b={b} must be a multiple of b_sub={b_sub}"
  NB = b // b_sub

  X_parts = []
  T_cur = T_in
  for k in range(NB):
    ks, ke = k * b_sub, (k + 1) * b_sub
    T_k    = T_cur[:, ks:ke]
    L_kk   = L_tri[ks:ke, ks:ke]
    X_k    = _small_trsm(T_k, L_kk)
    X_parts.append(X_k)
    if k < NB - 1:
      L_col   = L_tri[ke:, ks:ke]                    # (b - ke, b_sub)
      update  = X_k @ L_col.transpose(-1, -2)        # (M, b - ke)
      T_tail  = T_cur[:, ke:] - update
      T_cur   = Tensor.cat(T_cur[:, :ke], T_tail, dim=1)
  return Tensor.cat(*X_parts, dim=1)


def blocked_cholesky(A: Tensor, block: int | None = None) -> Tensor:
  """Right-looking blocked Cholesky with multi-kernel block-TRSM."""
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

    # Trailing update via Tensor matmul.
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
      # Block TRSM: many small TRSMs + matmuls, total ≈ O(b/b_sub) GEMMs.
      L_off = _block_trsm(S[b:, :], L_II, B_SUB)
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
    print(f"[{os.environ.get('DEV', A.device)}]  B={B} b_sub={B_SUB}  N={N:4d}  max_err={err:.3e}")
