"""Cholesky v7: blocked / multi-kernel right-looking Cholesky.

Uses a Python-side for loop over block rows to *build* a UOp graph — no
.realize() between iterations, so tinygrad still schedules every kernel in
one pipeline. The per-block payload is:

  1. Trailing update   S = A[I:, I:I+B] − L[I:, :I] @ L[I:I+B, :I]ᵀ
                        (one big tinygrad matmul, generated as an optimised GEMM)
  2. Small Cholesky    L[I:I+B, I:I+B] = chol(S[:B])              (custom kernel, reuses v5)
  3. Triangular solve  L[I+B:, I:I+B] = S[B:] @ L[I:I+B, I:I+B]⁻ᵀ (custom kernel below)

Tinygrad fuses the trailing-update GEMM with whatever it can, then runs
the two custom kernels sequentially. With B≈64 and N=1024 the dominant
cost becomes ~NB GEMMs of shape (N−I·B) × I·B × B, which tinygrad handles
with its tiled matmul scheduler — the exact same codegen path as
`Tensor.__matmul__`.
"""
from __future__ import annotations
import os
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

# re-use v5 for the small diagonal block Cholesky
try: from cholesky_v5 import cholesky as _small_chol
except ImportError: from .cholesky_v5 import cholesky as _small_chol

B = int(os.environ.get("CHOL_B", 64))
TRSM_T_I = int(os.environ.get("CHOL_TRSM_T", 256))


def trsm_kernel(X: UOp, T_in: UOp, L_tri: UOp) -> UOp:
  """Build an AST that solves X @ L_triᵀ = T_in, with L_tri lower triangular (B×B).

  Strategy: one thread per row (rows parallel, forward substitution sequential
  within each row). x_reg is a per-thread REG array of size b that persists
  across k iterations and holds the current row's partial solution.
  """
  M, b = X.shape[0], X.shape[1]
  assert T_in.shape == (M, b)
  assert L_tri.shape == (b, b)
  dtype = X.dtype.base
  zero = UOp.const(dtype, 0.0)

  # Pick a thread count that divides M (or match M when small).
  t_i = TRSM_T_I
  while M % t_i != 0 and t_i > 1: t_i //= 2
  tid = UOp.special(t_i, "lidx0")
  i_chunks = M // t_i
  i_outer  = UOp.range(i_chunks, 0, AxisType.REDUCE)
  i_my     = i_outer * t_i + tid

  # Per-thread solution row (b entries).
  x_reg = UOp.placeholder((b,), dtype, 0, addrspace=AddrSpace.REG)

  k = UOp.range(b, 1, AxisType.REDUCE)
  # Inner accumulation: sum_{l<k} L[k, l] * x[l].
  l = UOp.range(b, 2, AxisType.REDUCE)
  sum_reg = UOp.placeholder((1,), dtype, 1, addrspace=AddrSpace.REG)
  sum_reg = sum_reg.after(i_outer, k)[0].set(zero)
  contrib = (l < k).where(L_tri[k, l] * x_reg.after(i_outer, k)[l], zero)
  sum_reg = sum_reg[0].set(sum_reg.after(l)[0] + contrib, end=l)

  x_val = (T_in[i_my, k] - sum_reg[0]) / L_tri[k, k]
  # Two effects: persist into x_reg (for the next k), and emit to X.
  x_reg_w = x_reg.after(i_outer, k)[k].store(x_val)
  x_out_w = X[i_my, k].store(x_val)
  step_end = UOp.group(x_reg_w, x_out_w).end(k)
  ended = step_end.end(i_outer)
  return ended.sink(arg=KernelInfo(name=f"trsm_{M}x{b}", opts_to_apply=()))


def _trsm(T_in: Tensor, L_tri: Tensor) -> Tensor:
  X = Tensor.empty(T_in.shape[0], T_in.shape[1], dtype=T_in.dtype, device=T_in.device)
  return Tensor.custom_kernel(X, T_in, L_tri, fxn=trsm_kernel)[0]


def blocked_cholesky(A: Tensor, block: int | None = None) -> Tensor:
  """Right-looking blocked Cholesky. A must be square SPD with N % block == 0."""
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "square matrix"
  N = A.shape[0]
  b = block if block is not None else B
  if N % b != 0 or N <= b:
    # Fall back to the single-kernel v5 for unsupported sizes.
    return _small_chol(A)
  NB = N // b

  dtype, device = A.dtype, A.device
  zero_tile = Tensor.zeros(N, N, dtype=dtype, device=device)
  L = zero_tile

  for I in range(NB):
    I_start, I_end = I * b, (I + 1) * b

    # --- 1. Trailing update -------------------------------------------------
    A_rest = A[I_start:, I_start:I_end]                       # (N - I*b, b)
    if I > 0:
      L_rest_prior   = L[I_start:, :I_start]                  # (N - I*b, I*b)
      L_row_I_prior  = L[I_start:I_end, :I_start]             # (b, I*b)
      S = A_rest - L_rest_prior @ L_row_I_prior.transpose(-1, -2)
    else:
      S = A_rest

    # --- 2. Small Cholesky on the diagonal block ---------------------------
    L_II = _small_chol(S[:b, :])                              # (b, b)

    # --- 3. Triangular solve for the rows below ----------------------------
    parts = [L_II]
    if I < NB - 1:
      L_off = _trsm(S[b:, :], L_II)                           # (N - (I+1)*b, b)
      parts.append(L_off)

    # --- 4. Scatter the new column block into L ---------------------------
    top = Tensor.zeros(I_start, b, dtype=dtype, device=device)
    col_I = Tensor.cat(top, *parts, dim=0) if I > 0 else Tensor.cat(*parts, dim=0)  # (N, b)

    lr = []
    if I > 0:            lr.append(Tensor.zeros(N, I_start, dtype=dtype, device=device))
    lr.append(col_I)
    if I < NB - 1:       lr.append(Tensor.zeros(N, (NB - I - 1) * b, dtype=dtype, device=device))
    col_full = Tensor.cat(*lr, dim=1) if len(lr) > 1 else lr[0]  # (N, N)

    L = L + col_full

  return L


cholesky = blocked_cholesky


if __name__ == "__main__":
  import numpy as np
  np.random.seed(0)
  for N in [64, 128, 256, 512, 1024]:
    X = np.random.randn(N, N).astype(np.float32)
    A_np = X @ X.T + N * np.eye(N, dtype=np.float32)
    A = Tensor(A_np)
    L = cholesky(A).numpy()
    ref = np.linalg.cholesky(A_np)
    err = float(np.abs(L - ref).max())
    print(f"[{os.environ.get('DEV', A.device)}]  B={B}  N={N:4d}  max_err={err:.3e}")
