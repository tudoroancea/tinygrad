"""Cholesky v5: right-looking, 2D thread partition across (i, k).

v4 used T threads to parallelise the `i` rows of the trailing column J.
Each thread then ran the full k-loop serially, which for large N is the
slow path. v5 splits the same T-thread budget across two axes:

  T_I  = rows in parallel  (mapped to tid_i = tid // T_K)
  T_K  = k partial sums    (mapped to tid_k = tid %  T_K)

For every J we do:
  1. Every (tid_i, tid_k) computes L[i_my, k_my] * L[J, k_my] for its slice
     of k and accumulates into a REG partial.
  2. Shared memory fan-in across tid_k reduces the partial to a per-row s_i.
  3. Pass 1: every (tid_i, tid_k) thread writes a pass-1 value to L[i_my, J].
     For (i_my==J) that's sqrt(A[J,J] - s_J); for i_my<J it's 0 (upper tri);
     for i_my>J a placeholder 0 gets overwritten in pass 2.
     The T_K threads of a given row all write the same value (benign race).
  4. Barrier.
  5. Pass 2: threads with i_my>J store (A[i_my, J] - s_i) / L[J, J].

Arithmetic intensity improves because L[J, :] is loaded once per (i_outer,
tid_i) warp and reused by T_K threads sharing the same stream.
"""
from __future__ import annotations
import os
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

T_I = int(os.environ.get("CHOL_TI", 64))
T_K = int(os.environ.get("CHOL_TK", 16))


def cholesky_kernel(L: UOp, A: UOp) -> UOp:
  assert len(L.shape) == 2 and L.shape == A.shape
  N = L.shape[0]
  assert N == L.shape[1], "square"

  dtype = L.dtype.base

  if N == 1:
    return L[0, 0].store(A[0, 0].sqrt()).sink(arg=KernelInfo(name="cholesky_v5_1", opts_to_apply=()))

  # If the matrix is smaller than the configured tile, shrink the tile to fit.
  # v5 is meant for N >> T_I*T_K; this fall-back keeps correctness for tests.
  ti = min(T_I, N)
  tk = min(T_K, N)
  while N % ti != 0: ti //= 2
  while N % tk != 0: tk //= 2
  t_total = ti * tk
  assert N % ti == 0 and N % tk == 0

  zero = UOp.const(dtype, 0.0)

  tid   = UOp.special(t_total, "lidx0")
  tid_i = tid // tk
  tid_k = tid %  tk

  # Sequential diagonal column.
  J = UOp.range(N, 0, AxisType.REDUCE)

  # i partition: each tid_i owns rows i = i_outer*ti + tid_i.
  i_chunks = N // ti
  i_outer  = UOp.range(i_chunks, 1, AxisType.REDUCE)
  i_my     = i_outer * ti + tid_i

  # k partition: each tid_k owns k = k_outer*tk + tid_k.
  k_chunks = N // tk
  k_outer  = UOp.range(k_chunks, 2, AxisType.REDUCE)
  k_my     = k_outer * tk + tid_k

  # Per-thread partial sum.
  part = UOp.placeholder((1,), dtype, 0, addrspace=AddrSpace.REG)
  part = part.after(J, i_outer)[0].set(zero)

  Lv = L.after(J, i_outer)
  contrib = (k_my < J).where(Lv[i_my, k_my] * Lv[J, k_my], zero)
  part = part[0].set(part.after(k_outer)[0] + contrib, end=k_outer)

  # Shared memory fan-in across tid_k for each tid_i row.
  # Layout: [tid_i, tid_k] -> flat index tid_i*tk + tid_k == tid.
  sh = UOp.placeholder((t_total,), dtype, 1, addrspace=AddrSpace.LOCAL)
  sh_w = sh[tid].store(part[0])
  sh_r = sh.after(UOp.barrier(sh_w))

  # Every thread sums its row's tk partials (redundant across tid_k, cheap).
  red_k  = UOp.range(tk, 3, AxisType.REDUCE)
  s_full = UOp.placeholder((1,), dtype, 2, addrspace=AddrSpace.REG)
  s_full = s_full.after(J, i_outer)[0].set(zero)
  s_full = s_full[0].set(s_full.after(red_k)[0] + sh_r[tid_i * tk + red_k], end=red_k)
  s_i    = s_full[0]

  # -------- Pass 1: diagonal + upper triangle zeros, pass-2 placeholder -----
  # diag_val is only correct for the thread where i_my == J, but every thread
  # computes it and then we mask via WHERE.
  diag_val = (A[J, J] - s_i).sqrt()
  pass1    = (i_my < J).where(zero, (i_my.eq(J)).where(diag_val, zero))
  p1_store = L[i_my, J].store(pass1)
  Lp       = L.after(UOp.barrier(p1_store))

  # -------- Pass 2: off-diagonal rows -------------------------------------
  diag_ready = Lp[J, J]
  off_val    = (A[i_my, J] - s_i) / diag_ready
  # Keep pass1 value for i_my<=J; install off_val only for i_my>J.
  pass1_static = (i_my < J).where(zero, (i_my.eq(J)).where(diag_ready, zero))
  pass2        = (i_my > J).where(off_val, pass1_static)
  p2_store     = L[i_my, J].store(pass2)

  # Barrier + close ranges + sink.
  p2_barrier = UOp.barrier(p2_store)
  ended      = p2_barrier.end(i_outer).end(J)
  return ended.sink(arg=KernelInfo(name=f"cholesky_v5_{N}", opts_to_apply=()))


def cholesky(A: Tensor) -> Tensor:
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "square matrix"
  L = Tensor.empty(*A.shape, dtype=A.dtype, device=A.device)
  return Tensor.custom_kernel(L, A, fxn=cholesky_kernel)[0]


if __name__ == "__main__":
  import numpy as np
  np.random.seed(0)
  for N in [1, 4, 16, 32, 64, 128, 256]:
    X = np.random.randn(N, N).astype(np.float32)
    A_np = X @ X.T + N * np.eye(N, dtype=np.float32)
    A = Tensor(A_np)
    L = cholesky(A).numpy()
    ref = np.linalg.cholesky(A_np)
    err = float(np.abs(L - ref).max())
    print(f"[{os.environ.get('DEV', A.device)}] T_I={T_I} T_K={T_K}  N={N:3d}  max_err={err:.3e}")
