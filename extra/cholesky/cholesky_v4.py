"""Cholesky v4: right-looking, thread-per-row.

Flip the outer-loop structure from v1/v3 (sequential i, sequential j inside i)
to right-looking Cholesky: sequential diagonal column J, and a block of T
threads that cooperatively compute all rows of column J in parallel.

For each J = 0..N-1:
  1. Each thread owns a row `i`, possibly cycling through several rows if N>T.
  2. Everyone computes its own partial dot product  s_i = Σ_{k<J} L[i,k]·L[J,k].
  3. The thread with `i == J` stores L[J, J] = sqrt(A[J, J] - s_J).
     All other threads store 0 (if i<J → upper triangle) or a placeholder
     (if i>J → overwritten in pass 2).
  4. Barrier.
  5. Threads with i > J store L[i, J] = (A[i, J] - s_i) / L[J, J].

Net: the inner work per J is now O(N / T) sequential k-iterations per thread
(plus a one-time store), instead of v3's (N / T)·N nested loops. Expected
asymptotic cost drops from Θ(N³ / T) to Θ(N² · N / T) = Θ(N³ / T) — same
big-O but a tighter constant because the hot loop is no longer the outer
j-loop, and L[i, k] is reused across all rows of column J by every thread.
In practice we also get more arithmetic-per-byte since each thread's k-stream
reuses a single L[J, :] column which sits in L1 / L2.
"""
from __future__ import annotations
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace

import os
T = int(os.environ.get("CHOL_T", 128))  # threads per block


def cholesky_kernel(L: UOp, A: UOp) -> UOp:
  assert len(L.shape) == 2 and L.shape == A.shape
  N = L.shape[0]
  assert N == L.shape[1], "square matrix"
  assert N % T == 0 or N < T, f"v4 requires N a multiple of T={T} (or N<T)"

  dtype = L.dtype.base

  if N == 1:
    return L[0, 0].store(A[0, 0].sqrt()).sink(arg=KernelInfo(name="cholesky_v4_1", opts_to_apply=()))

  zero = UOp.const(dtype, 0.0)
  one  = UOp.const(dtype, 1.0)

  tid = UOp.special(T, "lidx0")

  # Sequential diagonal column.
  J = UOp.range(N, 0, AxisType.REDUCE)

  # Row chunking: each thread cycles through rows i = i_outer*T + tid.
  i_chunks = max(1, N // T)
  i_outer  = UOp.range(i_chunks, 1, AxisType.REDUCE)
  i        = i_outer * T + tid

  # Per-thread partial sum for its assigned row.
  s_reg = UOp.placeholder((1,), dtype, 0, addrspace=AddrSpace.REG)
  s_reg = s_reg.after(J, i_outer)[0].set(zero)

  Lv = L.after(J, i_outer)
  k  = UOp.range(N, 2, AxisType.REDUCE)  # inner reduction, static bound + mask k<J
  contrib = (k < J).where(Lv[i, k] * Lv[J, k], zero)
  s_reg = s_reg[0].set(s_reg.after(k)[0] + contrib, end=k)
  s_i = s_reg[0]

  # ----- Pass 1: write diagonal + upper triangle zeros -------------------
  # Only one thread (the one whose i == J) actually uses its s as `s_J` for the sqrt.
  # Other threads compute their own s_i (wasted for pass 1) but that's fine.
  diag_val = (A[J, J] - s_i).sqrt()  # only correct for the thread where i==J
  pass1_val = (i < J).where(zero, (i.eq(J)).where(diag_val, zero))  # 0 placeholder for i>J
  p1_store  = L[i, J].store(pass1_val)

  # Barrier so threads with i>J see the fresh L[J, J] from the thread where i==J.
  p1_barrier = UOp.barrier(p1_store)
  Lp = L.after(p1_barrier)

  # ----- Pass 2: off-diagonal write for i > J ---------------------------
  # For i == J and i < J, rewrite the value we wrote in pass 1 (a no-op value-wise).
  diag_after = Lp[J, J]
  off_val = (A[i, J] - s_i) / diag_after
  # What we wrote in pass 1 per i: 0 if i<J, diag_val if i==J, 0 if i>J.
  pass1_static = (i < J).where(zero, (i.eq(J)).where(diag_after, zero))
  pass2_val = (i > J).where(off_val, pass1_static)
  p2_store  = L[i, J].store(pass2_val)

  # End the loops. A final barrier prevents the next J iteration from racing ahead
  # of a straggling warp (only matters if we ever bump T above 32).
  p2_barrier = UOp.barrier(p2_store)
  ended = p2_barrier.end(i_outer).end(J)
  return ended.sink(arg=KernelInfo(name=f"cholesky_v4_{N}", opts_to_apply=()))


def cholesky(A: Tensor) -> Tensor:
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "square matrix"
  L = Tensor.empty(*A.shape, dtype=A.dtype, device=A.device)
  return Tensor.custom_kernel(L, A, fxn=cholesky_kernel)[0]


if __name__ == "__main__":
  import os, numpy as np
  np.random.seed(0)
  for N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    X = np.random.randn(N, N).astype(np.float32)
    A_np = X @ X.T + N * np.eye(N, dtype=np.float32)
    A = Tensor(A_np)
    L = cholesky(A).numpy()
    ref = np.linalg.cholesky(A_np)
    err = float(np.abs(L - ref).max())
    print(f"[{os.environ.get('DEV', A.device)}] N={N:3d}  max_err={err:.3e}")
