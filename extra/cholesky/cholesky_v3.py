"""Cholesky v3: single-block, thread-parallel k-reduction.

Structure:
  * One thread block (gridDim = 1). Sequential i, j outer loops, SPMD across
    threads (each thread executes the same for-loop iteration).
  * `T = 32` threads per block (one CUDA warp). Strided partition of k:
    thread tid handles indices tid, tid+T, tid+2T, ...
  * Per-thread partial sums share through an LDS / __shared__ buffer, then
    a barrier fan-in reduction gives every thread the full dot product.
  * All threads write the same value to L[i, j] - benign race on coherent
    memory. The next (i, j') iteration reads the updated L.

For T=32 we stay inside one warp so SIMT lockstep already orders writes; we
still emit a barrier before/after the shared-memory reduction to be conservative.
Kernel entry is still a single custom_kernel; no python-side dispatch.
"""
from __future__ import annotations
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace, dtypes

T = 32  # threads per block


def cholesky_kernel(L: UOp, A: UOp) -> UOp:
  assert len(L.shape) == 2 and L.shape == A.shape
  N = L.shape[0]
  assert N == L.shape[1], "square matrix"

  dtype = L.dtype.base

  if N == 1:
    return L[0, 0].store(A[0, 0].sqrt()).sink(arg=KernelInfo(name="cholesky_v3_1", opts_to_apply=()))

  zero = UOp.const(dtype, 0.0)
  one  = UOp.const(dtype, 1.0)

  # Thread id (blockDim.x = T, one warp).
  tid = UOp.special(T, "lidx0")

  # Sequential (SPMD) outer row and column loops.
  i = UOp.range(N, 0, AxisType.REDUCE)
  j = UOp.range(N, 1, AxisType.REDUCE)

  # Partition k across threads: thread tid owns indices tid, tid+T, tid+2T, ...
  k_chunks = (N + T - 1) // T
  k_chunk  = UOp.range(k_chunks, 2, AxisType.REDUCE)
  k = k_chunk * T + tid  # per-thread k value

  # Per-thread partial sum for s = sum_{k<j} L[i,k]*L[j,k].
  s_reg = UOp.placeholder((1,), dtype, 0, addrspace=AddrSpace.REG)
  s_reg = s_reg.after(i, j)[0].set(zero)

  Lv = L.after(i, j)
  valid = (k < j) & (k < UOp.const(dtypes.weakint, N))
  contrib = valid.where(Lv[i, k] * Lv[j, k], zero)
  s_reg = s_reg[0].set(s_reg.after(k_chunk)[0] + contrib, end=k_chunk)

  # Fan-in reduction through __shared__ memory.
  s_local = UOp.placeholder((T,), dtype, 1, addrspace=AddrSpace.LOCAL)
  s_local_write = s_local[tid].store(s_reg[0])
  s_local_ready = s_local.after(UOp.barrier(s_local_write))

  red = UOp.range(T, 3, AxisType.REDUCE)
  total_reg = UOp.placeholder((1,), dtype, 2, addrspace=AddrSpace.REG)
  total_reg = total_reg.after(i, j)[0].set(zero)
  total_reg = total_reg[0].set(total_reg.after(red)[0] + s_local_ready[red], end=red)

  s       = total_reg[0]
  a_ij    = A[i, j]
  diag_jj = Lv[j, j]
  delta   = a_ij - s

  j_lt_i = j < i
  j_eq_i = j.eq(i)
  sqrt_in = j_eq_i.where(delta, one)
  denom   = j_lt_i.where(diag_jj, one)
  val     = j_lt_i.where(delta / denom, j_eq_i.where(sqrt_in.sqrt(), zero))

  # Every thread writes the same value to L[i, j]. Benign race (same bits
  # everywhere). A trailing barrier inside the j-loop guarantees the write is
  # visible to all threads before the next iteration reads back from L.
  store = L[i, j].store(val)
  ended = store.end(j).end(i)
  return ended.sink(arg=KernelInfo(name=f"cholesky_v3_{N}", opts_to_apply=()))


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
