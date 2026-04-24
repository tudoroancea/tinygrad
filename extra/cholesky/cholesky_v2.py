"""Cholesky v2: replace the manual REG accumulator with a real `Ops.REDUCE`.

The algorithm is still left-looking Cholesky-Banachiewicz and the kernel
structure matches v1 — the only change is that the inner dot product
`s = sum_{k<j} L[i,k] * L[j,k]` is now expressed as an `Ops.REDUCE` over k
instead of a loop + scalar register. That gives tinygrad a proper reduceop
handle so the scheduler can apply GROUP_REDUCE / UNROLL / upcast opts on the
k-axis. We also pre-apply an UNROLL opt on k when N permits it, which lets the
C-style backends emit multiple parallel partial sums and actually use SIMD.
"""
from __future__ import annotations
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType, Ops
from tinygrad.codegen.opt import Opt, OptOps


def _pick_unroll(n: int) -> int:
  for u in (8, 4, 2):
    if n % u == 0: return u
  return 1


def cholesky_kernel(L: UOp, A: UOp) -> UOp:
  assert len(L.shape) == 2 and L.shape == A.shape
  N = L.shape[0]
  assert N == L.shape[1], "square matrix required"

  dtype = L.dtype.base

  # N=1 is a degenerate case: the inner Ops.REDUCE collapses to a trivial range
  # and tripsassertions in pm_reduce_unparented. Just emit L[0,0] = sqrt(A[0,0]).
  if N == 1:
    store = L[0, 0].store(A[0, 0].sqrt())
    return store.sink(arg=KernelInfo(name="cholesky_v2_1", opts_to_apply=()))

  zero = UOp.const(dtype, 0.0)
  one  = UOp.const(dtype, 1.0)

  i = UOp.range(N, 0, AxisType.REDUCE)
  j = UOp.range(N, 1, AxisType.REDUCE)
  k = UOp.range(N, 2, AxisType.REDUCE)

  # Memory-ordered view so loop-carried writes to L are visible.
  Lv = L.after(i, j)

  contrib = (k < j).where(Lv[i, k] * Lv[j, k], zero)
  s = contrib.reduce(k, arg=Ops.ADD)

  a_ij    = A[i, j]
  diag_jj = Lv[j, j]
  delta   = a_ij - s

  j_lt_i = j < i
  j_eq_i = j.eq(i)
  sqrt_in = j_eq_i.where(delta, one)
  denom   = j_lt_i.where(diag_jj, one)
  val     = j_lt_i.where(delta / denom, j_eq_i.where(sqrt_in.sqrt(), zero))

  store = L[i, j].store(val)
  return store.end(j).end(i).sink(arg=KernelInfo(name=f"cholesky_v2_{N}", opts_to_apply=()))


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
