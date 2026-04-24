"""
Codegenerable Cholesky factorization in tinygrad.

This implements Cholesky factorization A = L L^T (for symmetric positive
definite A) as a single custom kernel written directly in UOps.

Design notes:
- No python-dispatched loops: the algorithm's sequential structure is
  expressed with three nested UOp.range axes (i, j, k) that become a
  single kernel. The kernel is emitted once and run once.
- No .realize() inside the computation: all control flow is intra-kernel.
- Works on CPU and GPU backends that support the C-style renderer
  because we only rely on RANGE (for loops), LOAD/STORE, SQRT, FDIV,
  and standard ALU ops.

Algorithm (Cholesky-Banachiewicz):
    for i in 0..N:
      for j in 0..N:
        if j > i:        L[i,j] = 0
        elif j < i:      L[i,j] = (A[i,j] - sum_{k<j} L[i,k]*L[j,k]) / L[j,j]
        else:            L[i,i] = sqrt(A[i,i] - sum_{k<i} L[i,k]^2)

Uniformly: let s = sum_{k<j} L[i,k]*L[j,k] (for j<=i).
  When j==i, L[j,k]=L[i,k], so s = sum_{k<i} L[i,k]^2 as required.
  When j<i, s is the off-diagonal dot product.
  When j>i, s is unused (we store 0).

Sequential dependencies:
  i must be sequential (row i depends on rows 0..i-1).
  j must be sequential (column j in row i depends on L[i, 0..j-1]).
  k is a sequential reduction within each (i, j) cell.

All three axes are created with AxisType.REDUCE. That prevents
`convert_loop_to_global` from promoting the outer axes to parallel
block dims, which would break the loop-carried dependency on L.
(AxisType.REDUCE is still rendered as a plain for-loop by the C-style
renderer; the name refers to how tinygrad's scheduler classifies the
axis, not to any forced reduction op.)

The reads of L[i,k], L[j,k] and L[j,j] are tagged with `after(i, j)` so
tinygrad keeps the LOADs inside the i,j for-loop nest, and they
therefore see the most-recently-written values in memory.
"""

from tinygrad import Tensor
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.dtype import AddrSpace


def cholesky_kernel(L: UOp, A: UOp) -> UOp:
  """Build a single-kernel Cholesky AST.

  Inputs:
    L: the output lower-triangular factor buffer (NxN).
    A: the symmetric-positive-definite matrix to factor (NxN).

  Returns an Ops.SINK for use with Tensor.custom_kernel.
  """
  assert len(L.shape) == 2 and len(A.shape) == 2, "require 2D buffers"
  assert L.shape == A.shape, f"L and A shapes differ: {L.shape} vs {A.shape}"
  N = L.shape[0]
  assert N == L.shape[1], "Cholesky requires a square matrix"

  dtype = L.dtype.base
  zero = UOp.const(dtype, 0.0)
  one = UOp.const(dtype, 1.0)

  # Sequential outer (row) and middle (column) loops. We tag them REDUCE (not
  # LOOP) because tinygrad's `convert_loop_to_global` would otherwise promote
  # LOOP axes that appear at sink-level ENDs to parallel GLOBAL block dims —
  # which is incorrect for Cholesky because row i depends on rows 0..i-1 and
  # within row i column j depends on columns 0..j-1. REDUCE axes are always
  # rendered as sequential for-loops by tinygrad.
  i = UOp.range(N, 0, AxisType.REDUCE)
  j = UOp.range(N, 1, AxisType.REDUCE)
  # Inner k is also a sequential reduction (dot product of first j entries).
  k = UOp.range(N, 2, AxisType.REDUCE)

  # Per-thread register accumulator for s.
  s_reg = UOp.placeholder((1,), dtype, 0, addrspace=AddrSpace.REG)
  s_reg = s_reg.after(i, j)[0].set(zero)

  # Memory-ordered view of L: reads happen inside the i,j for-loops,
  # so loop-carried writes from prior iterations are visible.
  Lv = L.after(i, j)

  # s += (k < j) ? L[i,k] * L[j,k] : 0
  mask_k = k < j
  contrib = mask_k.where(Lv[i, k] * Lv[j, k], zero)
  s_reg = s_reg[0].set(s_reg.after(k)[0] + contrib, end=k)

  # After closing k, compute L[i, j].
  s = s_reg[0]
  a_ij = A[i, j]
  diag_jj = Lv[j, j]

  # Guard denominator and sqrt input so unused branches don't produce NaN.
  j_lt_i = j < i
  j_eq_i = j.eq(i)
  safe_denom = j_lt_i.where(diag_jj, one)
  off_diag_val = (a_ij - s) / safe_denom

  safe_sqrt_input = j_eq_i.where(a_ij - s, one)
  diag_val = safe_sqrt_input.sqrt()

  val = j_lt_i.where(off_diag_val, j_eq_i.where(diag_val, zero))

  store = L[i, j].store(val)
  return store.end(j).end(i).sink(arg=KernelInfo(name=f"cholesky_{N}", opts_to_apply=()))


def cholesky(A: Tensor) -> Tensor:
  """Compute the lower-triangular Cholesky factor of A with a single custom kernel."""
  assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "square matrix required"
  L = Tensor.empty(*A.shape, dtype=A.dtype, device=A.device)
  return Tensor.custom_kernel(L, A, fxn=cholesky_kernel)[0]


if __name__ == "__main__":
  import os, numpy as np

  np.random.seed(0)
  for N in [64]:
    # for N in [1, 2, 4, 8, 16, 32, 64]:
    X = np.random.randn(N, N).astype(np.float32)
    A_np = X @ X.T + N * np.eye(N, dtype=np.float32)  # SPD
    A = Tensor(A_np)
    L = cholesky(A)
    L_np = L.numpy()
    ref = np.linalg.cholesky(A_np)
    err = float(np.abs(L_np - ref).max())
    rel = err / float(np.abs(ref).max())
    dev = os.environ.get("DEV", A.device)
    print(f"[{dev}] N={N:3d}  max|L - ref|={err:.3e}  rel={rel:.3e}")
