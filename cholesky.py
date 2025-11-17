from typing import Callable

import numpy as np

from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.dtype import AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, KernelInfo

Tensor.manual_seed(127)

def tril1_kernel(L: UOp, A: UOp):
  assert L.ndim == A.ndim == 2 and L.shape[-2] == L.shape[-1] == A.shape[-2] == A.shape[-1]
  n = int(L.shape[0])
  i,j = UOp.range(n, 0), UOp.range(n, 1)
  return (
    L[i, j]
    .store(UOp(Ops.WHERE, dtype=L.dtype.base, src=(j <= i, A[i, j], UOp.const(L.dtype.base, 0.0))))
    .end(i, j)
    .sink(arg=KernelInfo(name="tril1_kernel"))
  )

def tril2_kernel(L: UOp, A:UOp):
  assert L.ndim == A.ndim == 2 and L.shape[-2] == L.shape[-1] == A.shape[-2] == A.shape[-1]
  n = int(L.shape[0])
  i,j = UOp.range(n, 0), UOp.range(n, 1)
  return L.index(i,j.valid(j<=i)).store(A[i,j]).end(i, j).sink(arg=KernelInfo(name="tril2_kernel"))

# NOTE: this realizes at each assignment
def chol(A: Tensor):
  assert len(A.shape) >= 2 and A.shape[-2] == A.shape[-1]
  n = int(A.shape[0])
  L = Tensor.zeros_like(A).contiguous()
  for i in range(n):
    for j in range(i + 1):
      sum = L[i, :j].dot(L[j, :j])
      if i == j:
        L[i, j] = (A[i, j] - sum).sqrt()
      else:
        L[i, j] = (A[i, j] - sum) / L[j, j]
  return L


# NOTE: this realizes at each assignment
def chol2(A: Tensor):
  assert len(A.shape) >= 2 and A.shape[-2] == A.shape[-1]
  n = int(A.shape[0])
  L = Tensor.zeros_like(A).contiguous()
  for i in range(n):
    L[i, i] = (A[i, i] - L[i, :i].square().sum()).sqrt()
    L[i + 1 :, i] = (A[i + 1 :, i] - L[i + 1 :, :i] @ L[i, :i]) / L[i, i]
  return L


def chol1_kernel(L: UOp, A: UOp) -> UOp:
  assert L.ndim == A.ndim == 2 and L.shape[-2] == L.shape[-1] == A.shape[-2] == A.shape[-1]
  assert L.dtype == A.dtype
  dtype = L.dtype.base
  n = int(L.shape[0])
  i, j = UOp.range(n, 0), UOp.range(n, 1)
  zero = UOp.const(dtype, 0.0)

  # create accumulation var
  s: UOp = UOp.placeholder((1,), dtype, 0, AddrSpace.REG)
  # initialize at 0
  s: UOp = s.after(i, j)[0].set(0)
  # reduce over range k
  k = UOp.range(n, 2, AxisType.REDUCE)
  m = UOp(Ops.WHERE, dtype, src=(k < j, L[i, k] * L[j, k], zero))
  s: UOp = s[0].set(s.after(k)[0] + m, end=k)

  c0 = UOp(Ops.WHERE, dtype=dtype, src=(i.eq(j), (A[i, j] - s[0]).sqrt(), (A[i, j] - s[0]) / L[j, j]))
  c1 = UOp(Ops.WHERE, dtype=dtype, src=(j <= i, c0, zero))
  ast = L[i, j].store(c1).end(i, j)
  return ast.sink(arg=KernelInfo(name="chol1_kernel"))

def chol2_kernel(L: UOp, A: UOp) -> UOp:
  assert L.ndim == A.ndim == 2 and L.shape[-2] == L.shape[-1] == A.shape[-2] == A.shape[-1]
  assert L.dtype == A.dtype
  dtype = L.dtype.base
  n = int(L.shape[0])
  i, j = UOp.range(n, 0), UOp.range(n, 1)

  # create accumulation var
  s: UOp = UOp.placeholder((1,), dtype, 0, AddrSpace.REG)
  # initialize at 0
  s: UOp = s.after(i, j)[0].set(0)
  # reduce over range k
  k = UOp.range(n, 2, AxisType.REDUCE)
  s: UOp = s[0].set(s.after(k)[0] + L[i, k.valid(k<j)] * L[j, k.valid(k<j)], end=k)

  c0 = UOp(Ops.WHERE, dtype=dtype, src=(i.eq(j), (A[i, j] - s[0]).sqrt(), (A[i, j] - s[0]) / L[j, j]))
  ast = L[i, j.valid(j<=i)].store(c0).end(i, j)
  return ast.sink(arg=KernelInfo(name="chol2_kernel"))

def test_kernel(prefix:str, A: Tensor, kernel: Callable[[UOp, UOp], UOp], numpy_fn: Callable[[np.ndarray], np.ndarray]|None):
  res = Tensor.empty_like(A).custom_kernel(A, fxn=kernel)[0].numpy()
  if A.shape[0] <= 4:
    print(f"{prefix}(A)=\n", res)
  if numpy_fn: np.testing.assert_array_almost_equal(res, numpy_fn(A.numpy()))

def main():
  n = 4
  A = Tensor.randn(n, n)
  A = Tensor.realize(Tensor.eye(n) + A.T @ A)
  print("")

  if n<=4: print("A=\n", A.numpy())
  test_kernel("tril1", A, tril1_kernel, np.tril)
  test_kernel("tril2", A, tril2_kernel, np.tril)

  print("")
  if n <= 4: print("np.chol(A)=\n", np.linalg.cholesky(A.numpy()))
  test_kernel("chol1", A, chol1_kernel, np.linalg.cholesky)
  test_kernel("chol2", A, chol2_kernel, None)


if __name__ == "__main__":
  main()
