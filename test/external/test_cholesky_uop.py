import unittest
import numpy as np
from tinygrad import Tensor

# import the custom-kernel Cholesky from extra/
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extra", "cholesky"))
from cholesky_uop import cholesky  # noqa: E402


def _spd(N: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  X = rng.standard_normal((N, N)).astype(np.float32)
  return X @ X.T + N * np.eye(N, dtype=np.float32)


class TestCholeskyUOp(unittest.TestCase):
  def _check(self, N: int, atol: float):
    A = _spd(N)
    L = cholesky(Tensor(A)).numpy()
    ref = np.linalg.cholesky(A)
    err = float(np.abs(L - ref).max())
    self.assertLess(err, atol, f"N={N} err={err}")
    # Also verify L @ L.T ~= A (tightly) and L is lower-triangular
    self.assertLess(float(np.abs(np.triu(L, k=1)).max()), 1e-6)
    self.assertLess(float(np.abs(L @ L.T - A).max()), 1e-3 * float(np.abs(A).max()))

  def test_1x1(self): self._check(1, 1e-6)
  def test_2x2(self): self._check(2, 1e-6)
  def test_8x8(self): self._check(8, 1e-6)
  def test_16x16(self): self._check(16, 1e-6)
  def test_64x64(self): self._check(64, 1e-5)


if __name__ == "__main__":
  unittest.main()
