"""Regression tests for extra/cholesky/* across variants and sizes.

Runs from the extra/cholesky directory:
  DEV=NV  .venv/bin/python -m unittest extra.cholesky.test_cholesky -v
"""
import importlib, sys, os, unittest
import numpy as np
from tinygrad import Tensor

sys.path.insert(0, os.path.dirname(__file__))


def _spd(N: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  X = rng.standard_normal((N, N)).astype(np.float32)
  return X @ X.T + N * np.eye(N, dtype=np.float32)


def _load(variant: str):
  return importlib.import_module(f"cholesky_{variant}").cholesky


def _check(chol, N: int, rel_tol: float = 1e-4):
  A = _spd(N)
  L = chol(Tensor(A)).numpy()
  ref = np.linalg.cholesky(A)
  err = float(np.abs(L - ref).max())
  ref_abs = float(np.abs(ref).max())
  assert np.isfinite(err), f"non-finite error for N={N}"
  assert err <= rel_tol * ref_abs, f"N={N}: err={err:.3e} exceeds {rel_tol} * max|ref|"
  # structural check: strictly upper triangle should be zero
  assert float(np.abs(np.triu(L, k=1)).max()) < 1e-5
  # reconstruction
  rec = L @ L.T
  assert float(np.abs(rec - A).max()) < rel_tol * float(np.abs(A).max())


class TestCholeskyV1(unittest.TestCase):
  chol = staticmethod(_load("v1"))
  def test_small(self):
    for N in (1, 2, 8, 16): _check(self.chol, N)
  def test_medium(self):
    for N in (32, 64): _check(self.chol, N)


class TestCholeskyV2(unittest.TestCase):
  chol = staticmethod(_load("v2"))
  def test_small(self):
    for N in (1, 2, 8, 16): _check(self.chol, N)
  def test_medium(self):
    for N in (32, 64): _check(self.chol, N)


class TestCholeskyV3(unittest.TestCase):
  chol = staticmethod(_load("v3"))
  def test_sizes(self):
    # v3 uses __shared__ memory + barriers; skip on backends without locals.
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v3 needs LOCAL memory support")
    for N in (1, 2, 8, 16, 32, 64, 128): _check(self.chol, N)


class TestCholeskyV4(unittest.TestCase):
  chol = staticmethod(_load("v4"))
  def test_sizes(self):
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v4 needs LOCAL memory support")
    # v4's fixed T=128 means N must be a multiple (or N<T); pick safe sizes.
    for N in (1, 128, 256): _check(self.chol, N)


class TestCholeskyV5(unittest.TestCase):
  chol = staticmethod(_load("v5"))
  def test_sizes(self):
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v5 needs LOCAL memory support")
    for N in (1, 16, 32, 64, 128, 256): _check(self.chol, N)


class TestCholeskyV7(unittest.TestCase):
  chol = staticmethod(_load("v7"))
  def test_sizes(self):
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v7 needs LOCAL memory support")
    for N in (128, 256, 512): _check(self.chol, N)


class TestCholeskyV9(unittest.TestCase):
  chol = staticmethod(_load("v9"))
  def test_sizes(self):
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v9 needs LOCAL memory support")
    for N in (256, 512, 1024): _check(self.chol, N)


class TestCholeskyV10(unittest.TestCase):
  chol = staticmethod(_load("v10"))
  def test_sizes(self):
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v10 needs LOCAL memory support")
    for N in (256, 512, 1024): _check(self.chol, N)


class TestCholeskyV11(unittest.TestCase):
  chol = staticmethod(_load("v11"))
  def test_sizes(self):
    from tinygrad.device import Device
    if not getattr(Device[Tensor.empty(1).device].renderer, "has_local", False):
      self.skipTest("v11 needs LOCAL memory support")
    for N in (256, 512, 1024): _check(self.chol, N)


class TestDispatcher(unittest.TestCase):
  def test_picks_working_kernel(self):
    from extra.cholesky import cholesky as dispatch
    for N in (1, 4, 16, 64, 256): _check(dispatch, N)


if __name__ == "__main__":
  unittest.main()
