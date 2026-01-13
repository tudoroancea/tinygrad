"""Tests for the new vmap implementation using graph rewriting."""
import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.transforms.vmap import vmap

class TestVmapBasic(unittest.TestCase):
  """Basic vmap tests for elementwise operations."""

  def test_simple_unary(self):
    """Test vmap over a simple unary function."""
    def fn(x):
      return x * 2

    vfn = vmap(fn)
    x = Tensor.ones(5, 3)  # 5 batches of 3-vectors
    result = vfn(x)
    expected = np.ones((5, 3)) * 2
    np.testing.assert_allclose(result.numpy(), expected)

  def test_simple_binary(self):
    """Test vmap over a simple binary function."""
    def fn(x, y):
      return x + y

    vfn = vmap(fn)
    x = Tensor.ones(5, 3)
    y = Tensor.ones(5, 3) * 2
    result = vfn(x, y)
    expected = np.ones((5, 3)) * 3
    np.testing.assert_allclose(result.numpy(), expected)

  def test_broadcast_second_arg(self):
    """Test vmap with second argument broadcasted (in_axes=(0, None))."""
    def fn(x, y):
      return x + y

    vfn = vmap(fn, in_axes=(0, None))
    x = Tensor.ones(5, 3)  # batched
    y = Tensor.ones(3) * 2  # not batched - broadcast
    result = vfn(x, y)
    expected = np.ones((5, 3)) * 3
    np.testing.assert_allclose(result.numpy(), expected)

  def test_broadcast_first_arg(self):
    """Test vmap with first argument broadcasted (in_axes=(None, 0))."""
    def fn(x, y):
      return x + y

    vfn = vmap(fn, in_axes=(None, 0))
    x = Tensor.ones(3) * 2  # not batched - broadcast
    y = Tensor.ones(5, 3)  # batched
    result = vfn(x, y)
    expected = np.ones((5, 3)) * 3
    np.testing.assert_allclose(result.numpy(), expected)


class TestVmapMovement(unittest.TestCase):
  """Tests for vmap with movement operations."""

  def test_reshape(self):
    """Test vmap over reshape."""
    def fn(x):
      return x.reshape(2, 3)

    vfn = vmap(fn)
    x = Tensor.arange(30).reshape(5, 6)  # 5 batches of 6-vectors
    result = vfn(x)
    self.assertEqual(result.shape, (5, 2, 3))

  def test_transpose(self):
    """Test vmap over transpose."""
    def fn(x):
      return x.T

    vfn = vmap(fn)
    x = Tensor.arange(30).reshape(5, 2, 3)  # 5 batches of 2x3 matrices
    result = vfn(x)
    self.assertEqual(result.shape, (5, 3, 2))

  def test_permute(self):
    """Test vmap over permute."""
    def fn(x):
      return x.permute(1, 0, 2)

    vfn = vmap(fn)
    x = Tensor.arange(120).reshape(5, 2, 3, 4)  # 5 batches of 2x3x4 tensors
    result = vfn(x)
    self.assertEqual(result.shape, (5, 3, 2, 4))


class TestVmapReduce(unittest.TestCase):
  """Tests for vmap with reduction operations."""

  def test_sum_all(self):
    """Test vmap over sum (reduce all)."""
    def fn(x):
      return x.sum()

    vfn = vmap(fn)
    x = Tensor.ones(5, 3)
    result = vfn(x)
    self.assertEqual(result.shape, (5,))
    np.testing.assert_allclose(result.numpy(), np.ones(5) * 3)

  def test_sum_axis(self):
    """Test vmap over sum with axis."""
    def fn(x):
      return x.sum(axis=0)

    vfn = vmap(fn)
    x = Tensor.ones(5, 2, 3)  # 5 batches of 2x3 matrices
    result = vfn(x)
    self.assertEqual(result.shape, (5, 3))

  def test_max_axis(self):
    """Test vmap over max with axis."""
    def fn(x):
      return x.max(axis=1)

    vfn = vmap(fn)
    x = Tensor.arange(30).reshape(5, 2, 3).float()
    result = vfn(x)
    self.assertEqual(result.shape, (5, 2))


class TestVmapInAxes(unittest.TestCase):
  """Tests for different in_axes configurations."""

  def test_in_axes_int(self):
    """Test in_axes as single int."""
    def fn(x, y):
      return x * y

    vfn = vmap(fn, in_axes=0)
    x = Tensor.ones(5, 3)
    y = Tensor.ones(5, 3) * 2
    result = vfn(x, y)
    np.testing.assert_allclose(result.numpy(), np.ones((5, 3)) * 2)

  def test_in_axes_tuple(self):
    """Test in_axes as tuple."""
    def fn(x, y):
      return x * y

    vfn = vmap(fn, in_axes=(0, 0))
    x = Tensor.ones(5, 3)
    y = Tensor.ones(5, 3) * 2
    result = vfn(x, y)
    np.testing.assert_allclose(result.numpy(), np.ones((5, 3)) * 2)

  def test_in_axes_none_requires_axis_size(self):
    """Test that in_axes=None for all raises error."""
    def fn(x):
      return x * 2

    vfn = vmap(fn, in_axes=None)
    x = Tensor.ones(3)
    with self.assertRaises(ValueError):
      vfn(x)


class TestVmapValidation(unittest.TestCase):
  """Tests for input validation."""

  def test_non_tensor_raises(self):
    """Test that non-Tensor arguments raise TypeError."""
    def fn(x):
      return x * 2

    vfn = vmap(fn)
    with self.assertRaises(TypeError):
      vfn([1, 2, 3])

  def test_axis_out_of_bounds(self):
    """Test that out-of-bounds axis raises ValueError."""
    def fn(x):
      return x * 2

    vfn = vmap(fn, in_axes=5)
    x = Tensor.ones(3, 4)
    with self.assertRaises(ValueError):
      vfn(x)

  def test_in_axes_length_mismatch(self):
    """Test that wrong in_axes tuple length raises ValueError."""
    def fn(x, y):
      return x + y

    vfn = vmap(fn, in_axes=(0,))  # Only 1 axis for 2 args
    x = Tensor.ones(3, 4)
    y = Tensor.ones(3, 4)
    with self.assertRaises(ValueError):
      vfn(x, y)


class TestVmapComparison(unittest.TestCase):
  """Tests comparing vmap output against expected values (JAX-like behavior)."""

  def test_matmul_broadcast(self):
    """Test vmap over matrix-vector multiply with broadcast matrix."""
    def fn(A, x):
      return A @ x

    # A is shared (broadcast), x is batched
    vfn = vmap(fn, in_axes=(None, 0))
    A = Tensor([[1, 2], [3, 4]]).float()  # 2x2 matrix
    xs = Tensor.ones(5, 2)  # 5 batches of 2-vectors
    result = vfn(A, xs)
    self.assertEqual(result.shape, (5, 2))
    # Each result should be A @ [1, 1] = [3, 7]
    expected = np.array([[3., 7.]] * 5)
    np.testing.assert_allclose(result.numpy(), expected)

  def test_exp_sin_chain(self):
    """Test chain of transcendental functions."""
    def fn(x):
      return x.exp().sin()

    vfn = vmap(fn)
    x = Tensor([0.0, 0.5, 1.0]).reshape(3, 1)  # 3 batches
    result = vfn(x)
    expected = np.sin(np.exp(np.array([[0.0], [0.5], [1.0]])))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_negative_axis(self):
    """Test negative axis indices."""
    def fn(x, y):
      return x + y

    vfn = vmap(fn, in_axes=(-1, -1))  # Batch over last axis
    x = Tensor.ones(3, 5)  # Last axis has size 5
    y = Tensor.ones(3, 5) * 2
    result = vfn(x, y)
    self.assertEqual(result.shape, (5, 3))  # Batch moved to front


class TestVmapNested(unittest.TestCase):
  """Tests for nested vmap (vmap of vmap)."""

  def test_double_vmap(self):
    """Test vmap(vmap(fn)) for 2D batching."""
    def fn(x, y):
      return x + y

    # Inner vmap over first axis, outer vmap over first axis of result
    vfn = vmap(vmap(fn))
    x = Tensor.ones(3, 4, 2)  # 3x4 batches of 2-vectors
    y = Tensor.ones(3, 4, 2) * 2
    result = vfn(x, y)
    self.assertEqual(result.shape, (3, 4, 2))
    np.testing.assert_allclose(result.numpy(), np.ones((3, 4, 2)) * 3)


class TestVmapIndexing(unittest.TestCase):
  """Tests for vmap with indexing operations."""

  def test_simple_index(self):
    """Test vmap over simple integer indexing."""
    def fn(x):
      return x[0]  # Get first element

    vfn = vmap(fn)
    x = Tensor.arange(15).reshape(5, 3).float()
    result = vfn(x)
    self.assertEqual(result.shape, (5,))
    expected = np.array([0., 3., 6., 9., 12.])
    np.testing.assert_allclose(result.numpy(), expected)

  def test_slice_index(self):
    """Test vmap over slice indexing."""
    def fn(x):
      return x[1:3]

    vfn = vmap(fn)
    x = Tensor.arange(20).reshape(5, 4).float()
    result = vfn(x)
    self.assertEqual(result.shape, (5, 2))

  def test_tensor_index(self):
    """Test vmap over tensor-based indexing."""
    def fn(x, idx):
      return x[idx]

    vfn = vmap(fn)
    x = Tensor.arange(15).reshape(5, 3).float()
    idx = Tensor([0, 2, 1, 0, 2])  # Different index for each batch
    result = vfn(x, idx)
    self.assertEqual(result.shape, (5,))
    # Expected: x[0,0], x[1,2], x[2,1], x[3,0], x[4,2] = 0, 5, 7, 9, 14
    expected = np.array([0., 5., 7., 9., 14.])
    np.testing.assert_allclose(result.numpy(), expected)

  def test_multidim_index(self):
    """Test vmap over multi-dimensional indexing."""
    def fn(x):
      return x[0, 1]

    vfn = vmap(fn)
    x = Tensor.arange(30).reshape(5, 2, 3).float()
    result = vfn(x)
    self.assertEqual(result.shape, (5,))
    expected = np.arange(30).reshape(5, 2, 3)[:, 0, 1]
    np.testing.assert_allclose(result.numpy(), expected)


if __name__ == "__main__":
  unittest.main()
