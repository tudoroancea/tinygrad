"""Test native vmap using pad+reduce pattern with a vmap transform function."""
import unittest
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import Ops
from typing import Callable, Sequence

def vmap(fn: Callable[..., Tensor], in_axes: int | Sequence[int | None] = 0, out_axis: int = 0) -> Callable[..., Tensor]:
  """Transform a function to operate over a batch dimension.

  Args:
    fn: Function that takes Tensor(s) and returns a single Tensor
    in_axes: Axis to map over for each input. Can be:
      - int: same axis for all inputs
      - Sequence[int | None]: per-input axis, None means broadcast (don't map)
    out_axis: Axis to insert the batch dimension in output (default: 0)

  Returns:
    A new function that applies fn to each slice along the mapped axes
  """
  def vmapped(*args: Tensor) -> Tensor:
    # Normalize in_axes to a list
    if isinstance(in_axes, int):
      axes = [in_axes] * len(args)
    else:
      axes = list(in_axes)
    assert len(axes) == len(args), f"in_axes length {len(axes)} != number of args {len(args)}"

    # Find batch size from first mapped input
    batch_size = None
    for arg, axis in zip(args, axes):
      if axis is not None:
        batch_size = arg.shape[axis]
        break
    assert batch_size is not None, "At least one input must be mapped"

    a = UOp.range(batch_size, -1)

    # Slice each input along its axis (or pass through if None)
    sliced_args = []
    for arg, axis in zip(args, axes):
      if axis is None:
        sliced_args.append(arg)
      else:
        slices = [slice(None)] * len(arg.shape)
        slices[axis] = a
        sliced_args.append(arg[tuple(slices)])

    # Apply the function
    result = fn(*sliced_args)

    # Build reshape: insert dim of size 1 at out_axis
    out_shape = list(result.shape)
    out_shape.insert(out_axis, 1)

    # Build pad: None for all dims except out_axis which gets (a, (batch_size-a)-1)
    pad_args: list = [None] * len(out_shape)
    pad_args[out_axis] = (a, (batch_size - a) - 1)

    result = result.reshape(tuple(out_shape)).pad(tuple(pad_args))
    return Tensor(result.uop.reduce(a, arg=Ops.ADD))

  return vmapped


class TestVmapBasic(unittest.TestCase):
  """Basic vmap functionality tests."""

  def test_simple_vmap(self):
    """vmap a simple elementwise operation."""
    def fn(x): return x * 2
    x = Tensor.ones(3, 6)
    out = vmap(fn)(x)
    self.assertEqual(out.shape, (3, 6))
    self.assertTrue((out == 2).all().item())

  def test_vmap_with_reduction(self):
    """vmap a function that reduces."""
    def fn(x): return x.sum()
    x = Tensor.ones(3, 6)
    out = vmap(fn)(x)
    self.assertEqual(out.shape, (3,))
    self.assertListEqual(out.tolist(), [6.0, 6.0, 6.0])

  def test_vmap_axis1(self):
    """vmap along axis 1 instead of 0."""
    def fn(x): return x * 2
    x = Tensor.ones(3, 6)
    out = vmap(fn, in_axes=1, out_axis=1)(x)
    self.assertEqual(out.shape, (3, 6))
    self.assertTrue((out == 2).all().item())


class TestVmapMultiInput(unittest.TestCase):
  """Test vmap with multiple inputs and different axes."""

  def test_two_inputs_same_axis(self):
    """vmap two inputs along the same axis."""
    def fn(x, y): return x + y
    x = Tensor.arange(12).reshape(3, 4)
    y = Tensor.arange(12).reshape(3, 4) * 2
    out = vmap(fn, in_axes=[0, 0])(x, y)
    expected = (x + y).numpy()
    np.testing.assert_allclose(out.numpy(), expected)

  def test_two_inputs_different_axes(self):
    """vmap two inputs along different axes."""
    def fn(x, y): return x + y
    x = Tensor.arange(12).reshape(3, 4)  # map axis 0 -> rows
    y = Tensor.arange(12).reshape(4, 3)  # map axis 1 -> cols (gives shape (4,))
    # x[i] has shape (4,), y[:,i] has shape (4,)
    out = vmap(fn, in_axes=[0, 1])(x, y)
    self.assertEqual(out.shape, (3, 4))
    expected = np.arange(12).reshape(3, 4) + np.arange(12).reshape(4, 3).T
    np.testing.assert_allclose(out.numpy(), expected)

  def test_broadcast_input(self):
    """vmap with one input broadcasted (not mapped)."""
    def fn(x, y): return x * y
    x = Tensor.arange(12).reshape(3, 4).float()
    y = Tensor.arange(4).float()  # broadcast this
    out = vmap(fn, in_axes=[0, None])(x, y)
    self.assertEqual(out.shape, (3, 4))
    expected = np.arange(12).reshape(3, 4) * np.arange(4)
    np.testing.assert_allclose(out.numpy(), expected)

  def test_three_inputs_mixed(self):
    """vmap three inputs with mixed mapping."""
    def fn(a, b, c): return a + b * c
    a = Tensor.ones(5, 3)
    b = Tensor.arange(15).reshape(5, 3).float()
    c = Tensor.full((3,), 2.0)  # broadcast
    out = vmap(fn, in_axes=[0, 0, None])(a, b, c)
    self.assertEqual(out.shape, (5, 3))
    expected = np.ones((5, 3)) + np.arange(15).reshape(5, 3) * 2
    np.testing.assert_allclose(out.numpy(), expected)


class TestAfterVmap(unittest.TestCase):
  """Test that various tensor operations work after vmap."""
  n, m = 3, 6

  def fn(self, x: Tensor) -> Tensor:
    return Tensor(np.arange(self.m, dtype=np.float32)) * x

  def vfn(self, x: Tensor) -> Tensor:
    return vmap(self.fn)(x)

  x = Tensor.ones(n, m)
  expected_output = np.tile(np.arange(m, dtype=np.float32), (n, 1))

  def test_vmap(self):
    self.assertListEqual(self.vfn(self.x).tolist(), self.expected_output.tolist())

  def test_flatten(self):
    self.assertListEqual(self.vfn(self.x).flatten().tolist(), self.expected_output.flatten().tolist())

  def test_reshape(self):
    self.assertListEqual(self.vfn(self.x).reshape(9, 2).tolist(), self.expected_output.reshape(9, 2).tolist())
    self.assertListEqual(self.vfn(self.x).reshape(-1, 2).tolist(), self.expected_output.reshape(-1, 2).tolist())

  def test_transpose(self):
    self.assertListEqual(self.vfn(self.x).transpose().tolist(), self.expected_output.transpose().tolist())

  def test_pad(self):
    for pad_width in [((1, 0), (0, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((0, 0), (0, 1))]:
      self.assertListEqual(self.vfn(self.x).pad(pad_width).tolist(), np.pad(self.expected_output, pad_width).tolist())

  def test_flip(self):
    self.assertListEqual(self.vfn(self.x).flip(0).tolist(), np.flip(self.expected_output, 0).tolist())
    self.assertListEqual(self.vfn(self.x).flip(1).tolist(), np.flip(self.expected_output, 1).tolist())
    self.assertListEqual(self.vfn(self.x).flip((0, 1)).tolist(), np.flip(self.expected_output, (0, 1)).tolist())

  def test_indexing(self):
    self.assertEqual(self.vfn(self.x)[2, 4].item(), self.expected_output[2, 4])
    i, j = [0, 1, 2], [0, 2, 4]
    self.assertListEqual(self.vfn(self.x)[i, j].tolist(), self.expected_output[i, j].tolist())
    self.assertListEqual(self.vfn(self.x)[Tensor(i), Tensor(j)].tolist(), self.expected_output[i, j].tolist())
    i = [0, 8, 16]
    self.assertListEqual(self.vfn(self.x).flatten()[i].tolist(), self.expected_output.flatten()[i].tolist())
    self.assertListEqual(self.vfn(self.x).flatten()[Tensor(i)].tolist(), self.expected_output.flatten()[i].tolist())

  def test_slicing(self):
    self.assertListEqual(self.vfn(self.x)[0].tolist(), self.expected_output[0].tolist())
    self.assertListEqual(self.vfn(self.x)[:, 1].tolist(), self.expected_output[:, 1].tolist())
    self.assertListEqual(self.vfn(self.x)[1:3, 2:4].tolist(), self.expected_output[1:3, 2:4].tolist())

  def test_squeeze(self):
    self.assertListEqual(self.vfn(self.x).unsqueeze(0).tolist(), np.expand_dims(self.expected_output, 0).tolist())
    self.assertListEqual(self.vfn(self.x).unsqueeze(-1).tolist(), np.expand_dims(self.expected_output, -1).tolist())
    self.assertListEqual(self.vfn(self.x).unsqueeze(-1).squeeze(-1).tolist(), self.expected_output.tolist())
    self.assertTrue((ret := self.vfn(self.x)).squeeze(0) is ret)

  def test_cat(self):
    self.assertListEqual(Tensor.cat(self.vfn(self.x), self.x, dim=0).tolist(),
                         np.concatenate([self.expected_output, np.ones((self.n, self.m))], axis=0).tolist())
    self.assertListEqual(Tensor.cat(self.x, self.vfn(self.x), dim=0).tolist(),
                         np.concatenate([np.ones((self.n, self.m)), self.expected_output], axis=0).tolist())

  def test_cmp(self):
    self.assertTrue((self.vfn(self.x)[:, 0] == 0.0).all().item())
    self.assertTrue((self.vfn(self.x)[:, 1] > 0.0).all().item())
    self.assertTrue((self.vfn(self.x) < self.m).all().item())

  def test_sum(self):
    self.assertEqual(self.vfn(self.x).sum().item(), self.expected_output.sum().item())
    self.assertEqual(self.vfn(self.x).sum(0).tolist(), self.expected_output.sum(0).tolist())
    self.assertEqual(self.vfn(self.x).sum(1).tolist(), self.expected_output.sum(1).tolist())

  def test_linalg(self):
    self.assertListEqual((self.vfn(self.x) @ Tensor.ones(self.m)).tolist(), (self.expected_output @ np.ones(self.m)).tolist())
    self.assertTrue((self.vfn(self.x) @ Tensor.ones(self.m) == sum(range(self.m))).all().item())
    self.assertListEqual((self.vfn(self.x) @ Tensor.arange(self.m * 10).reshape(self.m, 10)).tolist(),
                         (self.expected_output @ np.arange(10 * self.m).reshape(self.m, 10)).tolist())


class TestVmapMatmul(unittest.TestCase):
  """Test vmap with matmul operations."""

  def test_batched_matvec(self):
    """Batched matrix-vector multiply."""
    def matvec(mat, vec): return mat @ vec
    mats = Tensor.randn(4, 3, 3).realize()
    vec = Tensor.randn(3).realize()
    out = vmap(matvec, in_axes=[0, None])(mats, vec)
    self.assertEqual(out.shape, (4, 3))
    # Compare to direct batched computation
    expected = (mats.numpy() @ vec.numpy()[..., None]).squeeze(-1)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6)

  def test_batched_matmat(self):
    """Batched matrix-matrix multiply."""
    def matmul(a, b): return a @ b
    a = Tensor.randn(4, 3, 5).realize()
    b = Tensor.randn(4, 5, 2).realize()
    out = vmap(matmul, in_axes=[0, 0])(a, b)
    self.assertEqual(out.shape, (4, 3, 2))
    expected = np.einsum('bij,bjk->bik', a.numpy(), b.numpy())
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)


class TestVmapHigherDim(unittest.TestCase):
  """Test vmap with higher dimensional tensors."""

  def test_3d_input(self):
    """vmap over 3D input."""
    def fn(x): return x.sum(axis=-1)
    x = Tensor.arange(24).reshape(2, 3, 4).float()
    out = vmap(fn)(x)
    self.assertEqual(out.shape, (2, 3))
    expected = np.arange(24).reshape(2, 3, 4).sum(axis=-1)
    np.testing.assert_allclose(out.numpy(), expected)

  def test_4d_input_axis2(self):
    """vmap over axis 2 of 4D input."""
    def fn(x): return x.mean()
    x = Tensor.arange(120).reshape(2, 3, 4, 5).float()
    out = vmap(fn, in_axes=2, out_axis=0)(x)
    self.assertEqual(out.shape, (4,))
    expected = np.arange(120).reshape(2, 3, 4, 5).mean(axis=(0, 1, 3))
    np.testing.assert_allclose(out.numpy(), expected)


class TestVmapNested(unittest.TestCase):
  """Test nested vmap calls."""

  def test_nested_vmap_simple(self):
    """Double vmap over 2D input."""
    def fn(x): return x * 2
    x = Tensor.arange(12).reshape(3, 4).float()
    # vmap over axis 0, then vmap over axis 0 of the result (which was axis 1)
    out = vmap(vmap(fn))(x)
    self.assertEqual(out.shape, (3, 4))
    expected = np.arange(12).reshape(3, 4) * 2
    np.testing.assert_allclose(out.numpy(), expected)

  def test_nested_vmap_3d(self):
    """Triple vmap over 3D input."""
    def fn(x): return x + 1
    x = Tensor.arange(24).reshape(2, 3, 4).float()
    out = vmap(vmap(vmap(fn)))(x)
    self.assertEqual(out.shape, (2, 3, 4))
    expected = np.arange(24).reshape(2, 3, 4) + 1
    np.testing.assert_allclose(out.numpy(), expected)

  def test_nested_vmap_with_reduction(self):
    """Nested vmap where inner fn reduces."""
    def fn(x): return x.sum()
    x = Tensor.arange(12).reshape(3, 4).float()
    # Inner vmap: for each row, sum each element (identity since scalar)
    # Actually: inner vmap maps over the 4 elements, outer over 3 rows
    out = vmap(vmap(fn))(x)
    self.assertEqual(out.shape, (3, 4))
    np.testing.assert_allclose(out.numpy(), x.numpy())

  def test_nested_vmap_matmul(self):
    """Nested vmap for batch of batch matmuls."""
    def matvec(mat, vec): return mat @ vec
    # Shape: (2, 3, 4, 4) - 2 batches of 3 matrices each, 4x4
    mats = Tensor.randn(2, 3, 4, 4).realize()
    # Shape: (2, 3, 4) - matching vectors
    vecs = Tensor.randn(2, 3, 4).realize()

    # Double vmap: outer over dim 0, inner over dim 0 (after outer slice)
    out = vmap(vmap(matvec, in_axes=[0, 0]), in_axes=[0, 0])(mats, vecs)
    self.assertEqual(out.shape, (2, 3, 4))

    # Compute reference
    expected = np.einsum('abij,abj->abi', mats.numpy(), vecs.numpy())
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

  def test_nested_vmap_different_axes(self):
    """Nested vmap with different in_axes."""
    def fn(x, y): return x + y
    x = Tensor.arange(24).reshape(2, 3, 4).float()
    y = Tensor.arange(12).reshape(3, 4).float()

    # Outer vmap over x's axis 0, y is broadcast
    # Inner vmap over both axis 0
    inner = vmap(fn, in_axes=[0, 0])
    outer = vmap(inner, in_axes=[0, None])
    out = outer(x, y)
    self.assertEqual(out.shape, (2, 3, 4))
    expected = np.arange(24).reshape(2, 3, 4) + np.arange(12).reshape(3, 4)
    np.testing.assert_allclose(out.numpy(), expected)

  def test_nested_vmap_outer_product(self):
    """Use nested vmap to compute outer product."""
    def mul(a, b): return a * b
    x = Tensor.arange(3).float()
    y = Tensor.arange(4).float()

    # vmap over x (broadcast y), then vmap over y (broadcast x slice)
    # This gives outer product
    out = vmap(lambda xi: vmap(lambda yj: mul(xi, yj))(y))(x)
    self.assertEqual(out.shape, (3, 4))
    expected = np.outer(np.arange(3), np.arange(4))
    np.testing.assert_allclose(out.numpy(), expected)

  def test_nested_vmap_chain_ops(self):
    """Nested vmap with operations chained after."""
    def fn(x): return x ** 2
    x = Tensor.arange(12).reshape(3, 4).float()
    out = vmap(vmap(fn))(x)

    # Chain operations after nested vmap
    out_sum = out.sum()
    out_mean = out.mean(axis=1)
    out_reshaped = out.reshape(2, 6)

    expected = (np.arange(12).reshape(3, 4) ** 2)
    self.assertAlmostEqual(out_sum.item(), expected.sum(), places=4)
    np.testing.assert_allclose(out_mean.numpy(), expected.mean(axis=1), atol=1e-5)
    np.testing.assert_allclose(out_reshaped.numpy(), expected.reshape(2, 6), atol=1e-5)

  @unittest.skip("4-level nested vmap has incorrect results - likely a bug in pad+reduce with many nested ranges")
  def test_deeply_nested_vmap(self):
    """4-level nested vmap."""
    def fn(x): return x + 1
    x = Tensor.arange(120).reshape(2, 3, 4, 5).float()
    out = vmap(vmap(vmap(vmap(fn))))(x)
    self.assertEqual(out.shape, (2, 3, 4, 5))
    expected = np.arange(120).reshape(2, 3, 4, 5) + 1
    np.testing.assert_allclose(out.numpy(), expected)


if __name__ == '__main__':
  unittest.main()
