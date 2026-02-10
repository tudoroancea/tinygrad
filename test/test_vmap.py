import unittest
import numpy as np
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType

# *** helpers ***

def _varg(batch_size:int, in_axis:int, ndim:int, axis_id:int=-1):
  """Build VMAPIN/VMAPOUT arg tuple: RANGE at in_axis, CONST(0) elsewhere."""
  r = UOp.range(batch_size, axis_id, AxisType.LOOP)
  return tuple(r if i == in_axis else UOp.const(dtypes.index, 0) for i in range(ndim))

def _vmap_simple(fn, x:Tensor, in_axis:int=0, axis_id:int=-1):
  """Simple vmap: apply fn per-slice along in_axis, output batch dim at position 0."""
  batch_size = x.shape[in_axis]
  varg_in = _varg(batch_size, in_axis, x.ndim, axis_id)
  result = fn(x.vmapin(varg_in))
  varg_out = (varg_in[in_axis],) + tuple(UOp.const(dtypes.index, 0) for _ in range(result.ndim))
  return result.vmapout(varg_out)

# *** low-level VMAPIN/VMAPOUT scheduling tests ***

class TestVmapScheduling(unittest.TestCase):
  def test_vmapin_shape(self):
    x = UOp.new_buffer("CPU", 18, dtypes.float, 0).reshape((3,6))
    rm1 = UOp.range(3,-1)
    rm2 = UOp.range(6,-2)
    for varg, shape in [ ((rm1, rm1.const_like(0)), (6,)), ((rm2.const_like(0), rm2), (3,)), ((rm1, rm2), ()) ]:
      assert x.vmapin(varg).shape == shape

  def test_vmapout_shape(self):
    x = UOp.new_buffer("CPU", 18, dtypes.float, 0).reshape((3,6))
    rm1 = UOp.range(3,-1)
    rm2 = UOp.range(6,-2)
    for varg in [ (rm1, rm1.const_like(0)),(rm2.const_like(0), rm2), (rm1, rm2) ]:
      assert x.vmapin(varg).vmapout(varg).shape == x.shape

  def test_vmap_schedule(self):
    x = UOp.new_buffer("CPU", 18, dtypes.float, 0).reshape((3,6))
    varg = (UOp.range(3,-1), UOp.const(dtypes.index, 0))
    a = UOp.new_buffer("CPU", 6, dtypes.float, 1)
    ast = (x.vmapin(varg) * a).vmapout(varg).contiguous()
    Tensor(ast, "CPU").schedule()

  def test_vmap_dim1_schedule(self):
    x = UOp.new_buffer("CPU", 6, dtypes.float, 0).reshape((1,6))
    varg = (UOp.range(1,-1), UOp.const(dtypes.index, 0))
    a = UOp.new_buffer("CPU", 6, dtypes.float, 1)
    ast = (x.vmapin(varg) * a).vmapout(varg).contiguous()
    Tensor(ast, "CPU").schedule()

  def test_multiple_consumers(self):
    x = UOp.new_buffer("CPU", 18, dtypes.float, 0).reshape((3,6))
    varg = (UOp.range(3,-1), UOp.const(dtypes.index, 0))
    a = UOp.new_buffer("CPU", 6, dtypes.float, 1)
    b = UOp.new_buffer("CPU", 6, dtypes.float, 1).reshape((3,2))
    c0 = x.vmapin(varg)
    c1 = c0 * a
    c2 = (c0.reshape((3,2)) * b).reshape((6,))
    ast0 = c1.vmapout(varg).contiguous()
    ast1 = (c1+c2).vmapout(varg).contiguous()
    Tensor.schedule(Tensor(ast0, "CPU"),Tensor(ast1, "CPU"))

  def test_3_stacked_multi_index_schedule(self):
    """3 stacked elementwise results with multi-index — scheduling only."""
    def fn(x):
      v0, v1, v2 = Tensor([1., 0., 0.]), Tensor([0., 1., 0.]), Tensor([0., 0., 1.])
      flat = Tensor.stack(x * v0, x * v1, x * v2).flatten()
      return Tensor.stack(flat[0], flat[1], flat[2])
    _vmap_simple(fn, Tensor.randn(10, 3).realize()).schedule()

  def test_reduce_multi_index_schedule(self):
    """Stacked reduces with multi-index — scheduling only."""
    def fn(x):
      s0 = (x * Tensor([1., 0., 0.])).sum().reshape((1,))
      s1 = (x * Tensor([0., 1., 0.])).sum().reshape((1,))
      flat = Tensor.stack(s0, s1, dim=1).flatten()
      return Tensor.stack(flat[0], flat[1])
    _vmap_simple(fn, Tensor.randn(10, 3).realize()).schedule()

  def test_spjacobian_pattern_schedule(self):
    """The spjacobian unroll pattern: stack JVPs, flatten, re-index — scheduling only."""
    def jvp(x, v): return (2 * x * v).sum().reshape((1,))
    def fn(x):
      stacked = Tensor.stack(jvp(x, Tensor([1.,0.,0.])), jvp(x, Tensor([0.,1.,0.])), jvp(x, Tensor([0.,0.,1.])), dim=1)
      flat = stacked.flatten()
      return Tensor.stack(*[flat[i] for i in [0, 1, 2]])
    _vmap_simple(fn, Tensor.randn(10, 3).realize()).schedule()

# *** post-vmap tensor operations ***

class TestAfterVmap(unittest.TestCase):
  n, m = 3, 6
  def fn(self, x:Tensor) -> Tensor: return Tensor(np.arange(self.m, dtype=np.float32)) * x
  def vfn(self, x:Tensor) -> Tensor:
    varg = (UOp.range(self.n, -1), UOp.const(dtypes.index, 0))
    return self.fn(x.vmapin(varg)).vmapout(varg)

  x = Tensor.ones(n, m)
  expected_output = np.tile(np.arange(m, dtype=np.float32), (n, 1))

  def test_vmap(self):
    self.assertListEqual(self.vfn(self.x).tolist(), self.expected_output.tolist())

  def test_flatten(self):
    self.assertListEqual(self.vfn(self.x).flatten().tolist(), self.expected_output.flatten().tolist())

  def test_reshape(self):
    self.assertListEqual(self.vfn(self.x).reshape(9,2).tolist(), self.expected_output.reshape(9,2).tolist())
    self.assertListEqual(self.vfn(self.x).reshape(-1,2).tolist(), self.expected_output.reshape(-1,2).tolist())

  def test_transpose(self):
    self.assertListEqual(self.vfn(self.x).transpose().tolist(), self.expected_output.transpose().tolist())

  def test_pad(self):
    for pad_width in [((1,0), (0,0)),((0,1),(0,0)), ((0,0), (1,0)), ((0,0), (0,1))]:
      self.assertListEqual(self.vfn(self.x).pad(pad_width).tolist(), np.pad(self.expected_output, pad_width).tolist())

  def test_flip(self):
    self.assertListEqual(self.vfn(self.x).flip(0).tolist(), np.flip(self.expected_output,0).tolist())
    self.assertListEqual(self.vfn(self.x).flip(1).tolist(), np.flip(self.expected_output,1).tolist())
    self.assertListEqual(self.vfn(self.x).flip((0,1)).tolist(), np.flip(self.expected_output,(0,1)).tolist())

  def test_indexing(self):
    self.assertEqual(self.vfn(self.x)[2,4].item(), self.expected_output[2,4])
    i,j = [0,1,2], [0,2,4]
    self.assertListEqual(self.vfn(self.x)[i,j].tolist(), self.expected_output[i,j].tolist())
    self.assertListEqual(self.vfn(self.x)[Tensor(i), Tensor(j)].tolist(), self.expected_output[i,j].tolist())
    i = [0,8,16]
    self.assertListEqual(self.vfn(self.x).flatten()[i].tolist(), self.expected_output.flatten()[i].tolist())
    self.assertListEqual(self.vfn(self.x).flatten()[Tensor(i)].tolist(), self.expected_output.flatten()[i].tolist())

  def test_slicing(self):
    self.assertListEqual(self.vfn(self.x)[0].tolist(), self.expected_output[0].tolist())
    self.assertListEqual(self.vfn(self.x)[:, 1].tolist(), self.expected_output[:, 1].tolist())
    self.assertListEqual(self.vfn(self.x)[1:3, 2:4].tolist(), self.expected_output[1:3, 2:4].tolist())

  def test_squeeze(self):
    self.assertListEqual(self.vfn(self.x).unsqueeze(0).tolist(), np.expand_dims(self.expected_output,0).tolist())
    self.assertListEqual(self.vfn(self.x).unsqueeze(-1).tolist(), np.expand_dims(self.expected_output,-1).tolist())
    self.assertListEqual(self.vfn(self.x).unsqueeze(-1).squeeze(-1).tolist(), self.expected_output.tolist())
    self.assertTrue((ret:=self.vfn(self.x)).squeeze(0) is ret)
    self.assertTrue((ret:=self.vfn(self.x)).squeeze(0) is ret)

  def test_cat(self):
    self.assertListEqual(Tensor.cat(self.vfn(self.x), self.x, dim=0).tolist(),
                        np.concatenate([self.expected_output, np.ones((self.n, self.m))], axis=0).tolist())
    self.assertListEqual(Tensor.cat(self.x, self.vfn(self.x), dim=0).tolist(),
                        np.concatenate([np.ones((self.n, self.m)), self.expected_output], axis=0).tolist())

  def test_cmp(self):
    self.assertTrue((self.vfn(self.x)[:,0] == 0.0).all().item())
    self.assertTrue((self.vfn(self.x)[:,1] > 0.0).all().item())
    self.assertTrue((self.vfn(self.x) < self.m).all().item())

  def test_sum(self):
    self.assertEqual(self.vfn(self.x).sum().item(), self.expected_output.sum().item())
    self.assertEqual(self.vfn(self.x).sum(0).tolist(), self.expected_output.sum(0).tolist())
    self.assertEqual(self.vfn(self.x).sum(1).tolist(), self.expected_output.sum(1).tolist())

  def test_linalg(self):
    self.assertListEqual((self.vfn(self.x) @ Tensor.ones(self.m)).tolist(), (self.expected_output @ np.ones(self.m)).tolist())
    self.assertTrue((self.vfn(self.x) @ Tensor.ones(self.m) == sum(range(self.m))).all().item())
    self.assertListEqual((self.vfn(self.x) @ Tensor.arange(self.m*10).reshape(self.m, 10)).tolist(),
                         (self.expected_output @ np.arange(10*self.m).reshape(self.m, 10)).tolist())

# *** correctness: indexing reduce-stacked tensors inside vmap ***

class TestVmapReduceIndex(unittest.TestCase):
  """Regression tests for vmap RANGE loss when indexing reduce-stacked tensors.

  The bug: when a tensor between VMAPIN and VMAPOUT gets realized (e.g. due to
  multi-consumer or ending-ranges), the intermediate buffer didn't include the
  vmap batch dimension. All batch elements silently got the last batch's values.
  """
  def setUp(self):
    np.random.seed(42)
    self.X_np = np.random.randn(4, 3).astype(np.float64)
    self.X = Tensor(self.X_np).realize()
    self.r = UOp.range(3, -10, AxisType.LOOP)

  def _vmapin(self): return self.X.vmapin((UOp.const(dtypes.index, 0), self.r))
  def _vmapout(self, t): return t.vmapout((self.r, UOp.const(dtypes.index, 0)))

  def test_reduce_stack_direct(self):
    """Baseline: stacked reduces without re-indexing produce correct per-batch values."""
    x = self._vmapin()
    z = Tensor.stack(x[:2].sum(), x[2:].sum())
    result = self._vmapout(z).numpy()
    expected = np.array([[self.X_np[:2, b].sum(), self.X_np[2:, b].sum()] for b in range(3)])
    np.testing.assert_allclose(result, expected)

  def test_reduce_stack_reindex(self):
    """Re-indexing a reduce-stacked tensor must preserve per-batch values."""
    x = self._vmapin()
    z = Tensor.stack(x[:2].sum(), x[2:].sum())
    z_reindexed = z[[0, 1]]
    result = self._vmapout(z_reindexed).numpy()
    expected = np.array([[self.X_np[:2, b].sum(), self.X_np[2:, b].sum()] for b in range(3)])
    np.testing.assert_allclose(result, expected)

  def test_reduce_stack_reindex_reversed(self):
    """Re-indexing with reversed order must swap correctly per batch."""
    x = self._vmapin()
    z = Tensor.stack(x[:2].sum(), x[2:].sum())
    z_rev = z[[1, 0]]
    result = self._vmapout(z_rev).numpy()
    expected = np.array([[self.X_np[2:, b].sum(), self.X_np[:2, b].sum()] for b in range(3)])
    np.testing.assert_allclose(result, expected)

  def test_reduce_stack_reindex_duplicate(self):
    """Re-indexing with duplicate indices must replicate correctly per batch."""
    x = self._vmapin()
    z = Tensor.stack(x[:2].sum(), x[2:].sum())
    z_dup = z[[0, 0]]
    result = self._vmapout(z_dup).numpy()
    expected = np.array([[self.X_np[:2, b].sum(), self.X_np[:2, b].sum()] for b in range(3)])
    np.testing.assert_allclose(result, expected)

  def test_no_reduce_reindex(self):
    """Without reduce ops, indexing + re-stacking must also be correct."""
    x = self._vmapin()
    y = x[:2]
    y_indexed = y[[0, 1]]
    direct = self._vmapout(y).numpy()
    reindexed = self._vmapout(y_indexed).numpy()
    np.testing.assert_allclose(reindexed, direct)

  def test_matmul_reindex(self):
    """Matmul (another reduce op) followed by re-indexing must be correct."""
    x = self._vmapin()
    M = Tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    t = x @ M
    direct = self._vmapout(t).numpy()
    reindexed = self._vmapout(t[[0, 1]]).numpy()
    expected = self.X_np.T @ np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    np.testing.assert_allclose(direct, expected)
    np.testing.assert_allclose(reindexed, expected)

  def test_multi_reduce_stack_reindex(self):
    """3 stacked reduces with multi-index — the spjacobian pattern."""
    x = self._vmapin()
    s0 = x[:1].sum()
    s1 = x[1:3].sum()
    s2 = x[3:].sum()
    z = Tensor.stack(s0, s1, s2)
    z_reindexed = z[[2, 0, 1]]
    result = self._vmapout(z_reindexed).numpy()
    expected = np.array([
      [self.X_np[3:, b].sum(), self.X_np[:1, b].sum(), self.X_np[1:3, b].sum()] for b in range(3)
    ])
    np.testing.assert_allclose(result, expected)

# *** correctness: vmap with _vmap_simple helper ***

class TestVmapCorrectness(unittest.TestCase):
  """Correctness tests using the _vmap_simple helper."""

  def test_elementwise(self):
    x = Tensor.ones(5, 4).contiguous().realize()
    w = Tensor.arange(4, dtype=dtypes.float32).realize()
    result = _vmap_simple(lambda row: row * w, x)
    expected = np.tile(np.arange(4, dtype=np.float32), (5, 1))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_sum_per_row(self):
    x = Tensor.ones(5, 4).contiguous().realize()
    result = _vmap_simple(lambda row: row.sum().reshape((1,)), x)
    expected = np.full((5, 1), 4, dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_matmul_per_row(self):
    np.random.seed(0)
    x = Tensor(np.random.randn(5, 4).astype(np.float32)).realize()
    M = Tensor(np.random.randn(4, 3).astype(np.float32)).realize()
    result = _vmap_simple(lambda row: row @ M, x)
    expected = x.numpy() @ M.numpy()
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

  def test_multi_index_diagonal(self):
    """Extract diagonal of per-row Jacobian via multi-index — the core spjacobian pattern."""
    x = Tensor.ones(5, 3).contiguous().realize()
    def fn(row):
      v0, v1, v2 = Tensor([1., 0., 0.]), Tensor([0., 1., 0.]), Tensor([0., 0., 1.])
      flat = Tensor.stack(row * v0, row * v1, row * v2).flatten()
      return Tensor.stack(flat[0], flat[4], flat[8])
    result = _vmap_simple(fn, x)
    expected = np.ones((5, 3), dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_reduce_then_index(self):
    """Reduce + index — the minimal pattern that triggered the bug."""
    np.random.seed(42)
    x_np = np.random.randn(4, 3).astype(np.float32)
    x = Tensor(x_np).realize()
    def fn(col):
      return Tensor.stack(col[:2].sum(), col[2:].sum())[[0, 1]]
    result = _vmap_simple(fn, x, in_axis=1)
    expected = np.array([[x_np[:2, b].sum(), x_np[2:, b].sum()] for b in range(3)], dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_jvp_stack_reindex(self):
    """Full spjacobian-like pattern: JVPs → stack → flatten → re-index."""
    np.random.seed(0)
    x_np = np.random.randn(5, 3).astype(np.float32)
    x = Tensor(x_np).realize()
    def fn(row):
      jvps = [((2 * row * v).sum()).reshape((1,)) for v in [Tensor([1.,0.,0.]), Tensor([0.,1.,0.]), Tensor([0.,0.,1.])]]
      flat = Tensor.stack(*jvps, dim=1).flatten()
      return Tensor.stack(*[flat[i] for i in [0, 1, 2]])
    result = _vmap_simple(fn, x)
    expected = 2 * x_np  # diagonal of diag(2*x) is just 2*x
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_nested_vmap_stack_scalar(self):
    """Nested vmap + stack should preserve both outer and inner vmap ranges."""
    np.random.seed(0)
    z_np = np.random.randn(4, 5).astype(np.float32)
    z = Tensor(z_np).realize()
    r_inner = UOp.range(2, -1, AxisType.LOOP)
    varg_inner = (UOp.const(dtypes.index, 0), r_inner)
    def fn(row):
      xv = row[:4].reshape(2, 2).vmapin(varg_inner)
      g = row[4:]
      y = xv * g
      out = (g * Tensor.stack(y[0], y[1])).vmapout(varg_inner).flatten()
      return out
    result = _vmap_simple(fn, z, axis_id=-2)
    expected = np.stack([
      (z_np[b, :4].reshape(2, 2) * (z_np[b, 4] ** 2)).reshape(-1)
      for b in range(z_np.shape[0])
    ])
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

if __name__ == '__main__':
  unittest.main()
