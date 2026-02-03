"""
Tests for vmap scheduling with multi-consumer indexing patterns.

These tests verify that vmap schedules correctly when:
1. Stacking 3+ computed tensors (e.g., results of x * v0, x * v1, x * v2)
2. Extracting multiple scalar indices from the flattened result via Tensor.stack(t[i], t[j], ...)

This pattern occurs in spjacobian when:
- Multiple JVPs are computed and stacked
- The sparse uncompression step extracts specific indices

The fix: vmap ranges are converted to AxisType.OUTER in run_rangeify so they can
remain open during kernel splitting (split_store only allows open OUTER ranges).
"""

from typing import Callable
import numpy as np

from tinygrad import Tensor, UOp, dtypes
from tinygrad.engine.schedule import complete_create_schedule_with_vars
from tinygrad.uop.ops import AxisType


def vmap(
  f: Callable[[Tensor], Tensor], in_axis: int = 0, axis_id: int = -1, axis_type: AxisType = AxisType.LOOP
) -> Callable[[Tensor], Tensor]:
  """
  Simple vmap implementation for testing scheduling.
  Maps f over the in_axis dimension, producing output with batch dim at position 0.
  """
  def wrapper(x: Tensor) -> Tensor:
    batch_size = x.shape[in_axis]
    r = UOp.range(batch_size, axis_id, axis_type)
    # Build varg: range for the batched axis, 0 for others
    varg = tuple(r if i == in_axis else UOp.const(dtypes.index, 0) for i in range(x.ndim))
    # Apply function and wrap output
    inner_result = f(x.vmapin(varg))
    # vmapout expects varg matching inner_result's shape + 1 for the batch dim
    out_varg = (r,) + tuple(UOp.const(dtypes.index, 0) for _ in range(inner_result.ndim))
    return inner_result.vmapout(out_varg)
  return wrapper


def can_schedule(t: Tensor) -> bool:
  """Returns True if the tensor can be scheduled, False otherwise."""
  sink = UOp.sink(t.uop)
  try:
    complete_create_schedule_with_vars(sink)
    return True
  except RuntimeError as e:
    if "input to kernel must be AFTER or BUFFER" in str(e):
      return False
    raise


class TestVmapMultiConsumerIndexing:
  """Tests isolating the multi-consumer indexing patterns."""

  def test_3_stacked_no_index(self):
    """3 stacked tensors without indexing - schedules and runs correctly."""
    def fn(x):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      stacked = Tensor.stack(x * v0, x * v1, x * v2)  # (3, 3)
      return stacked.flatten()  # (9,)

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_3_stacked_single_index(self):
    """3 stacked tensors with single index."""
    def fn(x):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      stacked = Tensor.stack(x * v0, x * v1, x * v2).flatten()
      return stacked[0].reshape((1,))  # Single scalar

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_3_stacked_shrink(self):
    """3 stacked tensors with shrink (not multi-index)."""
    def fn(x):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      stacked = Tensor.stack(x * v0, x * v1, x * v2).flatten()
      return stacked[:3]  # Shrink

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_2_stacked_multi_index(self):
    """2 stacked tensors with multi-index."""
    def fn(x):
      v0, v1 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0])
      stacked = Tensor.stack(x * v0, x * v1).flatten()  # (6,)
      return Tensor.stack(stacked[0], stacked[1], stacked[2])  # Multi-index

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_3_stacked_multi_index(self):
    """3 stacked tensors with multi-index - previously failing case."""
    def fn(x):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      stacked = Tensor.stack(x * v0, x * v1, x * v2).flatten()  # (9,)
      return Tensor.stack(stacked[0], stacked[1], stacked[2])  # Multi-consumer

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_3_stacked_2_index(self):
    """3 stacked tensors with 2 indices - previously failing case."""
    def fn(x):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      stacked = Tensor.stack(x * v0, x * v1, x * v2).flatten()
      return Tensor.stack(stacked[0], stacked[1])  # Just 2 indices

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()


class TestVmapReduceMultiIndex:
  """Tests with REDUCE operations."""

  def test_reduce_single_index(self):
    """Single REDUCE with single index."""
    def fn(x):
      reduced = x.sum().reshape((1,))
      return reduced[0].reshape((1,))

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_reduce_expand_multi_index(self):
    """REDUCE -> expand -> multi-index - previously failing case."""
    def fn(x):
      reduced = x.sum()  # scalar
      expanded = reduced.expand((3,))
      return Tensor.stack(expanded[0], expanded[1], expanded[2])

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_2_reduces_stacked_multi_index(self):
    """2 REDUCEs stacked with multi-index - previously failing case."""
    def fn(x):
      s0 = (x * Tensor([1.0, 0.0, 0.0])).sum().reshape((1,))
      s1 = (x * Tensor([0.0, 1.0, 0.0])).sum().reshape((1,))
      stacked = Tensor.stack(s0, s1, dim=1).flatten()  # (2,)
      return Tensor.stack(stacked[0], stacked[1])

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()


class TestSpjacobianPattern:
  """The actual pattern from spjacobian that triggers this bug."""

  def test_spjacobian_unroll_pattern(self):
    """
    spjacobian with unroll=True does:
    1. Stack multiple JVP results (each may contain REDUCE)
    2. Flatten
    3. Extract indices via Tensor.stack(*[flat[i] for i in uncompression_idx])
    """
    def jvp_fn(x, v):
      return (2 * x * v).sum().reshape((1,))

    def spjac_like(x):
      v0 = Tensor([1.0, 0.0, 0.0])
      v1 = Tensor([0.0, 1.0, 0.0])
      v2 = Tensor([0.0, 0.0, 1.0])

      jvp0 = jvp_fn(x, v0)
      jvp1 = jvp_fn(x, v1)
      jvp2 = jvp_fn(x, v2)

      stacked = Tensor.stack(jvp0, jvp1, jvp2, dim=1)
      flattened = stacked.flatten()

      uncompression_idx = [0, 1, 2]
      return Tensor.stack(*[flattened[i] for i in uncompression_idx])

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(spjac_like, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_spjacobian_pattern_no_reduce(self):
    """Same pattern but for a function without REDUCE."""
    def jvp_fn_no_reduce(x, v):
      return 2 * x * v

    def spjac_like(x):
      v0 = Tensor([1.0, 0.0, 0.0])
      v1 = Tensor([0.0, 1.0, 0.0])
      v2 = Tensor([0.0, 0.0, 1.0])

      jvp0 = jvp_fn_no_reduce(x, v0)
      jvp1 = jvp_fn_no_reduce(x, v1)
      jvp2 = jvp_fn_no_reduce(x, v2)

      stacked = Tensor.stack(jvp0, jvp1, jvp2)
      flattened = stacked.flatten()

      uncompression_idx = [0, 4, 8]  # Diagonal elements
      return Tensor.stack(*[flattened[i] for i in uncompression_idx])

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(spjac_like, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()


class TestMinimalReproduction:
  """Minimal cases for debugging."""

  def test_minimal_3_stack_2_index(self):
    """Minimal case: stack 3 computed tensors, extract 2 scalar indices."""
    def fn(x):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      flat = Tensor.stack(x * v0, x * v1, x * v2).flatten()
      return Tensor.stack(flat[0], flat[1])

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()

  def test_minimal_2_stack_3_index(self):
    """2 stacked tensors with 3 indices."""
    def fn(x):
      a = x * 1.0
      b = x * 2.0
      flat = Tensor.stack(a, b).flatten()
      return Tensor.stack(flat[0], flat[1], flat[2])

    x_batch = Tensor.randn(10, 3).realize()
    result = vmap(fn, in_axis=0)(x_batch)
    assert can_schedule(result)
    result.realize()


class TestVmapCorrectness:
  """
  Tests verifying vmap produces correct outputs.
  Uses the pattern from test_vmap.py where all batch elements have the same value.
  """

  def test_vmap_elementwise_ones(self):
    """vmap over elementwise multiplication with uniform input."""
    n, m = 5, 4
    x = Tensor.ones(n, m).contiguous().realize()
    weights = Tensor.arange(m, dtype=dtypes.float32).realize()

    varg = (UOp.range(n, -1), UOp.const(dtypes.index, 0))
    result = (x.vmapin(varg) * weights).vmapout(varg)
    result.realize()

    # With ones input, each row gets multiplied by weights
    expected = np.tile(np.arange(m, dtype=np.float32), (n, 1))
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_vmap_sum_ones(self):
    """vmap over sum with uniform input."""
    n, m = 5, 4
    x = Tensor.ones(n, m).contiguous().realize()

    varg = (UOp.range(n, -1), UOp.const(dtypes.index, 0))
    result = x.vmapin(varg).sum().reshape((1,)).vmapout((varg[0], UOp.const(dtypes.index, 0)))
    result.realize()

    # Sum of m ones = m for each row
    expected = np.full((n, 1), m, dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

  def test_vmap_multi_index_ones(self):
    """Verify the multi-index pattern with uniform input."""
    n = 5
    x = Tensor.ones(n, 3).contiguous().realize()

    def fn_per_row(row):
      v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
      stacked = Tensor.stack(row * v0, row * v1, row * v2).flatten()
      return Tensor.stack(stacked[0], stacked[4], stacked[8])

    varg = (UOp.range(n, -1), UOp.const(dtypes.index, 0))
    inner = fn_per_row(x.vmapin(varg))
    # inner has shape (3,), we want output shape (n, 3)
    out_varg = (varg[0], UOp.const(dtypes.index, 0))
    result = inner.vmapout(out_varg)
    result.realize()

    # Each row is [1,1,1], diagonal extracts [1,1,1]
    expected = np.ones((n, 3), dtype=np.float32)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
