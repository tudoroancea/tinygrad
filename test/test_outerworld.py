import unittest
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import AxisType, Ops

class TestOuterworldReduce(unittest.TestCase):
  def test_reduce(self):
    x = Tensor.ones(5, 5).contiguous()
    a = UOp.range(5, -1, AxisType.REDUCE)
    out = x[a]
    # TODO: syntax for this
    t = Tensor(UOp(Ops.REDUCE, dtype=out.uop.dtype, src=(out.uop, a), arg=Ops.ADD))
    self.assertListEqual(t.tolist(), [5.,5.,5.,5.,5.])

class TestOuterworld(unittest.TestCase):
  def test_range_plus_1(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t+1==cpy).all().item())

  def test_range_plus_1_transpose(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(10, 1).expand(10, a).contiguous().realize()

    self.assertTrue(((t+1).T==cpy).all().item())

  def test_flip_range(self):
    t = Tensor.rand(10, 10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[9-a]
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t.flip(0)==cpy).all().item())

  def test_vmap(self):
    def f(x): return x.sum(axis=0)*2

    x = Tensor.ones(3, 10, 2).contiguous()

    # vmap across axis 0
    a = UOp.range(3, -1)
    out = f(x[a])
    out = out.reshape(1, 2).expand(a, 2).contiguous()

    # 3x2 grid of 20
    out.realize()
    self.assertTrue((out==20).all().item())

  def test_fancy_vmap(self):
    def f(x,y): return x+y

    x = Tensor.arange(9).reshape(3,3).contiguous()
    y = Tensor.arange(9).reshape(3,3).contiguous()

    a = UOp.range(3, -1)
    out = f(x[:,a], y[a,:])
    # TODO: this should support flatten
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    self.assertListEqual([[0,4,8],[4,8,12],[8,12,16]], out.tolist())

  def test_indexing_after_vmap(self):
    # Test: after vmapping, do additional fancy indexing
    x = Tensor.arange(24).reshape(4, 6).contiguous()

    # vmap across axis 0
    a = UOp.range(4, -1)
    vmapped = x[a]  # shape should be (6,) but with outerworld range

    # Now do fancy indexing on the vmapped result
    indices = Tensor([0, 2, 4])
    result = vmapped[indices]  # Select elements 0, 2, 4 from each row

    # Expand back to get the full result
    result = result.reshape(1, 3).expand(a, 3).contiguous().realize()

    # Expected: [[0,2,4], [6,8,10], [12,14,16], [18,20,22]]
    expected = [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18, 20, 22]]
    self.assertListEqual(result.tolist(), expected)

  def test_flat_indexing_after_vmap(self):
    # Test: after vmapping, reshape then do flat indexing
    x = Tensor.arange(24).reshape(4, 2, 3).contiguous()

    # vmap across axis 0
    a = UOp.range(4, -1)
    vmapped = x[a]  # shape should be (2, 3) but with outerworld range

    # Flatten the vmapped tensor - this requires computing prod(shape)
    # which is tricky when shape contains RANGE ops
    flattened = vmapped.flatten()  # Should be shape (6,)

    # Now do flat indexing
    result = flattened[3]  # Get element at flat position 3

    # Expand back to get the full result
    result = result.reshape(1).expand(a).contiguous().realize()

    # Expected: [3, 9, 15, 21] (element at flat position 3 of each 2x3 slice)
    expected = [3, 9, 15, 21]
    self.assertListEqual(result.tolist(), expected)

if __name__ == '__main__':
  unittest.main()