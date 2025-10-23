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

  # Trigonometric operations
  def test_vmap_sin(self):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).contiguous()
    a = UOp.range(3, -1)
    out = x[a].sin()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.sin().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_cos(self):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).contiguous()
    a = UOp.range(3, -1)
    out = x[a].cos()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.cos().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_exp(self):
    x = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]).contiguous()
    a = UOp.range(3, -1)
    out = x[a].exp()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.exp().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_log(self):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).contiguous()
    a = UOp.range(3, -1)
    out = x[a].log()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.log().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_sqrt(self):
    x = Tensor([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0], [49.0, 64.0, 81.0]]).contiguous()
    a = UOp.range(3, -1)
    out = x[a].sqrt()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.sqrt().realize()
    self.assertTrue((out == expected).all().item())

  # Movement operations
  def test_vmap_transpose(self):
    x = Tensor.arange(60).reshape(3, 4, 5).contiguous()
    a = UOp.range(3, -1)
    out = x[a].transpose(0, 1)
    out = out.reshape(1, 5, 4).expand(a, 5, 4).contiguous().realize()
    expected = x.transpose(1, 2).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_permute(self):
    x = Tensor.arange(60).reshape(3, 4, 5).contiguous()
    a = UOp.range(3, -1)
    out = x[a].permute(1, 0)
    out = out.reshape(1, 5, 4).expand(a, 5, 4).contiguous().realize()
    expected = x.permute(0, 2, 1).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_reshape(self):
    x = Tensor.arange(24).reshape(3, 8).contiguous()
    a = UOp.range(3, -1)
    out = x[a].reshape(2, 4)
    out = out.reshape(1, 2, 4).expand(a, 2, 4).contiguous().realize()
    expected = x.reshape(3, 2, 4).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_pad(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a].pad(((1, 1), (2, 2)))
    out = out.reshape(1, 6, 8).expand(a, 6, 8).contiguous().realize()
    expected = x.pad(((0, 0), (1, 1), (2, 2))).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_squeeze(self):
    x = Tensor.arange(12).reshape(3, 1, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a].squeeze(0)
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = x.squeeze(1).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_unsqueeze(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a].unsqueeze(0)
    out = out.reshape(1, 1, 4).expand(a, 1, 4).contiguous().realize()
    expected = x.unsqueeze(1).realize()
    self.assertTrue((out == expected).all().item())

  # Indexing operations
  def test_vmap_slice(self):
    x = Tensor.arange(30).reshape(3, 10).contiguous()
    a = UOp.range(3, -1)
    out = x[a][2:7]
    out = out.reshape(1, 5).expand(a, 5).contiguous().realize()
    expected = x[:, 2:7].realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_shrink(self):
    x = Tensor.arange(30).reshape(3, 10).contiguous()
    a = UOp.range(3, -1)
    out = x[a].shrink(((1, 6),))
    out = out.reshape(1, 5).expand(a, 5).contiguous().realize()
    expected = x[:, 1:6].realize()
    self.assertTrue((out == expected).all().item())

  # Binary operations
  def test_vmap_mul(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a] * 3.0
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = (x * 3.0).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_sub(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a] - 5.0
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = (x - 5.0).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_div(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous() + 1.0
    a = UOp.range(3, -1)
    out = x[a] / 2.0
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = (x / 2.0).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_pow(self):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).contiguous()
    a = UOp.range(2, -1)
    out = x[a] ** 2.0
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = (x ** 2.0).realize()
    self.assertTrue((out == expected).all().item())

  # Unary operations
  def test_vmap_neg(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = -x[a]
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = (-x).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_abs(self):
    x = Tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]).contiguous()
    a = UOp.range(2, -1)
    out = x[a].abs()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.abs().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_relu(self):
    x = Tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]).contiguous()
    a = UOp.range(2, -1)
    out = x[a].relu()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.relu().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_sigmoid(self):
    x = Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).contiguous()
    a = UOp.range(2, -1)
    out = x[a].sigmoid()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.sigmoid().realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_tanh(self):
    x = Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]).contiguous()
    a = UOp.range(2, -1)
    out = x[a].tanh()
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = x.tanh().realize()
    self.assertTrue((out == expected).all().item())

  # Reduce operations
  def test_vmap_max(self):
    x = Tensor.arange(24).reshape(3, 2, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a].max(axis=1)
    out = out.reshape(1, 2).expand(a, 2).contiguous().realize()
    expected = x.max(axis=2).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_min(self):
    x = Tensor.arange(24).reshape(3, 2, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a].min(axis=1)
    out = out.reshape(1, 2).expand(a, 2).contiguous().realize()
    expected = x.min(axis=2).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_mean(self):
    x = Tensor.arange(24).reshape(3, 2, 4).contiguous().float()
    a = UOp.range(3, -1)
    out = x[a].mean(axis=1)
    out = out.reshape(1, 2).expand(a, 2).contiguous().realize()
    expected = x.mean(axis=2).realize()
    self.assertTrue((out == expected).all().item())

  # Comparison operations
  def test_vmap_gt(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a] > 5
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = (x > 5).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_lt(self):
    x = Tensor.arange(12).reshape(3, 4).contiguous()
    a = UOp.range(3, -1)
    out = x[a] < 8
    out = out.reshape(1, 4).expand(a, 4).contiguous().realize()
    expected = (x < 8).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_eq(self):
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).contiguous()
    a = UOp.range(3, -1)
    out = x[a] == 5
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = (x == 5).realize()
    self.assertTrue((out == expected).all().item())

  def test_vmap_where(self):
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).contiguous()
    y = Tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]).contiguous()
    a = UOp.range(3, -1)
    out = (x[a] > 5).where(x[a], y[a])
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    expected = (x > 5).where(x, y).realize()
    self.assertTrue((out == expected).all().item())

if __name__ == '__main__':
  unittest.main()