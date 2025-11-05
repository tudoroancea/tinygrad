import unittest

import numpy as np

from tinygrad import Tensor, UOp, dtypes


class TestVmap(unittest.TestCase):
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

  def test_vmap(self):
    x = UOp.new_buffer("CPU", 18, dtypes.float, 0).reshape((3,6))
    varg = (UOp.range(3,-1), UOp.const(dtypes.index, 0))
    a = UOp.new_buffer("CPU", 6, dtypes.float, 1)
    c0 = x.vmapin(varg)
    c1 = c0 * a
    c2 = c1.vmapout(varg)
    ast = c2.contiguous()
    Tensor(ast, "CPU").schedule()

  def test_vmap_dim1(self):
    x = UOp.new_buffer("CPU", 6, dtypes.float, 0).reshape((1,6))
    varg = (UOp.range(1,-1), UOp.const(dtypes.index, 0))
    a = UOp.new_buffer("CPU", 6, dtypes.float, 1)
    c0 = x.vmapin(varg)
    c1 = c0 * a
    c2 = c1.vmapout(varg)
    ast = c2.contiguous()
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

class TestAfterVmap(unittest.TestCase):
  n,m = 3,6
  def fn(self, x:Tensor)->Tensor:
    return Tensor(np.arange(self.m, dtype=np.float32)) * x
  def vfn(self, x:Tensor)->Tensor:
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
