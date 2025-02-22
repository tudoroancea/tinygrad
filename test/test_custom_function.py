# this is an example of how you can write terrible DSP compute breaking ops like warpPerspective
# here we use a CUSTOM op to write atan2

import unittest
import numpy as np
from typing import Optional, Tuple
from tinygrad.helpers import prod

# *** first, we implement the atan2 op at the lowest level ***
# `atan2_gpu` for GPUBuffers and `atan2_cpu` for CPUBuffers

from tinygrad.ops import ASTRunner, CompiledBuffer
from tinygrad.runtime.ops_cpu import CPUBuffer

# we don't always have GPU support, so the type signature is the abstract CompiledBuffer instead of GPUBuffer
def atan2_gpu(a:CompiledBuffer, b:CompiledBuffer) -> CompiledBuffer:
  from tinygrad.runtime.ops_gpu import GPUBuffer
  assert type(a) == GPUBuffer and type(b) == GPUBuffer, "gpu function requires GPUBuffers"
  ret = GPUBuffer(a.shape)
  ASTRunner("atan2", """
    __kernel void atan2(global float *c, global float *a, global float *b) {
      int idx = get_global_id(0);
      c[idx] = atan2(a[idx], b[idx]);
    }""", global_size=[prod(ret.shape)]).build(GPUBuffer.runtime_type).exec([ret, a.contiguous(), b.contiguous()])
  return ret

def atan2_cpu(a:CPUBuffer, b:CPUBuffer) -> CPUBuffer:
  return CPUBuffer(np.arctan2(a._buf, b._buf))

# *** second, we write the ATan2 mlop ***
# NOTE: The derivative of atan2 doesn't need a custom op! https://www.liquisearch.com/atan2/derivative
# In general, it is also optional to write a backward function, just your backward pass won't work without it

from tinygrad.ops import ASTRunner, LazyOp, LoadOps, BinaryOps, UnaryOps
from tinygrad.lazy import LazyBuffer
from tinygrad.tensor import Function

class ATan2(Function):
  def forward(self, a:LazyBuffer, b:LazyBuffer) -> LazyBuffer:
    assert prod(a.shape) == prod(b.shape) and a.device == b.device, "shape or device mismatch"
    self.a, self.b = a, b
    ast = LazyOp(LoadOps.CUSTOM, (a, b), {"GPU": atan2_gpu, "CPU": atan2_cpu}[a.device])
    return LazyBuffer(a.device, a.shape, LoadOps, ast)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    denom = (self.a.binary_op(BinaryOps.MUL, self.a)).binary_op(BinaryOps.ADD, self.b.binary_op(BinaryOps.MUL, self.b))
    return grad_output.binary_op(BinaryOps.MUL, self.b.binary_op(BinaryOps.DIV, denom)) if self.needs_input_grad[0] else None, \
           grad_output.binary_op(BinaryOps.MUL, self.a.unary_op(UnaryOps.NEG).binary_op(BinaryOps.DIV, denom)) if self.needs_input_grad[1] else None

# *** third, we use our lovely new mlop in some tests ***

from tinygrad.tensor import Tensor, Device

@unittest.skipUnless(Device.DEFAULT in ["CPU", "GPU"], "atan2 is only implemented for CPU and GPU")
class TestCustomFunction(unittest.TestCase):
  def test_atan2_forward(self):
    # create some random Tensors, permute them just because we can
    a = Tensor.randn(4,4,requires_grad=True).permute(1,0)
    b = Tensor.randn(4,4,requires_grad=True).permute(1,0)

    # run the forward pass. note: up until the .numpy(), it's all lazy
    c = ATan2.apply(a, b)
    print(c.numpy())

    # check the forward pass (in numpy)
    np.testing.assert_allclose(c.numpy(), np.arctan2(a.numpy(), b.numpy()), atol=1e-5)

  # fun fact, this never actually calls forward, so it works in all the backends
  def test_atan2_backward(self):
    # have to go forward before we can go backward
    a = Tensor.randn(4,4,requires_grad=True).permute(1,0)
    b = Tensor.randn(4,4,requires_grad=True).permute(1,0)
    c = ATan2.apply(a, b)

    # run the backward pass
    c.mean().backward()
    assert a.grad is not None and b.grad is not None, "tinygrad didn't compute gradients"
    print(a.grad.numpy())
    print(b.grad.numpy())

    # check the backward pass (in torch)
    import torch
    ta, tb = torch.tensor(a.numpy(), requires_grad=True), torch.tensor(b.numpy(), requires_grad=True)
    tc = torch.atan2(ta, tb)
    tc.mean().backward()
    assert ta.grad is not None and tb.grad is not None, "torch didn't compute gradients"
    np.testing.assert_allclose(a.grad.numpy(), ta.grad.numpy(), atol=1e-5)
    np.testing.assert_allclose(b.grad.numpy(), tb.grad.numpy(), atol=1e-5)

  def test_atan2_jit(self):
    # custom ops even work in the JIT!
    from tinygrad.jit import TinyJit

    @TinyJit
    def jitted_atan2(a:Tensor, b:Tensor) -> Tensor:
      return ATan2.apply(a, b).realize()

    for _ in range(5):
      a = Tensor.randn(4,4,requires_grad=True).permute(1,0)
      b = Tensor.randn(4,4,requires_grad=True).permute(1,0)
      c = jitted_atan2(a, b)
      np.testing.assert_allclose(c.numpy(), np.arctan2(a.numpy(), b.numpy()), atol=1e-5)

if __name__ == "__main__":
  unittest.main()
