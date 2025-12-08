import numpy as np
from tinygrad import Tensor, Context
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.engine.realize import get_program
from tinygrad.uop.ops import Ops


def test_render_simple():
    """Test rendering a simple kernel with register reuse."""

    # Create tensors and a simple computation
    # Use rand to avoid constant folding
    a = Tensor.rand(100).realize()
    b = Tensor.rand(100).realize()
    c = a + b

    # Get the schedule
    sched = Tensor.schedule(c)

    # Find the compute kernel schedule item
    for si in sched:
        if si.ast.op is Ops.SINK:
            renderer = ClangRenderer()
            print("=== Standard Renderer ===")
            print(get_program(si.ast, renderer).src)
            print("\n=== Register Reuse Render ===")
            with Context(REUSE_REGISTERS=True):
                print(get_program(si.ast, renderer).src)

def test_render_complex():
    """Test rendering a simple kernel with register reuse."""

    # Create tensors and a simple computation
    # Use rand to avoid constant folding
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = (a + b) * (a-b) + b

    # Get the schedule
    sched = Tensor.schedule(c)

    # Find the compute kernel schedule item
    for si in sched:
        if si.ast.op is Ops.SINK:
            renderer = ClangRenderer()
            print("=== Standard Renderer ===")
            print(get_program(si.ast, renderer).src)
            print("\n=== Register Reuse Render ===")
            with Context(REUSE_REGISTERS=True):
                print(get_program(si.ast, renderer).src)

def test_realize_simple():
    """Test realizing a simple kernel with register reuse."""

    # Create tensors and a simple computation
    # Use rand to avoid constant folding
    a = Tensor.rand(100).realize()
    b = Tensor.rand(100).realize()
    c = a + b

    np.testing.assert_array_almost_equal(c.numpy(), a.numpy() + b.numpy())

def bicycle_cont(x: Tensor, u: Tensor) -> Tensor:
  theta, v = x[2], x[3]
  a, delta = u[0], u[1]
  return Tensor.stack(v * Tensor.cos(theta), v * Tensor.sin(theta), v * Tensor.tan(delta), a)


def bicycle_disc_rk4(x: Tensor, u: Tensor) -> Tensor:
  dt = 0.01
  k1 = bicycle_cont(x, u)
  k2 = bicycle_cont(x + dt * k1 / 2, u)
  k3 = bicycle_cont(x + dt * k2 / 2, u)
  k4 = bicycle_cont(x + dt * k3, u)
  return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def test_render_bicycle():
    """Test rendering a simple kernel with register reuse."""

    # Create tensors and a simple computation
    # Use rand to avoid constant folding
    x = Tensor.rand(4).realize()
    u = Tensor.rand(2).realize()
    xnext = bicycle_disc_rk4(x,u)

    # Get the schedule
    sched = Tensor.schedule(xnext)

    # Find the compute kernel schedule item
    for si in sched:
        if si.ast.op is Ops.SINK:
            renderer = ClangRenderer()
            print("=== Standard Renderer ===")
            print(get_program(si.ast, renderer).src)
            print("\n=== Register Reuse Render ===")
            with Context(REUSE_REGISTERS=True):
                print(get_program(si.ast, renderer).src)

    print(xnext.tolist())
