"""
vmap: Vectorizing map transform for tinygrad.

This module implements a JAX-like vmap that vectorizes a function over a batch dimension.
The approach is to directly manipulate Tensor operations rather than UOp graph rewriting.
"""
from __future__ import annotations
from typing import Callable
import functools
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp, Ops, GroupOp
from tinygrad.dtype import dtypes

# Type alias for axis specification
AxisSpec = int | None | tuple[int | None, ...]

def _normalize_in_axes(in_axes: AxisSpec, num_args: int) -> tuple[int | None, ...]:
  """Normalize in_axes to a tuple with one entry per argument."""
  if isinstance(in_axes, int) or in_axes is None:
    return tuple(in_axes for _ in range(num_args))
  if isinstance(in_axes, tuple):
    if len(in_axes) != num_args:
      raise ValueError(f"in_axes tuple length {len(in_axes)} doesn't match number of arguments {num_args}")
    return in_axes
  raise TypeError(f"in_axes must be int, None, or tuple, got {type(in_axes)}")

def _validate_axis(axis: int | None, ndim: int, arg_idx: int) -> int | None:
  """Validate and normalize a single axis value."""
  if axis is None:
    return None
  if not isinstance(axis, int):
    raise TypeError(f"axis must be int or None, got {type(axis)} for argument {arg_idx}")
  if axis < -ndim or axis >= ndim:
    raise ValueError(f"axis {axis} out of bounds for argument {arg_idx} with {ndim} dimensions")
  return axis if axis >= 0 else ndim + axis

def _get_batch_size(args: tuple[Tensor, ...], in_axes: tuple[int | None, ...]) -> int | None:
  """Extract batch size from arguments based on in_axes."""
  batch_sizes = []
  for arg, axis in zip(args, in_axes):
    if axis is not None:
      batch_sizes.append(arg.shape[axis])
  if not batch_sizes:
    return None
  if not all(s == batch_sizes[0] for s in batch_sizes):
    raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
  return batch_sizes[0]

def _move_axis_to_front(t: Tensor, axis: int) -> Tensor:
  """Move the specified axis to position 0."""
  if axis == 0:
    return t
  perm = [axis] + [i for i in range(t.ndim) if i != axis]
  return t.permute(*perm)


class _VmapTracer:
  """
  Tracer that wraps Tensors to track batch dimensions through computation.

  This is the core mechanism: we wrap inputs as tracers, execute the function,
  and the tracer records what operations were performed. Then we replay those
  operations on the actual batched inputs.
  """
  _ops_log: list = []  # Class variable to log operations during tracing

  def __init__(self, shape: tuple[int, ...], dtype, device: str, batch_axis: int | None = 0):
    self.shape = shape  # Unbatched shape (what the traced function sees)
    self.dtype = dtype
    self.device = device
    self.batch_axis = batch_axis  # 0 for batched, None for broadcast
    self._id = id(self)

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def __repr__(self):
    return f"VmapTracer(shape={self.shape}, batch_axis={self.batch_axis})"


def _trace_function(fn: Callable, placeholders: list[Tensor]) -> tuple[Tensor, dict[int, Tensor]]:
  """
  Trace a function by calling it with placeholder tensors.
  Returns the output tensor and a mapping from placeholder ids to the placeholders.
  """
  placeholder_map = {id(p.uop): p for p in placeholders}
  output = fn(*placeholders)
  return output, placeholder_map


def _apply_batched(fn: Callable, batched_args: list[Tensor], unbatched_shapes: list[tuple[int, ...]], batch_size: int) -> Tensor:
  """
  Apply a function with batched arguments.

  Strategy:
  1. Create placeholder tensors with unbatched shapes
  2. Call the function to get the output expression
  3. Substitute placeholder UOps with batched input UOps
  4. Adjust movement/reduce ops in the result for batch dimension
  """
  # Create placeholders with unbatched shapes
  placeholders = []
  for shape, batched_arg in zip(unbatched_shapes, batched_args):
    placeholder = Tensor.empty(*shape, dtype=batched_arg.dtype, device=batched_arg.device)
    placeholders.append(placeholder)

  # Trace the function
  output = fn(*placeholders)

  # Build the transformation map: for each node in the traced graph,
  # we need to determine the corresponding batched node

  # Create a substitution from placeholder base UOps to batched base UOps
  # The key insight: we walk the output graph bottom-up, transforming each node

  return _transform_traced_output(output, placeholders, batched_args, batch_size)


def _transform_traced_output(output: Tensor, placeholders: list[Tensor], batched_args: list[Tensor], batch_size: int) -> Tensor:
  """
  Transform the traced output to work with batched inputs.

  We use a memoized recursive transformation that:
  1. Maps placeholder UOps to batched arg UOps
  2. Transforms each operation to handle batch dimension
  """
  # Map from original UOp to (transformed Tensor, has_batch)
  cache: dict[UOp, tuple[Tensor, bool]] = {}

  # Initialize with input mappings
  for placeholder, batched_arg in zip(placeholders, batched_args):
    cache[placeholder.uop] = (batched_arg, True)

  def transform(uop: UOp) -> tuple[Tensor, bool]:
    if uop in cache:
      return cache[uop]

    # Transform sources first
    src_results = []
    for s in uop.src:
      if s.op in {Ops.UNIQUE, Ops.DEVICE}:
        continue  # Skip infrastructure nodes
      if s._shape is not None:  # Only process shaped nodes
        src_results.append((s, *transform(s)))

    # If no shaped sources, this might be a leaf (const, buffer, etc.)
    if not src_results:
      return _handle_leaf(uop, batch_size, cache)

    # Get the primary source tensor
    src_uop, src_tensor, src_has_batch = src_results[0]

    # Handle different op types
    result: Tensor
    has_batch: bool

    if uop.op in GroupOp.ALU:
      result, has_batch = _transform_alu(uop, src_results, batch_size)
    elif uop.op is Ops.REDUCE_AXIS:
      result, has_batch = _transform_reduce(uop, src_tensor, src_has_batch, batch_size)
    elif uop.op in GroupOp.Movement:
      result, has_batch = _transform_movement(uop, src_tensor, src_has_batch, batch_size)
    elif uop.op in {Ops.CAST, Ops.BITCAST}:
      result = src_tensor.cast(uop.dtype)
      has_batch = src_has_batch
    elif uop.op in {Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD}:
      result = src_tensor.contiguous()
      has_batch = src_has_batch
    elif uop.op is Ops.DETACH:
      result = src_tensor.detach()
      has_batch = src_has_batch
    else:
      raise NotImplementedError(f"vmap does not support op {uop.op}")

    cache[uop] = (result, has_batch)
    return result, has_batch

  result, _ = transform(output.uop)
  return result


def _handle_leaf(uop: UOp, batch_size: int, cache: dict) -> tuple[Tensor, bool]:
  """Handle leaf nodes (constants, etc.)."""
  if uop.op is Ops.CONST:
    # Constants should be broadcast to batch size if used in batched context
    # For now, just return the constant - broadcasting will happen in ALU ops
    val = uop.arg
    device = uop.device if hasattr(uop, 'device') and uop.device else None
    t = Tensor.full((), val, dtype=uop.dtype, device=device)
    cache[uop] = (t, False)
    return t, False
  elif uop.op is Ops.BUFFER:
    # This shouldn't be reached for inputs (they should be in cache)
    # but might be for intermediate buffers
    raise RuntimeError(f"Unexpected BUFFER UOp not in input map")
  else:
    # Skip infrastructure nodes
    raise RuntimeError(f"Unexpected leaf op {uop.op}")


def _transform_alu(uop: UOp, src_results: list[tuple[UOp, Tensor, bool]], batch_size: int) -> tuple[Tensor, bool]:
  """Transform ALU operations."""
  # Check if any source is batched
  any_batched = any(has_batch for _, _, has_batch in src_results)

  # Get tensors and align batch dimensions if needed
  tensors = []
  for _, t, has_batch in src_results:
    if any_batched and not has_batch:
      # Broadcast to batch dimension
      t = t.unsqueeze(0).expand(batch_size, *t.shape)
    tensors.append(t)

  # Apply the operation
  if uop.op is Ops.ADD:
    return tensors[0] + tensors[1], any_batched
  elif uop.op is Ops.MUL:
    return tensors[0] * tensors[1], any_batched
  elif uop.op is Ops.SUB:
    return tensors[0] - tensors[1], any_batched
  elif uop.op is Ops.FDIV:
    return tensors[0] / tensors[1], any_batched
  elif uop.op is Ops.NEG:
    return -tensors[0], any_batched
  elif uop.op is Ops.EXP2:
    return tensors[0].exp2(), any_batched
  elif uop.op is Ops.LOG2:
    return tensors[0].log2(), any_batched
  elif uop.op is Ops.SIN:
    return tensors[0].sin(), any_batched
  elif uop.op is Ops.SQRT:
    return tensors[0].sqrt(), any_batched
  elif uop.op is Ops.RECIPROCAL:
    return tensors[0].reciprocal(), any_batched
  elif uop.op is Ops.MAX:
    return tensors[0].maximum(tensors[1]), any_batched
  elif uop.op is Ops.WHERE:
    return tensors[0].where(tensors[1], tensors[2]), any_batched
  elif uop.op is Ops.CMPLT:
    return tensors[0] < tensors[1], any_batched
  elif uop.op is Ops.CMPEQ:
    return tensors[0] == tensors[1], any_batched
  elif uop.op is Ops.CMPNE:
    return tensors[0] != tensors[1], any_batched
  else:
    raise NotImplementedError(f"vmap ALU op {uop.op}")


def _transform_reduce(uop: UOp, src: Tensor, src_has_batch: bool, batch_size: int) -> tuple[Tensor, bool]:
  """Transform reduce operations."""
  if not src_has_batch:
    # Source not batched, output not batched
    reduce_op, axes = uop.arg
    if reduce_op is Ops.ADD:
      return src.sum(axis=axes), False
    elif reduce_op is Ops.MAX:
      return src.max(axis=axes), False
    elif reduce_op is Ops.MUL:
      return src.prod(axis=axes), False
    raise NotImplementedError(f"vmap reduce op {reduce_op}")

  # Source is batched - shift axes by 1
  reduce_op, axes = uop.arg
  shifted_axes = tuple(ax + 1 for ax in axes)

  if reduce_op is Ops.ADD:
    return src.sum(axis=shifted_axes), True
  elif reduce_op is Ops.MAX:
    return src.max(axis=shifted_axes), True
  elif reduce_op is Ops.MUL:
    return src.prod(axis=shifted_axes), True
  raise NotImplementedError(f"vmap reduce op {reduce_op}")


def _transform_movement(uop: UOp, src: Tensor, src_has_batch: bool, batch_size: int) -> tuple[Tensor, bool]:
  """Transform movement operations."""
  if not src_has_batch:
    # Source not batched - just apply normally
    if uop.op is Ops.RESHAPE:
      return src.reshape(*uop.marg), False
    elif uop.op is Ops.PERMUTE:
      return src.permute(*uop.marg), False
    elif uop.op is Ops.EXPAND:
      return src.expand(*uop.marg), False
    elif uop.op is Ops.PAD:
      return src.pad(uop.marg), False
    elif uop.op is Ops.SHRINK:
      return src.shrink(uop.marg), False
    elif uop.op is Ops.FLIP:
      return src.flip(uop.marg), False
    raise NotImplementedError(f"vmap movement op {uop.op}")

  # Source is batched - adjust for batch dimension at position 0
  if uop.op is Ops.RESHAPE:
    # Prepend batch_size to target shape
    target_shape = (batch_size,) + tuple(uop.marg)
    return src.reshape(*target_shape), True
  elif uop.op is Ops.PERMUTE:
    # Keep batch at 0, shift other axes
    perm = (0,) + tuple(p + 1 for p in uop.marg)
    return src.permute(*perm), True
  elif uop.op is Ops.EXPAND:
    # Prepend batch_size
    target_shape = (batch_size,) + tuple(uop.marg)
    return src.expand(*target_shape), True
  elif uop.op is Ops.PAD:
    # Prepend (0, 0) for batch dimension
    padding = ((0, 0),) + tuple(uop.marg)
    return src.pad(padding), True
  elif uop.op is Ops.SHRINK:
    # Prepend (0, batch_size) for batch dimension
    shrink = ((0, batch_size),) + tuple(uop.marg)
    return src.shrink(shrink), True
  elif uop.op is Ops.FLIP:
    # Original flip axes need to be shifted
    # uop.marg is a tuple of bools for each axis
    flip_mask = (False,) + tuple(uop.marg)
    axes_to_flip = tuple(i for i, f in enumerate(flip_mask) if f)
    return src.flip(axes_to_flip), True

  raise NotImplementedError(f"vmap movement op {uop.op}")


# ===== Main vmap Function =====

def vmap(fn: Callable[..., Tensor], in_axes: AxisSpec = 0) -> Callable[..., Tensor]:
  """
  Vectorizing map. Creates a function which maps fn over argument axes.

  Args:
      fn: Function to be mapped over additional axes. Should take Tensor inputs
          and return a Tensor output.
      in_axes: An integer, None, or tuple specifying which input array axes to map over.
               - int: map over that axis for all inputs
               - None: broadcast (don't map) all inputs
               - tuple: per-input specification (int or None for each arg)

  Returns:
      A batched version of fn that processes inputs with an additional batch dimension.

  Example:
      >>> def add(x, y): return x + y
      >>> batched_add = vmap(add)
      >>> x = Tensor.ones(10, 3)  # 10 batches of 3-vectors
      >>> y = Tensor.ones(10, 3)
      >>> result = batched_add(x, y)  # shape: (10, 3)
  """
  @functools.wraps(fn)
  def vmapped_fn(*args: Tensor) -> Tensor:
    if not all(isinstance(arg, Tensor) for arg in args):
      raise TypeError("All arguments must be Tensors")

    axes = _normalize_in_axes(in_axes, len(args))
    validated_axes = tuple(
      _validate_axis(ax, arg.ndim, i)
      for i, (ax, arg) in enumerate(zip(axes, args))
    )

    batch_size = _get_batch_size(args, validated_axes)
    if batch_size is None:
      raise ValueError("At least one input must have a mapped axis (in_axes != None for all)")

    # Prepare batched inputs (move batch axis to front)
    batched_args = []
    unbatched_shapes = []
    for arg, axis in zip(args, validated_axes):
      if axis is not None:
        batched_args.append(_move_axis_to_front(arg, axis))
        # Unbatched shape: remove the batch dimension
        shape = list(arg.shape)
        shape.pop(axis)
        unbatched_shapes.append(tuple(shape))
      else:
        # Not batched - expand to have batch dimension for consistency
        batched_args.append(arg.unsqueeze(0).expand(batch_size, *arg.shape))
        unbatched_shapes.append(arg.shape)

    return _apply_batched(fn, batched_args, unbatched_shapes, batch_size)

  return vmapped_fn
