# Vmap Scheduling Bug: Fixed

## Summary

When vmapping a function that stacks 3+ computed tensors and extracts multiple scalar indices via `Tensor.stack(t[i], t[j], ...)`, the scheduler previously failed with:

```
RuntimeError: input to kernel must be AFTER or BUFFER, not Ops.INDEX
```

**Status: FIXED** - vmap ranges now stay as `AxisType.LOOP` (inside the kernel), preserving fusion semantics.

## Root Cause

The vmap range created by `VMAPIN` was not being closed by the `END` node during kernel splitting.

When `VMAPOUT` removes the vmap range from its output ranges, that range doesn't propagate forward to the BUFFERIZE/STORE indexing. However, the vmap range is still present in the `.ranges` property of the store's value (because the computation inside vmap uses that range for indexing).

The `split_store` function in `tinygrad/schedule/rangeify.py` refuses to convert nodes to kernels if they have any non-OUTER ranges still open:

```python
def split_store(ctx:list[UOp], x:UOp) -> UOp|None:
    if len([r for r in x.ranges if r.arg[-1] != AxisType.OUTER]): return None
```

## The Fix

**Two changes were made:**

### 1. Keep vmap ranges as LOOP (not OUTER)

**Location:** `tinygrad/schedule/indexing.py`, `run_rangeify()` function, VMAPIN handling

The previous fix converted vmap's LOOP ranges to OUTER ranges, which defeated the purpose of vmap (fusing batch dimension into the kernel). Now we keep them as LOOP:

```python
if x.op == Ops.VMAPIN:
  # insert back outerworld ranges (vmap ranges stay as-is, they'll be closed at global store boundaries)
  out_rngs_iter = iter(out_rngs)
  rngs = tuple(a if a.op == Ops.RANGE else next(out_rngs_iter) for a in x.src[1:])
```

### 2. Close all dependent non-OUTER ranges at global store boundaries

**Location:** `tinygrad/schedule/rangeify.py`, `bufferize_to_store()` function

For global stores (kernel boundaries), we now also close any non-OUTER ranges that the store's value depends on. This captures vmap ranges that were "hidden" by VMAPOUT:

```python
def bufferize_to_store(ctx:itertools.count, x:UOp, idx:UOp, allow_locals=True):
  size = prod(x.shape)
  rngs = sorted(idx.ranges, key=lambda x: x.arg)

  # for global stores, close all non-OUTER ranges the store value depends on (includes vmap ranges hidden by VMAPOUT)
  if x.arg.addrspace == AddrSpace.GLOBAL:
    value_ranges = [r for r in x.src[0].ranges if r.op is Ops.RANGE and r.arg[-1] != AxisType.OUTER]
    for r in value_ranges:
      if r not in rngs: rngs.append(r)
    rngs = sorted(rngs, key=lambda x: x.arg)
  ...
```

## Why This Works

1. Vmap ranges stay as `AxisType.LOOP` - semantically "loops inside the kernel"
2. When creating the kernel-boundary END at global stores, we now close all dependent non-OUTER ranges
3. This makes `split_store` happy (no open non-OUTER ranges) while keeping the batch dimension fused
4. `AxisType.OUTER` is still reserved for actual outer loops that should execute the kernel multiple times

## Semantic Difference: LOOP vs OUTER vmap

| AxisType | Behavior | Use Case |
|----------|----------|----------|
| `AxisType.LOOP` | Batch dim fused into kernel | Default vmap - efficient for small ops |
| `AxisType.OUTER` | Kernel executed N times | Large ops where fusion isn't beneficial |

## Test Commands

```bash
# Run all vmap tests
.venv/bin/python -m pytest test/test_vmap.py test/test_vmap_red.py test/test_outerworld.py -v

# Run minimal reproduction
.venv/bin/python -c "
from tinygrad import Tensor, UOp, dtypes
from tinygrad.engine.schedule import complete_create_schedule_with_vars
from tinygrad.uop.ops import AxisType

def vmap(f, in_axis=0, axis_id=-1, axis_type=AxisType.LOOP):
    def wrapper(x):
        batch_size = x.shape[in_axis]
        r = UOp.range(batch_size, axis_id, axis_type)
        varg = tuple(r if i == in_axis else UOp.const(dtypes.index, 0) for i in range(x.ndim))
        inner_result = f(x.vmapin(varg))
        out_varg = (r,) + tuple(UOp.const(dtypes.index, 0) for _ in range(inner_result.ndim))
        return inner_result.vmapout(out_varg)
    return wrapper

def fn(x):
    v0, v1, v2 = Tensor([1.0, 0.0, 0.0]), Tensor([0.0, 1.0, 0.0]), Tensor([0.0, 0.0, 1.0])
    flat = Tensor.stack(x * v0, x * v1, x * v2).flatten()
    return Tensor.stack(flat[0], flat[1])

x_batch = Tensor.empty((10, 3))
result = vmap(fn, in_axis=0)(x_batch)
sink = UOp.sink(result.uop)
complete_create_schedule_with_vars(sink)
print('SUCCESS!')
"
```
