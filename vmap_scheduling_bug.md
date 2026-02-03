# Vmap Scheduling Bug: Unclosed Range Prevents Kernel Splitting

## Summary

When vmapping a function that stacks 3+ computed tensors and extracts multiple scalar indices via `Tensor.stack(t[i], t[j], ...)`, the scheduler fails with:

```
RuntimeError: input to kernel must be AFTER or BUFFER, not Ops.INDEX
```

## Root Cause

The vmap range `(-1, AxisType.LOOP)` remains open inside an `Ops.END` node. The `split_store` function in `tinygrad/schedule/rangeify.py` refuses to convert such nodes to kernels:

```python
def split_store(ctx:list[UOp], x:UOp) -> UOp|None:
    # if we have any outer ranges open here, we don't split
    if len([r for r in x.ranges if r.arg[-1] != AxisType.OUTER]): return None
    ...
```

This leaves `STORE`/`END` nodes unconverted, which later causes `create_schedule` to fail when it expects all kernel inputs to be `AFTER`, `BUFFER`, `MSELECT`, `MSTACK`, or `BIND`.

## The Fix

**Location:** `tinygrad/schedule/indexing.py`, lines 267-277, in `run_rangeify()` function

**Change:** Convert vmap ranges from `AxisType.LOOP` to `AxisType.OUTER` during VMAPIN handling:

```python
if x.op == Ops.VMAPIN:
  out_rngs_iter = iter(out_rngs)
  def _to_outer(a):
    if a.op != Ops.RANGE: return next(out_rngs_iter)
    if a.arg[-1] == AxisType.LOOP: return a.replace(arg=a.arg[:-1] + (AxisType.OUTER,))
    return a
  rngs = tuple(_to_outer(a) for a in x.src[1:])
```

**Why this works:**

1. `AxisType.OUTER` ranges are semantically "loops outside the kernel" - they represent iteration over batch dimensions
2. `split_store` explicitly allows open OUTER ranges: `if r.arg[-1] != AxisType.OUTER`
3. In `create_schedule`, OUTER ranges become RANGE/END pairs that execute the kernel multiple times
4. This correctly models vmap semantics: the batch dimension is an outer loop around kernel execution

**Why using `AxisType.OUTER` directly in vmap didn't work:**

Even with OUTER vmap ranges, there can be other LOOP ranges (from output indexing) that remain open at STORE/END nodes. The fix must happen in the rangeify pipeline where ranges are propagated, not just at vmap wrapper construction.

## Failure Chain (Before Fix)

1. **Rangeify** creates an `END` node with vmap range `(-1, AxisType.LOOP)` still open
2. **`split_store`** returns `None` (silently refuses to convert to kernel)
3. **`split_kernels`** pattern matcher doesn't transform the `END` node
4. **`create_schedule`** iterates kernel sources, finds `Ops.INDEX`/`Ops.ADD` instead of valid buffer ops
5. **Error thrown**: `"input to kernel must be AFTER or BUFFER, not Ops.INDEX"`

## Minimal Reproduction

```python
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
    flat = Tensor.stack(x * v0, x * v1, x * v2).flatten()  # 3 stacked tensors
    return Tensor.stack(flat[0], flat[1])  # Multi-index extraction

x_batch = Tensor.empty((10, 3))
result = vmap(fn, in_axis=0)(x_batch)
sink = UOp.sink(result.uop)
complete_create_schedule_with_vars(sink)  # FAILS without fix, PASSES with fix
```

## Key Observations

### What triggers the bug
- **3+ stacked tensors** (2 works, 3+ fails)
- **Multi-index extraction** via `Tensor.stack(t[i], t[j], ...)`
- **REDUCE operations lower the threshold** (fails with 2 stacked + reduce)

### What works (before fix)
- Single index extraction: `t[0]`
- Shrink-based extraction: `t[:3]`
- 2 stacked tensors with multi-index
- 3+ stacked without multi-index

### The problematic END node structure (before fix)

After `bufferize_to_store`, before `split_kernels`:

```
Ops.END: 1 total ranges, 1 non-OUTER
  (-1, AxisType.LOOP)   <-- vmap range still open!
```

The END contains a STORE with 2 ranges:
```
Ops.STORE: 2 total ranges, 2 non-OUTER
  (1, AxisType.LOOP)    <-- output index range
  (-1, AxisType.LOOP)   <-- vmap range
```

### After fix

```
Ops.END: 1 total ranges, 0 non-OUTER
  (-1, AxisType.OUTER)   <-- vmap range is OUTER, allowed to remain open
```

## Why `limit_bufs` Doesn't Help

Initial hypothesis was that `limit_bufs` silently fails. This is **incorrect**.

- `limit_bufs` only matches `GroupOp.Binary` and `GroupOp.Ternary` ops
- It correctly inserts `BUFFERIZE` nodes when buffer count exceeds limit
- The vmap range issue exists **regardless** of `limit_bufs` activity
- Even with `MAX_KERNEL_BUFFERS=3` forcing bufferization, the same END survives

## Relevant Code Locations

| File | Function/Pattern | Role |
|------|-----------------|------|
| `schedule/indexing.py:267` | `run_rangeify()` VMAPIN handling | **FIX LOCATION** - converts LOOP→OUTER |
| `schedule/rangeify.py:487` | `split_store()` | Guards against non-OUTER open ranges |
| `schedule/rangeify.py:516` | `split_kernels` | Pattern matcher that converts STORE/END → KERNEL |
| `engine/schedule.py:38` | `create_schedule()` | Throws error when finding invalid kernel inputs |

## Test File

See `test/test_vmap_red.py` (16 tests):

**Scheduling tests (previously failing, now pass):**
- `TestVmapMultiConsumerIndexing::test_3_stacked_multi_index`
- `TestVmapMultiConsumerIndexing::test_3_stacked_2_index`
- `TestVmapReduceMultiIndex::test_reduce_expand_multi_index`
- `TestVmapReduceMultiIndex::test_2_reduces_stacked_multi_index`
- `TestSpjacobianPattern::test_spjacobian_unroll_pattern`
- `TestSpjacobianPattern::test_spjacobian_pattern_no_reduce`
- `TestMinimalReproduction::test_minimal_3_stack_2_index`

**Correctness tests (verify outputs with uniform input):**
- `TestVmapCorrectness::test_vmap_elementwise_ones`
- `TestVmapCorrectness::test_vmap_sum_ones`
- `TestVmapCorrectness::test_vmap_multi_index_ones`

## Debug Commands

```bash
# See rangeify debug output
DEBUG_RANGEIFY=1 uv run python -c "..."

# Visualize the graph (starts web server)
VIZ=1 uv run python -c "..."

# Run all vmap tests
uv run pytest test/test_vmap.py test/test_vmap_red.py test/test_outerworld.py -v
```
