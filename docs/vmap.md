# tinygrad vmap: Implementation with VMAPIN / VMAPOUT

## Overview

tinygrad implements `vmap` (vectorized map) as a graph-level transformation using two special UOps: **`VMAPIN`** and **`VMAPOUT`**. These bracket a subgraph that logically runs per-batch-element, and the scheduler fuses them into a single kernel with a loop over the batch dimension. There is no Python-level loop—`vmap` is a purely structural annotation on the compute graph.

This document covers:
1. Why new ops were needed
2. The UOp semantics and shape rules
3. How the scheduler (rangeify) processes vmap nodes
4. The intermediate-realization bug and its fix
5. Supported constructs

---

## 1. Why New Ops? (Motivation)

The existing vmap functionality (demonstrated in `test_outerworld.py`) leverages `SHRINK`, `EXPAND`, and `RESHAPE` ops to make the vmapped dimension vanish and reappear. However, at their core, these movement ops were designed to either modify the total number of elements or the number of dimensions, but **not both at the same time**. Vmapping requires exactly this — slicing out one dimension (reducing both element count and rank) and later re-inserting it — and mimicking this with clever combinations of existing ops leads to unexpected behaviors in rangeify with outerworld ranges.

The `run_rangeify` function traverses the graph **backwards**, creating ranges at realized ops and propagating them backward. Outerworld ranges (ranges that already exist, like the vmap batch `RANGE`) are propagated **forward**. The ranges created during realization are based on the dimensions in the shape of the realized op. When outerworld ranges appear in the shape, `IndexingContext.new_range` does not recreate them — it passes them through.

This design assumes that realized ops (in particular `CONTIGUOUS`) don't contain **transformations** of a `RANGE` op in their shape, like `rm1 // 3` or `rm1 * 3`. When vmap is implemented with movement ops, such transformed symbolic expressions inevitably appear in shapes, making a variety of operations impossible without complex static inference of the static portion of symbolic shapes.

The solution: **don't store the outerworld ranges in the shape at all**. Instead, store them in the `src` tuple of dedicated ops (`VMAPIN` / `VMAPOUT`) that make the vmapped dimension appear and disappear. Both the static shape inference and the backward range propagation become much simpler — the shapes are always concrete integers, and the vmap `RANGE` is carried as metadata alongside the graph rather than embedded in it.

---

## 2. UOp Definitions

### 2.1 `VMAPIN`

```
UOp(Ops.VMAPIN, dtype, src=(input, *vargs))
```

**Purpose:** Marks the entry into a vmap scope. Slices one dimension out of the input tensor using a `RANGE` loop variable.

**Shape rule:** The output shape drops every axis whose corresponding `varg` is a `RANGE`, keeping only axes whose `varg` is `CONST(0)`.

```python
# Example: input shape (3, 6), varg = (RANGE(3), CONST(0))
# Output shape: (6,) — axis 0 is "consumed" by the RANGE
```

From `ops.py`:
```python
case Ops.VMAPIN:
  assert self.src[0].ndim == len(self.src[1:])
  return tuple(d for d, a in zip(self.src[0]._shape, self.src[1:]) if a == UOp.const(dtypes.index, 0))
```

**Spec constraint** (from `spec.py`): Each `varg` must be either a `RANGE` or `CONST(0)` with `dtypes.index`, and the input's ndim must equal the number of vargs.

### 2.2 `VMAPOUT`

```
UOp(Ops.VMAPOUT, dtype, src=(result, *vargs))
```

**Purpose:** Marks the exit from a vmap scope. Re-inserts the batch dimension(s) stripped by `VMAPIN`.

**Shape rule:** The output shape inserts a new axis (with size from the `RANGE`) at each position where the `varg` is a `RANGE`.

```python
# Example: result shape (6,), varg = (RANGE(3), CONST(0))
# Output shape: (3, 6) — axis 0 is re-inserted with size 3
```

From `ops.py`:
```python
case Ops.VMAPOUT:
  out_shape = list(self.src[0]._shape)
  for i, a in enumerate(self.src[1:]):
    if a.op == Ops.RANGE:
      out_shape.insert(i, a.src[0].arg)
  return tuple(out_shape)
```

**Spec constraint:** The inner result's ndim must equal the count of `CONST(0)` entries in the vargs.

### 2.3 GroupOp.VMap

Both ops belong to `GroupOp.VMap = {Ops.VMAPIN, Ops.VMAPOUT}`. This group is used throughout the scheduler for special-case handling.

---

## 3. Tensor-Level API

At the `Tensor` level, two methods wrap the UOp constructors:

```python
def vmapin(self, varg: tuple[UOp, ...]) -> Tensor: ...
def vmapout(self, varg: tuple[UOp, ...]) -> Tensor: ...
```

A typical usage pattern:

```python
r = UOp.range(batch_size, axis_id, AxisType.LOOP)
varg_in = (r, UOp.const(dtypes.index, 0))  # map over axis 0 of a 2D input
inner = x.vmapin(varg_in)       # shape drops axis 0
result = fn(inner)               # arbitrary computation on the slice
varg_out = (r,) + tuple(UOp.const(dtypes.index, 0) for _ in range(result.ndim))
output = result.vmapout(varg_out)  # re-inserts batch dim at position 0
```

The helper `_vmap_simple` in the test suite encapsulates this:

```python
def _vmap_simple(fn, x, in_axis=0, axis_id=-1):
  batch_size = x.shape[in_axis]
  varg_in = _varg(batch_size, in_axis, x.ndim, axis_id)
  result = fn(x.vmapin(varg_in))
  varg_out = (varg_in[in_axis],) + tuple(UOp.const(dtypes.index, 0) for _ in range(result.ndim))
  return result.vmapout(varg_out)
```

---

## 4. Scheduler Integration: Rangeify (`tinygrad/schedule/indexing.py`)

The rangeify pass (`run_rangeify`) converts the high-level tensor graph into a lower-level representation with explicit `RANGE` loops, `BUFFERIZE` (store), and `INDEX` (load) operations. Vmap processing happens in several stages.

### 4.1 Realization

`VMAPOUT` is **always realized** (like `CONTIGUOUS`, `COPY`, `STORE`):

```python
(UPat({Ops.COPY, Ops.BUFFER_VIEW, Ops.CONTIGUOUS, Ops.STORE, Ops.ENCDEC, Ops.VMAPOUT}, name="tr"), realize),
```

This means every `VMAPOUT` creates a buffer boundary — the vmap computation writes its results to a buffer. Note that adding `VMAPOUT` to the always-realized set introduces `BUFFERIZE` ops and new ranges, but these can be optimized out by the "remove bufferize with cost function" pass when the buffer is unnecessary.

### 4.2 Range Assignment (Backward Pass)

During the main backward pass over the topological order, ranges propagate from outputs to inputs. `VMAPIN` and `VMAPOUT` have special range handling:

**VMAPOUT — creating output ranges:**
When a `VMAPOUT` is realized, its output ranges are built by reusing the `RANGE` UOps from its vargs (for batch dimensions) and creating new ranges for the remaining dimensions:

```python
out_rngs = tuple(a if a.op == Ops.RANGE else rctx.new_range(d)
                 for a, d in zip(x.src[1:], x.shape)) if x.op == Ops.VMAPOUT else ...
```

**VMAPOUT — stripping ranges for the inner graph:**
When propagating ranges backward to the inner result, `VMAPOUT` removes the batch dimension ranges and only passes through the ranges for the non-batch axes:

```python
elif x.op == Ops.VMAPOUT:
  assert len(x.src[0].shape) == x.src[1:].count(UOp.const(dtypes.index, 0))
  rngs = tuple(r for a, r in zip(x.src[1:], out_rngs) if a.op == Ops.CONST)
```

**VMAPIN — re-inserting ranges:**
`VMAPIN` reverses the stripping: it inserts the `RANGE` vargs back at the positions consumed during input, so the actual buffer access includes the batch loop variable:

```python
elif x.op == Ops.VMAPIN:
  out_rngs_iter = iter(out_rngs)
  rngs = tuple(a if a.op == Ops.RANGE else next(out_rngs_iter) for a in x.src[1:])
```

**Dissolution during `pm_apply_rangeify`:**
After range assignment, `VMAPIN` and `VMAPOUT` nodes are dissolved — they are replaced by their inner source, since the range information they carry has been absorbed into the `range_map`:

```python
if x.op in GroupOp.VMap:
  assert len(new_srcs) == 1
  return new_srcs[0]
```

### 4.3 The "Ambient Range" Design

The vmap `RANGE` is **ambient**: it exists because `VMAPIN` references it, but it is not tracked in the per-node `out_rngs` for inner nodes. This means inner nodes (between `VMAPIN` and `VMAPOUT`) don't "know" about the batch dimension in their range tuples — they only see the local dimensions. The batch loop is only introduced when the kernel is generated because the `RANGE` appears in the compute graph via `VMAPIN`.

This works perfectly when there is **no intermediate realization** between `VMAPIN` and `VMAPOUT`.

---

## 5. The Intermediate Realization Bug and Fix

### 5.1 The Problem

When an inner node (between `VMAPIN` and `VMAPOUT`) gets **realized** — i.e., the scheduler decides it needs an intermediate buffer — the buffer dimensions are determined by `closed_ranges`, which come from the node's `out_rngs`. Since the vmap `RANGE` was stripped by `VMAPOUT` and is "ambient," it is **not** in `out_rngs`. The intermediate buffer is allocated without the batch dimension.

**Concrete example:**
```python
x = X.vmapin((CONST(0), RANGE(3)))  # shape (4,)
z = Tensor.stack(x[:2].sum(), x[2:].sum())  # shape (2,)
z_reindexed = z[[0, 1]]  # triggers multi-consumer realization
result = z_reindexed.vmapout((RANGE(3), CONST(0)))
```

The indexing `z[[0, 1]]` creates `EXPAND`/`REDUCE_AXIS` ops that give `z` multiple consumers. The ending-ranges heuristic decides `z` must be realized. The intermediate buffer gets shape `(2,)` instead of `(3, 2)`. The store kernel loops over the vmap range but writes all 3 batch iterations to the same 2 positions — only the last iteration survives.

### 5.2 The Fix

Three additive changes in `tinygrad/schedule/indexing.py`:

#### (a) Forward Pass: Track Vmap Dependency (`vmap_map`)

A new field `vmap_map: dict[UOp, tuple[UOp, ...]]` on `IndexingContext` records which nodes depend on a vmap input and which `RANGE`(s) they use. A forward pass over the topological order propagates vmap ranges from `VMAPIN` to all transitive consumers, stopping at `VMAPOUT`:

```python
for x in tsink_toposort:
  if x.op == Ops.VMAPIN:
    rctx.vmap_map[x] = tuple(a for a in x.src[1:] if a.op == Ops.RANGE)
  elif x.op == Ops.VMAPOUT:
    continue  # don't propagate past VMAPOUT
  else:
    for s in x.src:
      if s in rctx.vmap_map and x not in rctx.vmap_map:
        rctx.vmap_map[x] = rctx.vmap_map[s]
```

**Why forward, not backward:** Backward propagation from `VMAPOUT` would mark every node between `VMAPOUT` and `VMAPIN` — including external constants (like index buffers `[0, 1]` for gather operations) that don't depend on the batch. Adding vmap ranges to those buffers causes size mismatches. Forward propagation from `VMAPIN` only marks nodes whose computation actually varies per batch element.

#### (b) Prepend Vmap Ranges to `BUFFERIZE`

When creating a `BUFFERIZE` for a realized inner node that is in `vmap_map`, the vmap ranges are prepended to `closed_ranges`. This makes the intermediate buffer large enough to hold all batch elements. `VMAPOUT` is excluded because its `out_rngs` already contain the vmap range by construction:

```python
s_vmap = ctx.vmap_map.get(s, ()) if s.op != Ops.VMAPOUT else ()
closed_ranges = s_vmap + closed_ranges
```

#### (c) Include Vmap Ranges in `INDEX` for Loads

The `INDEX` that reads from the bufferized node must also use the vmap ranges so it addresses the correct batch slice. For most consumers inside the vmap scope, the vmap ranges come from `vmap_map`. For `VMAPOUT` (which is not in `vmap_map` since propagation stops there), the vmap ranges are extracted from its own `out_rngs`:

```python
x_vmap = ctx.vmap_map.get(x, ())
if not x_vmap and s_vmap and x.op == Ops.VMAPOUT:
  x_vmap = tuple(r for a, r in zip(x.src[1:], ctx.range_map[x][1]) if a.op == Ops.RANGE)
local_idx_rngs = tuple(r for i, r in enumerate(ctx.range_map[x][0]) if i in realized_ranges)
new_src = new_src.index(*x_vmap, *local_idx_rngs) if s_vmap else new_src.index(*local_idx_rngs)
```

### 5.3 Effect on Generated Code

**Before (buggy):** Intermediate buffer has size 2, store ignores batch index:
```c
void r_3_2_2_2(double* data0_2, double* data1_12, ...) {
  for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {
    *(data0_2+0) = (val3+val0);   // no Lidx2 offset — last iteration wins
    *(data0_2+1) = (val1+val2);
  }
}
```

**After (fixed):** Intermediate buffer has size 6 (3×2), store offsets by batch index:
```c
void r_3_2_2_2(double* data0_6, double* data1_12, ...) {
  for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {
    int alu0 = (Lidx2<<1);
    *(data0_6+(alu0+1)) = (val1+val2);   // offset by batch
    *(data0_6+alu0) = (val3+val0);
  }
}
```

---

## 6. Supported Constructs

### 6.1 Basic Operations

| Construct | Example | Status |
|---|---|---|
| Elementwise ops | `x * w`, `x + y`, `2 * x` | ✅ |
| Unary ops | `x.exp2()`, `x.sqrt()`, etc. | ✅ |
| Ternary ops | `cond.where(a, b)` | ✅ |
| Casting | `x.cast(dtypes.float16)` | ✅ |

### 6.2 Reductions

| Construct | Example | Status |
|---|---|---|
| Sum | `x.sum()`, `x.sum(axis=0)` | ✅ |
| Mean | `x.mean()` | ✅ |
| Max | `x.max()` | ✅ |
| Matmul / dot product | `x @ M`, `row @ matrix` | ✅ |

### 6.3 Movement Operations

| Construct | Example | Status |
|---|---|---|
| Reshape | `x.reshape((2, 3))` | ✅ |
| Flatten | `x.flatten()` | ✅ |
| Permute / Transpose | `x.transpose()`, `x.permute(...)` | ✅ |
| Expand | `x.expand((3,))` | ✅ |
| Shrink / Slice | `x[:2]`, `x[1:3, 2:4]` | ✅ |
| Pad | `x.pad(((1,0), (0,1)))` | ✅ |
| Flip | `x.flip(0)` | ✅ |
| Unsqueeze / Squeeze | `x.unsqueeze(0)` | ✅ |

### 6.4 Stacking and Concatenation

| Construct | Example | Status |
|---|---|---|
| Stack | `Tensor.stack(a, b, c)` | ✅ |
| Stack of reduces | `Tensor.stack(x[:2].sum(), x[2:].sum())` | ✅ |
| Cat | `Tensor.cat(vmapped, other, dim=0)` | ✅ |

### 6.5 Indexing (Including Multi-Consumer Patterns)

| Construct | Example | Status |
|---|---|---|
| Scalar index | `flat[0]` | ✅ |
| Slice | `x[:3]` | ✅ |
| Fancy index (list) | `z[[0, 1]]` | ✅ |
| Fancy index (Tensor) | `z[Tensor([0, 1])]` | ✅ |
| Multi-scalar stack | `Tensor.stack(flat[0], flat[4], flat[8])` | ✅ |
| Index after reduce+stack | `Tensor.stack(x[:2].sum(), x[2:].sum())[[0,1]]` | ✅ (fixed) |

### 6.6 Post-Vmap Operations

All standard tensor operations work on the output of a `VMAPOUT`:

| Construct | Example | Status |
|---|---|---|
| Reshape / Flatten | `vmap_result.reshape(9, 2)` | ✅ |
| Transpose | `vmap_result.transpose()` | ✅ |
| Pad / Flip | `vmap_result.pad(...)` | ✅ |
| Indexing / Slicing | `vmap_result[2, 4]`, `vmap_result[1:3]` | ✅ |
| Reductions | `vmap_result.sum()`, `vmap_result.sum(0)` | ✅ |
| Comparisons | `vmap_result == 0.0`, `vmap_result > 1.0` | ✅ |
| Matmul | `vmap_result @ weights` | ✅ |
| Cat with non-vmap | `Tensor.cat(vmap_result, other)` | ✅ |

### 6.7 Nested Vmap

Nested vmap is supported (vmap inside vmap), up to a tested depth of 3 levels. Deeply nested vmap (4+ levels) has known issues with the pad+reduce pattern.

| Construct | Status |
|---|---|
| `vmap(vmap(fn))` | ✅ |
| `vmap(vmap(vmap(fn)))` | ✅ |
| `vmap(vmap(vmap(vmap(fn))))` | ⚠️ Known issues |
| Nested with different in_axes | ✅ |
| Nested with reductions | ✅ |
| Nested with matmul | ✅ |

### 6.8 Multi-Input Vmap

When using the pad+reduce vmap pattern (not `VMAPIN`/`VMAPOUT`), multi-input vmap is supported:

| Construct | Status |
|---|---|
| Same axis for all inputs | ✅ |
| Different axes per input | ✅ |
| Broadcasting (unmapped) inputs | ✅ |
| Mixed mapped/broadcast inputs | ✅ |

### 6.9 The `spjacobian` Pattern

The primary motivation for this implementation. The pattern is:

1. Compute multiple JVPs (each may contain reduces)
2. Stack them: `Tensor.stack(jvp0, jvp1, jvp2)`
3. Flatten: `.flatten()`
4. Re-index specific elements: `Tensor.stack(*[flat[i] for i in indices])`

This pattern creates multi-consumer nodes that trigger intermediate realization — exactly the case fixed by the `vmap_map` forward propagation.

---

## 7. Alternative Vmap Implementation: pad+reduce

There is a second vmap implementation that does **not** use `VMAPIN`/`VMAPOUT`. Instead it uses `pad` and `REDUCE` to achieve the same effect (see `test_native_vmap_pad.py`). The idea:

1. Slice the input using a `RANGE` variable
2. Apply the function
3. Reshape to insert a size-1 batch dimension
4. Pad that dimension to the full batch size, using the `RANGE` variable as the pad offset
5. Reduce (sum) over the `RANGE` to collapse the pad — only the non-zero slice survives

This approach works with standard tensor operations and doesn't need special scheduler support. It supports multi-input vmap and nesting. However, it has limitations at deeper nesting levels (4+).

---

## 8. Architecture Diagram

```
  Input Buffer (batch_size, ...)
        │
   ┌────▼────┐
   │ VMAPIN  │  vargs = (RANGE(batch_size), CONST(0), ...)
   └────┬────┘  shape: drops RANGE axes
        │
   [ user fn ]  elementwise, reduce, reshape, index, etc.
        │
   ┌────▼────┐
   │ VMAPOUT │  vargs = (RANGE(batch_size), CONST(0), ...)
   └────┬────┘  shape: re-inserts RANGE axes
        │
  Output Buffer (batch_size, ...)
```

**In the scheduler:**

```
Backward pass (range assignment):
  VMAPOUT creates out_rngs with RANGE for batch dims
    │
    ▼ strips batch RANGE, passes local ranges inward
  Inner nodes get local ranges only (batch RANGE is "ambient")
    │
    ▼ VMAPIN re-inserts batch RANGE for buffer indexing
  Input buffer indexed with both batch RANGE and local ranges

Forward pass (vmap_map):
  VMAPIN ──▶ propagates vmap RANGE to all dependent nodes ──▶ stops at VMAPOUT

When intermediate realization occurs:
  BUFFERIZE: closed_ranges = vmap_ranges + local_ranges  (buffer includes batch dim)
  INDEX:     index(*vmap_ranges, *local_ranges)           (load addresses batch slice)
```
