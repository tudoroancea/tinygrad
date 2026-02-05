# Fix: RANGE lost when intermediate nodes are realized inside a vmap scope

## The bug

When a tensor between `VMAPIN` and `VMAPOUT` gets realized (written to an intermediate buffer), the vmap RANGE is not included in that buffer's dimensions. The store kernel loops over the vmap range but writes to fixed positions without offsetting by the loop variable, so only the last iteration's values survive. All batch elements silently get the last batch's values.

This affects any pattern where a RANGE-dependent tensor containing reduce operations is indexed and re-stacked inside a vmap scope. The indexing creates multi-consumer nodes that trigger intermediate realization via the ending-ranges logic.

## Root cause

The range scheduler (`tinygrad/schedule/indexing.py`) propagates ranges backward from outputs to inputs. `VMAPOUT` strips its vmap RANGE from the ranges it passes to inner nodes — by design, since the inner computation has fewer dimensions than the outer shape. The vmap range is supposed to be "ambient": present in the generated kernel because `VMAPIN` references it, but not tracked in the per-node range tuples.

This works correctly when there is no intermediate realization: the final kernel (at the `VMAPOUT` boundary) naturally loops over the vmap range because `VMAPIN` introduces it into the compute graph.

The problem occurs when an inner node gets realized. Realization creates a `BUFFERIZE` (store) and `INDEX` (load) pair, and the dimensions of the intermediate buffer are determined by `closed_ranges` — the ranges from the node's `out_rngs`. Since the vmap range was stripped by `VMAPOUT` and never added back, `out_rngs` for inner nodes doesn't include it. The intermediate buffer is allocated with only the local dimensions (e.g., size 2 instead of 3×2), and the store writes all batch iterations to the same positions.

**Concrete example:** `Tensor.stack(x[:2].sum(), x[2:].sum())` produces an `ADD` node of shape `(2,)`. When this is indexed with `[[0,1]]`, the gather pattern creates `EXPAND`/`REDUCE_AXIS` ops that give `ADD` multiple consumers. The ending-ranges heuristic (line ~240) compares range IDs and decides `ADD` must be realized. A new range `r2` of size 2 is created — but the vmap range `rm10` of size 3 is nowhere in this buffer's dimensions.

## The fix

Three changes in `tinygrad/schedule/indexing.py`, all additive:

### 1. Forward pass to track vmap dependency (`vmap_map`)

A new field `vmap_map: dict[UOp, tuple[UOp, ...]]` on `IndexingContext` records which nodes are inside a vmap scope and which vmap RANGE(s) they depend on.

A forward pass over the topological order (before the main backward rangeify loop) propagates vmap ranges from `VMAPIN` to all transitive consumers, stopping at `VMAPOUT`. This correctly identifies only nodes whose values actually vary per batch element — external inputs like index buffers that happen to be used inside the scope are not marked.

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

### 2. Prepend vmap ranges to `BUFFERIZE` closed ranges

In `create_bufferize_and_index_based_on_ranges`, when creating a `BUFFERIZE` for a realized source `s` that is in `vmap_map`, the vmap ranges are prepended to `closed_ranges`. This makes the intermediate buffer large enough to hold all batch elements.

`VMAPOUT` is excluded because its `out_rngs` already contain the vmap range by construction.

```python
s_vmap = ctx.vmap_map.get(s, ()) if s.op != Ops.VMAPOUT else ()
closed_ranges = s_vmap + closed_ranges
```

### 3. Include vmap ranges in `INDEX` when loading from the buffer

The `INDEX` that reads from the bufferized node must also use the vmap ranges so it addresses the correct batch slice. For most consumers inside the vmap scope, the vmap ranges come from `vmap_map`. For `VMAPOUT` (which is not in `vmap_map` since we stop propagation there), the vmap ranges are extracted from its own `out_rngs`.

```python
x_vmap = ctx.vmap_map.get(x, ())
if not x_vmap and s_vmap and x.op == Ops.VMAPOUT:
    x_vmap = tuple(r for a,r in zip(x.src[1:], ctx.range_map[x][1]) if a.op == Ops.RANGE)
local_idx_rngs = tuple(r for i,r in enumerate(ctx.range_map[x][0]) if i in realized_ranges)
new_src = new_src.index(*x_vmap, *local_idx_rngs) if s_vmap else new_src.index(*local_idx_rngs)
```

## Why forward propagation (not backward)

The initial approach was to propagate vmap ranges backward from `VMAPOUT` through consumers. This marks every node between `VMAPOUT` and `VMAPIN` — including external constant buffers (like `[0, 1]` for indexing) that don't depend on the batch. Adding vmap ranges to those buffers causes size mismatches.

Forward propagation from `VMAPIN` only marks nodes whose computation actually depends on vmap input data. External constants that feed into the scope but don't originate from `VMAPIN` are correctly left unmarked.

## Effect on generated code

Before (buggy) — intermediate buffer has size 2, store ignores batch index:
```c
void r_3_2_2_2(double* data0_2, double* data1_12, ...) {
  for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {
    *(data0_2+0) = (val3+val0);   // no Lidx2 offset — last iteration wins
    *(data0_2+1) = (val1+val2);
  }
}
```

After (fixed) — intermediate buffer has size 6 (3×2), store offsets by batch index:
```c
void r_3_2_2_2(double* data0_6, double* data1_12, ...) {
  for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {
    int alu0 = (Lidx2<<1);
    *(data0_6+(alu0+1)) = (val1+val2);   // offset by batch
    *(data0_6+alu0) = (val3+val0);
  }
}
```

## Test results

- All 2573 existing tests pass (the only pre-existing failure is an unrelated `test_autogen` libclang issue).
- The `test_vmap.py::TestVmap::test_multiple_consumers` test — which exercises the exact multi-consumer + vmap pattern — passes.
- The minimal reproducer (`tinygrad_vmap_bug.py`) now passes for all variants: reduce+index, no-reduce+index, and matmul+index.
