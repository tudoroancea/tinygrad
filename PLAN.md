# vmap Transform Implementation Plan

## Overview

Implement a `vmap` function similar to `jax.vmap` that vectorizes a function over a batch dimension. The implementation uses **graph tracing and transformation** to adjust operations for batching.

**Location**: `tinygrad/transforms/vmap.py`

**API**:
```python
def vmap(fn: Callable[..., Tensor], in_axes: int | None | tuple[int | None, ...] = 0) -> Callable[..., Tensor]:
    """Vectorizing map. Creates a function which maps fn over argument axes.
    
    Args:
        fn: Function to be mapped over additional axes.
        in_axes: An integer, None, or tuple specifying which input array axes to map over.
                 - int: map over that axis for all inputs
                 - None: broadcast (don't map) for all inputs  
                 - tuple: per-input specification
    
    Returns:
        Batched version of fn that adds a batch dimension to inputs/outputs.
    """
```

---

## Phase 1: Core Infrastructure

Set up the module structure and basic tracing mechanism.

- [x] **1.1** Create `tinygrad/transforms/` directory with `__init__.py`
- [x] **1.2** Create `tinygrad/transforms/vmap.py` with basic structure
- [x] **1.3** Implement input validation and `in_axes` normalization
  - Handle `int`, `None`, and `tuple` forms
  - Validate axis bounds for each input
- [x] **1.4** Implement graph capture mechanism
  - Create placeholder tensors for tracing
  - Call function to build lazy UOp graph
  - Extract the resulting UOp DAG

---

## Phase 2: Batching Rules for Elementwise Ops

Define how each operation transforms under batching. Start with the simplest cases.

- [x] **2.1** Implement tracking of which tensors have batch dimensions
- [x] **2.2** Implement batching rules for unary elementwise ops
  - `EXP2`, `LOG2`, `SIN`, `SQRT`, `RECIPROCAL`, `NEG`
  - Rule: output has same batch axis as input
- [x] **2.3** Implement batching rules for binary elementwise ops
  - `ADD`, `MUL`, `SUB`, `FDIV`, `MAX`
  - Rule: broadcast non-batched input, output has batch axis
- [x] **2.4** Implement batching rules for comparison ops
  - `CMPLT`, `CMPEQ`, `CMPNE`
- [x] **2.5** Implement batching rules for ternary ops
  - `WHERE`: all three inputs may have batch dimension
- [x] **2.6** Implement batching for `CAST`, `BITCAST`

---

## Phase 3: Batching Rules for Movement Ops

Movement ops require careful axis manipulation.

- [x] **3.1** `RESHAPE`
  - Insert batch dimension into new shape
  - `(batch, *old_shape) -> (batch, *new_shape)`
- [x] **3.2** `PERMUTE`
  - Shift permutation indices to account for batch axis
  - Keep batch axis at position 0
- [x] **3.3** `EXPAND`
  - Insert batch dimension into expand shape
- [x] **3.4** `PAD`
  - Add `(0, 0)` padding for batch dimension
- [x] **3.5** `SHRINK`
  - Add full range for batch dimension
- [x] **3.6** `FLIP`
  - Shift flip axes by 1

---

## Phase 4: Batching Rules for Reduce Ops

Reduce ops need axis adjustment.

- [x] **4.1** `REDUCE_AXIS`
  - Shift reduction axes by 1 (to account for batch dim at position 0)
  - Output retains batch dimension
- [x] **4.2** Handle partial reduction (specific axes)
  - Correctly shift only the specified axes

---

## Phase 5: Batching Rules for Other Ops

Handle remaining operations.

- [x] **5.1** `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `DETACH`
  - Pass through batch axis
- [ ] **5.2** `COPY`
  - Preserve batch axis through device copy
- [ ] **5.3** Handle constants properly
  - When used with batched tensors, broadcast correctly

---

## Phase 6: Graph Rewriting Implementation

Implement the actual transformation.

- [x] **6.1** Implement recursive graph transformation with memoization
- [x] **6.2** Handle UOp infrastructure nodes (UNIQUE, DEVICE, BUFFER)
- [x] **6.3** Implement proper constant handling with broadcasting
- [x] **6.4** Implement output construction from transformed graph

---

## Phase 7: Testing

- [x] **7.1** Basic tests: elementwise ops (`add`, `mul`, etc.)
- [x] **7.2** Movement op tests: `reshape`, `permute`, `transpose`, `pad`
- [x] **7.3** Reduce op tests: `sum`, `max` with various axes
- [x] **7.4** Mixed `in_axes` tests: some batched, some broadcasted
- [x] **7.5** Composition tests: `vmap(vmap(fn))` for nested batching
- [x] **7.6** Compare outputs against JAX for correctness
- [x] **7.7** Test edge cases:
  - Input validation
  - Axis bounds checking

---

## Phase 8: Polish & Integration

- [x] **8.1** Export `vmap` from `tinygrad/__init__.py`
- [x] **8.2** Add docstrings and type hints
- [ ] **8.3** Performance comparison with manual batching
- [ ] **8.4** Update `AGENTS.md` with implementation notes

---

## Implementation Notes

### Approach

The implementation uses a **trace-and-transform** approach:

1. **Trace**: Call the function with placeholder tensors (unbatched shapes) to capture the computation graph
2. **Map inputs**: Associate placeholder UOps with batched input tensors
3. **Transform**: Walk the traced graph recursively, transforming each operation:
   - Track which tensors have batch dimensions
   - For ALU ops: broadcast non-batched inputs
   - For movement ops: adjust shape/axis arguments
   - For reduce ops: shift reduction axes
4. **Return**: The transformed output tensor with batch dimension at position 0

### Key Files

| File | Description |
|------|-------------|
| `tinygrad/transforms/__init__.py` | Exports `vmap` |
| `tinygrad/transforms/vmap.py` | Main implementation (~350 lines) |
| `tinygrad/__init__.py` | Updated to export `vmap` |
| `test/test_vmap2.py` | Test suite (16 tests, all passing) |

### Differences from Existing VMAPIN/VMAPOUT

The existing approach in `test/test_vmap.py` uses special `VMAPIN`/`VMAPOUT` UOps that mark batch dimension boundaries in the graph. This new implementation instead:

- Works at the Tensor level, not UOp level
- Traces with placeholder tensors and transforms the graph
- More similar to JAX's approach

Both implementations can coexist.

---

## Future Extensions

- [ ] `out_axes` parameter for controlling output batch axis position
- [ ] `axis_size` parameter for cases where all `in_axes=None`
- [ ] Pytree support for nested dict/list inputs
- [ ] Support for functions with multiple tensor outputs
- [ ] Gradient support (`vmap` of `grad`)
