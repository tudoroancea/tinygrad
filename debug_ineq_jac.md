# Debugging: bicycle_ocp ineq_constraints_jac correctness failure

## Symptom

`uv run pytest -sk 'bicycle and correct'` fails. The ineq constraints **values** are correct, but the **Jacobian** is wrong: all stage blocks contain the **last stage's** Jacobian values instead of each stage's own values. The eq_constraints_jac passes fine.

## Setup

- `nx=4, nu=2, ngamma=1, N=3`
- `m_stage_ineq=2`, no state_diff/control_diff ineq, `m_final_ineq=1`
- Stage ineq: `c(x,u,γ) = [Σx², Σu²]` — very simple sum-of-squares
- The Jacobian is a 7×23 sparse CSC matrix with 34 nnz

## Root cause: tinygrad bug with scalar indexing on RANGE-dependent reduce-stacked tensors

The bug is in `spjacobian` (`src/anvil/ad/jacobian.py:80-93`). The `unroll=True` path does:

```python
flattened_compressed_jac = Tensor.stack(*[jvp(...)[0] for j in range(n_colors)], dim=1).flatten()
return Tensor.stack(*[flattened_compressed_jac[i] for i in uncompression_idx])
```

When `spjacobian` is called inside a `vmap`, the `flattened_compressed_jac` is a 1D tensor with an implicit RANGE dependency from the outer vmap. The final `Tensor.stack(*[flattened_compressed_jac[i] for i in uncompression_idx])` performs scalar indexing on this tensor.

**The tinygrad bug:** Scalar indexing (`t[i]`) on a 1D tensor that was created by `Tensor.stack` of reduced scalars, where the tensor has an implicit RANGE dependency from `vmapin`, causes the RANGE to be lost. All batch elements silently get the last batch element's values.

### Minimal reproducer (no anvil dependency)

See `tinygrad_vmap_bug.py` for a standalone script. The core pattern:

```python
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType
import numpy as np
import os; os.environ["SPEC"] = "0"

X = Tensor(np.random.randn(4, 3))
r = UOp.range(3, -10, AxisType.LOOP)
x = X.vmapin((UOp.const(dtypes.index, 0), r))  # (4,) with RANGE

# Reduced scalars
s0 = x[:2].sum()
s1 = x[2:].sum()

# Stack into 1D
t = Tensor.stack(s0, s1)  # (2,) — correct RANGE dependency

# BUG: scalar indexing + re-stacking loses RANGE
reindexed = Tensor.stack(t[0], t[1])  # all batch elements = last batch

# Direct vmapout works correctly:
t.vmapout((r, UOp.const(dtypes.index, 0)))  # ✓ correct

# Re-indexed vmapout is wrong:
reindexed.vmapout((r, UOp.const(dtypes.index, 0)))  # ✗ all rows identical
```

### Key observations

1. **Without reduce operations**, scalar indexing + re-stacking works fine:
   ```python
   t = Tensor.stack(x[0], x[1])  # no reduce
   Tensor.stack(t[0], t[1])  # ✓ correct
   ```

2. **With reduce operations**, scalar indexing breaks:
   ```python
   t = Tensor.stack(x[:2].sum(), x[2:].sum())  # with reduce
   Tensor.stack(t[0], t[1])  # ✗ RANGE lost
   ```

3. **Using `t[i]` alone** (without re-stacking) crashes in rangeify:
   ```
   AssertionError in schedule/indexing.py:274 (run_rangeify)
   assert len(x.src[0].shape) == x.src[1:].count(UOp.const(dtypes.index, 0))
   ```

4. The RANGE is present in the UOp graph at every step — the loss happens during scheduling/codegen.

## Why eq_constraints_jac works

The dynamics Jacobian uses the `unroll=False` path with `sep_uncompression=True`:

```python
flattened_compressed_jacobian = vmap(lambda v: jvp(...)[0], 1, 1, -1)(basis).flatten()
flattened_compressed_jacobian = flattened_compressed_jacobian.contiguous()
return flattened_compressed_jacobian[uncompression_idx]
```

This uses array indexing (`tensor[idx_list]`) rather than scalar indexing + `Tensor.stack`, which avoids the bug.

## Impact on generated code

In the generated C++ (`anvil_bicycle_ocp.cpp`), the bug manifests as:
- Kernel `r_3_4_2_4_2_4_2_4_4`: loop over 3 stages writes to fixed output positions `data0_8[0..7]` without offsetting by the loop index → only last iteration's values survive
- Kernel `E_3_6`: reads the 8 values once and copies the same values to all 3 stage slots

## Proposed fix directions

### Option A: Fix in tinygrad (preferred)

Fix the rangeify/scheduling pass to correctly propagate RANGE through scalar indexing on reduce-stacked tensors. The RANGE is present in the UOp graph; it's lost during the lowering from tensor-level to kernel-level operations. The crash in `schedule/indexing.py:274` suggests the rangeify pass doesn't handle the case where a scalar (`shape=()`) with RANGE dependency is indexed from a 1D reduce-stacked tensor.

### Option B: Work around in spjacobian (anvil-side)

Use array indexing instead of scalar indexing + Tensor.stack:

```python
# Instead of:
return Tensor.stack(*[flattened_compressed_jac[i] for i in uncompression_idx])

# Use:
return flattened_compressed_jac[uncompression_idx]
```

This might also need `.contiguous()` before indexing (as the `sep_uncompression` path does).

### Option C: Always use the `unroll=False` path for vmapped calls

The `unroll=False` path with `sep_uncompression=True` works correctly because it avoids scalar indexing.

## Files involved

- `src/anvil/ad/jacobian.py:43-97` — `spjacobian` function (where the bug manifests)
- `src/anvil/ad/sparsity.py:235-329` — `compute_jacobian_coloring` (computes uncompression_idx)
- `src/anvil/utils.py:73-85` — `vmap` utility
- `src/anvil/ocp/problem.py:471-554` — `ineq_constraints_jac_fn` (calls vmap of spjacobian)
- `tinygrad_vmap_bug.py` — standalone minimal reproducer
- `.venv/lib/python3.14/site-packages/tinygrad/schedule/indexing.py:274` — crash site in tinygrad
