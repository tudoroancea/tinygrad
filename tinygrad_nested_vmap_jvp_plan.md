# Plan: Fix tinygrad nested vmap JVP bug (inner vmap + stacked value depending on gamma)

## Summary
`spjacobian` wraps JVP in an outer vmap. When the primal contains an **inner vmap** and the JVP tangent depends on a **non-vmapped scalar** (e.g., `gamma`), the outer vmap JVP produces zero derivatives. The issue occurs when `Tensor.stack` is applied to values that **already depend on that scalar**, and the result is then **multiplied by the scalar**. Direct JVP works; only the outer-vmap JVP fails.

This is a tinygrad realization issue with nested RANGE variables (inner VMAPIN/OUT + outer VMAPIN/OUT), not an Anvil JVP rule issue.

## Minimal reproduction (standalone)
```python
from tinygrad import Tensor, dtypes, UOp
from tinygrad.uop.ops import AxisType
from anvil.ad.jvp import jvp
from anvil.utils import vmap
from tinygrad.helpers import Context
import numpy as np

# inner vmap + scalar gamma; outer vmap for JVP
z = Tensor.rand(3, dtype=dtypes.float64)  # [x0, x1, gamma]

r_inner = UOp.range(1, -1, AxisType.LOOP)
varg = (UOp.const(dtypes.index, 0), r_inner)

xv = z[:2].reshape(2, 1).vmapin(varg)  # (2,)
g = z[2:]  # scalar

y = xv * g
output = (g * Tensor.stack(y[0], y[1])).vmapout(varg).flatten()  # (2,)

# Direct JVP w.r.t. gamma (correct)
with Context(TRACK_MATCH_STATS=0):
    t = Tensor([0.0, 0.0, 1.0], dtype=dtypes.float64)
    direct = jvp((output,), (z,), (t,))[0].numpy()

# Outer vmap JVP (incorrect, gamma column = 0)
with Context(TRACK_MATCH_STATS=0):
    jac = vmap(lambda v: jvp((output,), (z,), (v,))[0], 1, 1, -2)(Tensor.eye(3, dtype=dtypes.float64)).numpy()

print("direct:", direct)        # non-zero
print("vmap:", jac[:, 2])       # all zeros (BUG)
```

**Expected**: `jac[:, 2] == direct`

**Actual**: `jac[:, 2]` is all zeros; x-columns are correct.

## Key observations
- Direct JVP is correct. The issue only appears when wrapping JVP in an outer vmap.
- The bug appears when a **stacked** tensor depends on `gamma` and is then multiplied by `gamma`.
- Simplifications that avoid `Tensor.stack` or avoid reusing gamma in `g * stack(y)` make the bug disappear.
- The tangent graph is non-zero, but realization drops the `dg * stack(y)` term.

## Hypothesis
The interaction between nested RANGE variables (inner vmap RANGE and outer vmap RANGE) causes the **product rule term** `y * dg` to be lost during graph rewrite or scheduling (rangeify / bufferize / simplify). The term exists in the UOp graph but is eliminated or mis-typed when creating the schedule for nested VMAPOUT.

## Investigation plan
1. **Confirm tinygrad-only reproduction**
   - Reproduce in tinygrad directly (no anvil helpers) if possible.
   - Add a minimal unit test in tinygrad (`test/test_vmap.py` or similar).

2. **Inspect UOp graphs**
   - Dump the UOp graph for:
     - `output` (primal)
     - `jvp((output,), (z,), (t_body,))[0]` (tangent)
   - Verify the `y * dg` term exists in the JVP graph.

3. **Trace where the term is dropped**
   - Compare `graph_rewrite` outputs for:
     - direct JVP
     - outer vmap JVP
   - Focus on `schedule/rangeify.py` and `pm_const_buffer_folding` to see if the `dg * stack(y)` term is being simplified away.

4. **Check VMAPOUT / RANGE handling**
   - Inspect `tinygrad/uop/spec.py` and `schedule/rangeify.py` rules for `VMAPIN`/`VMAPOUT`.
   - Verify that tensors depending on **multiple RANGEs** are kept and not collapsed.

5. **Check stack implementation**
   - `Tensor.stack` lowers to `unsqueeze + cat`. Ensure those movement ops propagate RANGE vars correctly.
   - Specifically check that the second stack (on `y[0], y[1]`) keeps the outer RANGE dependency in `dg`.

## Proposed fix (likely locations)
- `tinygrad/schedule/rangeify.py`: ensure the rangeify map preserves multi-RANGE dependencies across `CAT`/`RESHAPE`/`PERMUTE` used by `stack`.
- `tinygrad/uop/spec.py`: verify spec for `VMAPOUT` allows inputs with extra RANGE dependencies.
- `tinygrad/tensor.py` stack lowering: ensure stack doesn’t introduce bufferization that drops a RANGE var.

## Validation plan
- Add tinygrad unit test for the minimal reproduction above.
- Ensure existing vmap tests pass.
- Re-run Anvil integration test `tests/integration/bicycle_ocp/test_vmap_spjacobian.py` once fixed (remove xfail).

## Expected outcome
- Outer-vmap JVP correctly matches direct JVP for the minimal example.
- `spjacobian(eq_constraints_fn)` returns a correct Jacobian for vmapped OCP constraints, enabling Phase 2 Step 3 automation.
