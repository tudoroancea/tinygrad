"""
Minimal reproducer for tinygrad vmap bug: indexing a RANGE-dependent tensor
that contains reduce operations loses the RANGE dependency.

All batch elements silently get the last batch element's values.

Run with:
  CPU=1 SPEC=0 python tinygrad_vmap_bug.py
"""

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType
import numpy as np

np.random.seed(42)

# === Setup: batched input via vmapin ===
X_np = np.random.randn(4, 3)
X = Tensor(X_np)
r = UOp.range(3, -10, AxisType.LOOP)
x = X.vmapin((UOp.const(dtypes.index, 0), r))  # shape (4,), varies per batch

# === Build a 1D tensor with reduce operations ===
# Tensor.stack of reduced scalars — each depends on the RANGE
z = Tensor.stack(x[:2].sum(), x[2:].sum())  # shape (2,)

# === BUG: any indexing on z loses the RANGE ===
z_reindexed = z[[0, 1]]  # should be identity, but all batches = last batch

# Materialize
direct = z.vmapout((r, UOp.const(dtypes.index, 0))).numpy()
reindexed = z_reindexed.vmapout((r, UOp.const(dtypes.index, 0))).numpy()

expected = np.array([[X_np[:2, b].sum(), X_np[2:, b].sum()] for b in range(3)])

print("Direct vmapout (CORRECT):")
print(direct)
print()
print("Indexed vmapout (BUGGY — all rows = last batch):")
print(reindexed)
print()
print("Expected:")
print(expected)
print()

direct_ok = np.allclose(direct, expected)
reindexed_ok = np.allclose(reindexed, expected)
print(f"Direct correct:  {direct_ok}")
print(f"Indexed correct: {reindexed_ok}")

if not reindexed_ok:
    print()
    print("BUG CONFIRMED: indexing a RANGE-dependent tensor containing reduce ops")
    print("causes the RANGE to be lost. All batch elements get the last batch's values.")
    print()
    print("Without reduce ops, indexing works correctly:")

    # Proof: no reduce -> indexing works
    y = x[:2]  # (2,) — no reduce
    y_indexed = y[[0, 1]]
    y_direct = y.vmapout((r, UOp.const(dtypes.index, 0))).numpy()
    y_reindexed = y_indexed.vmapout((r, UOp.const(dtypes.index, 0))).numpy()
    print(f"  No-reduce direct == indexed: {np.allclose(y_direct, y_reindexed)}")

    # Also fails with matmul (another reduce op)
    print()
    print("Also reproduces with matmul:")
    M = Tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    t = x @ M  # (2,) — reduce via dot product
    t_direct = t.vmapout((r, UOp.const(dtypes.index, 0))).numpy()
    t_indexed = t[[0, 1]].vmapout((r, UOp.const(dtypes.index, 0))).numpy()
    t_expected = X_np.T @ np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    print(f"  Matmul direct correct:  {np.allclose(t_direct, t_expected)}")
    print(f"  Matmul indexed correct: {np.allclose(t_indexed, t_expected)}")

assert direct_ok, "Direct vmapout should be correct"
assert reindexed_ok, "Indexed vmapout should match direct (this is the bug)"
