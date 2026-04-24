# Codegenerable Cholesky in tinygrad

A collection of Cholesky-factorization (A = L·Lᵀ) implementations written
directly against tinygrad's UOp / `custom_kernel` API.

Constraints every variant respects:
- No `.realize()` inside the compute — one realize at the boundary.
- No Python loops that dispatch kernels between iterations. Kernels either
  all live inside a single custom kernel (v1-v5) or are chained in a single
  UOp graph that tinygrad schedules as one pipeline (multi-kernel variants).
- Works with stock tinygrad — no monkey-patching, no compiler flags.

## Usage

```python
from extra.cholesky import cholesky
from tinygrad import Tensor

L = cholesky(Tensor(A))   # A is (N, N) SPD; L is lower-triangular with A = L @ L.T
```

The dispatcher picks the fastest variant that fits the device and N.

## Variants

| File | Algorithm | Parallelism | Backends |
|---|---|---|---|
| `cholesky_v1.py` | Left-looking triple loop, REDUCE axes | 1 thread | any |
| `cholesky_v2.py` | Same, rewritten to use `Ops.REDUCE` for the inner dot product | 1 thread | any |
| `cholesky_v3.py` | Single warp partitions k; LDS fan-in | 32 threads | GPU |
| `cholesky_v4.py` | Right-looking, T threads parallelise row i | 128 threads | GPU |
| `cholesky_v5.py` | Right-looking, 2-D thread tile `T_I × T_K` (1024 threads) | 1024 threads | GPU |

Run any file directly for a correctness sweep:

```
DEV=NV  .venv/bin/python extra/cholesky/cholesky_v5.py
DEV=CPU .venv/bin/python extra/cholesky/cholesky_v1.py
```

## Bench

```
DEV=NV  .venv/bin/python extra/cholesky/bench.py              # all variants × default sizes
DEV=NV  VER=v5 N=1024  .venv/bin/python extra/cholesky/bench.py
DEV=CPU VER=v1,numpy    .venv/bin/python extra/cholesky/bench.py
```

Bench uses wallclock (`time.perf_counter`) around one `.realize()` per
iteration, with explicit device synchronize. `VER=numpy` runs
`np.linalg.cholesky` for reference.

## Tests

```
DEV=NV  .venv/bin/python -m unittest extra.cholesky.test_cholesky -v
DEV=CPU .venv/bin/python -m unittest extra.cholesky.test_cholesky -v
```

GPU-only variants skip cleanly on backends without LOCAL memory support.

## Design notes (short)

- `AxisType.REDUCE` is used for every sequential axis (row loop, column loop,
  inner dot product). REDUCE is the only loop type that tinygrad's
  `convert_loop_to_global` leaves as a serial for-loop; `LOOP` axes closed at
  sink level silently become parallel block dims and break the algorithm.
- Per-thread state (dot-product accumulator, etc.) lives in an `AddrSpace.REG`
  placeholder. Cross-thread combination goes through an `AddrSpace.LOCAL`
  buffer with `UOp.barrier`.
- Pass-1 writes the diagonal entry `L[J,J] = √(A[J,J] − s_J)`; a barrier; pass-2
  writes off-diagonal entries `L[i,J] = (A[i,J] − s_i) / L[J,J]`. Every thread
  writes its own row, so the two passes cover the whole column.
- Edge cases we hit while building this: dynamic `UOp.range(j, …)` bounds
  crash `simplify_merge_adjacent` on same-type adjacent axes; `UNROLL` on a
  scalar REG accumulator emits invalid C; `tag=1` on the sink bypasses
  `apply_opts` but is rejected by `tensor_spec`. v5 uses static bounds + REDUCE
  axes + plain `Opts.to_apply=()` to sidestep all three.
