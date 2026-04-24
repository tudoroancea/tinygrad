# Codegenerable Cholesky in tinygrad

Implementations of Cholesky factorization (A = L · Lᵀ) written directly
against tinygrad's UOp / `custom_kernel` API.

Two rules every variant respects:

- No `.realize()` inside the compute — one final realize (or `TinyJit`).
- No Python loops that *dispatch kernels between iterations*. Python loops
  that *build* a lazy UOp graph are fine because tinygrad still schedules
  every resulting kernel into a single pipeline. The moment a `.realize()`
  sits inside a loop, the budget is spent and tinygrad has to re-enter
  scheduling in Python — that's what we forbid.

## Usage

```python
from extra.cholesky import cholesky
from tinygrad import Tensor

L = cholesky(Tensor(A))    # A is (N, N) SPD; L is lower-triangular with A = L @ L.T
```

The dispatcher picks the fastest variant that fits the device and N.

## Variants

### Single-kernel (one custom_kernel, no TRSM split)

| File | Algorithm | Parallelism | Backends |
|---|---|---|---|
| `cholesky_v1.py` | Left-looking triple loop, REDUCE axes | 1 thread | any |
| `cholesky_v2.py` | Same, rewritten to use `Ops.REDUCE` for the inner dot product | 1 thread | any |
| `cholesky_v3.py` | Single warp partitions k; LDS fan-in | 32 threads | GPU |
| `cholesky_v4.py` | Right-looking, T threads parallelise row i | 128 threads | GPU |
| `cholesky_v5.py` | Right-looking, 2-D thread tile `T_I × T_K` = 64 × 16 = 1024 threads | 1024 threads | GPU |

### Multi-kernel (chain of kernels in one UOp graph)

| File | Algorithm | Key idea |
|---|---|---|
| `cholesky_v7.py` | Blocked right-looking, one custom TRSM per block row | tinygrad matmul for trailing update, custom TRSM custom kernel for off-diagonal |
| `cholesky_v8.py` | v7 + blocked TRSM (small b_sub TRSMs + GEMMs) | splits the big TRSM but fires many kernels |
| `cholesky_v9.py` | v7, but compute L_II⁻ᵀ once against I and matmul | replaces the M×b TRSM with a tiny b×b TRSM-vs-identity + big GEMM |
| `cholesky_v10.py` | v9 + recursive 2×2 block inversion for L_II⁻ᵀ | base-case TRSM + matmul recursion — eliminates the last sequential hot spot |

All multi-kernel variants fall back to v5 when N is too small or not a
multiple of the block size.

## Dispatcher

`extra/cholesky/__init__.py` exports `cholesky(A)` which picks:

- `v1` for CPU / PYTHON.
- `v5` for GPU with N < 256 or N not a multiple of 128.
- `v10` for GPU with large, block-aligned N.

## Bench

```
DEV=NV  .venv/bin/python extra/cholesky/bench.py
DEV=NV  VER=v5,v10,cusolver N=4096  .venv/bin/python extra/cholesky/bench.py
DEV=CPU VER=v1,numpy                 .venv/bin/python extra/cholesky/bench.py
JIT=0   DEV=NV VER=v7                 .venv/bin/python extra/cholesky/bench.py
```

Bench wraps each variant in `TinyJit` by default (set `JIT=0` to disable).
`VER=numpy` runs `np.linalg.cholesky`, `VER=cusolver` runs
`torch.linalg.cholesky` on CUDA — that's cuSOLVER's `cusolverDnSpotrf` under
the hood and is our dense-fp32 baseline on NVIDIA GPUs.

### Reference numbers (NVIDIA, fp32, with TinyJit)

|         N | v1     | v5     | v7     | v9     | v10    | cuSOLVER | numpy |
|----------:|-------:|-------:|-------:|-------:|-------:|---------:|------:|
|        64 | 2.2 ms |   0.2 ms |   0.7 ms |   0.2 ms |   0.2 ms |   0.04 ms |  0.01 ms |
|       256 |  160 ms |   1.7 ms |  10 ms |   1.7 ms |   1.7 ms |   0.13 ms |  0.3 ms |
|      1024 |    — |  71 ms |  42 ms |  10 ms |   3.1 ms |   0.49 ms | 11.6 ms |
|      4096 |    — |    — | 230 ms |  53 ms |  23 ms |   2.1 ms | 343 ms |
|      8192 |    — |    — |    — |    — | 107 ms |   7.9 ms |    — |

"—" means the variant is too slow to be practical at that size.

Summary:
- v1 → v5 is ~100× (same-structure, single-kernel parallelisation).
- v5 → v10 is ~20× at N=4096 (multi-kernel blocking + tinygrad GEMMs).
- v10 is still ~15× slower than cuSOLVER, which uses tensor-core cuBLAS GEMMs
  and hand-tuned LAPACK-style blocking. tinygrad's default GEMMs do ~2 TFLOPS
  on these shapes vs cuBLAS' ~20 TFLOPS with tensor cores.

## Tests

```
DEV=NV  .venv/bin/python -m unittest extra.cholesky.test_cholesky -v
DEV=CPU .venv/bin/python -m unittest extra.cholesky.test_cholesky -v
```

GPU-only variants skip cleanly on backends without LOCAL memory support.

## Design notes

- **AxisType.REDUCE for every sequential axis.** It's the only axis type that
  tinygrad's `convert_loop_to_global` leaves as a serial for-loop; `LOOP`
  axes that appear in a sink-level `END` silently become parallel block
  dims and break the algorithm.
- **REG placeholders for per-thread state, LOCAL + `UOp.barrier` for
  cross-thread fan-in.** Same pattern as `extra/gemm/amd_uop_matmul.py`.
- **Two-pass write in v4/v5.** The diagonal thread writes `L[J,J]`; a
  barrier; every thread writes its own row's `L[i,J]`. Avoids gated stores
  on tinygrad's INDEX, which can't carry a bool gate through 2-D indexing.
- **Multi-kernel is Python-loop building a lazy UOp graph.** Inside
  `blocked_cholesky`, `for I in range(NB): L = ... (tinygrad ops) ...`
  never calls `.realize()`. The entire graph is scheduled together; with
  `TinyJit` the Python loop runs once and subsequent calls hit the compiled
  dispatch table.
- **Edge cases hit while building this:**
  - Dynamic `UOp.range(j, …)` bounds crash `simplify_merge_adjacent` on
    same-type adjacent axes (the merged bound references the substituted
    range → cycle → `KeyError`).
  - `Opt(OptOps.UNROLL, k)` on a scalar REG accumulator emits invalid C
    (`*(float4){…} = …`). UNROLL needs a vector-shaped accumulator.
  - `tag=1` on the sink bypasses `apply_opts` but is rejected by
    `tensor_spec` ("no tags allowed in tensor graph").
  - `Ops.REDUCE` with a trivial range (N=1) trips `reduce_unparented`'s
    `all(x.op is Ops.RANGE for x in ...)` assertion — special-cased in
    v2/v3/v4/v5 with a direct sqrt.
