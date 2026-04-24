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
| `cholesky_v11.py` | v10 + Tensor.pad-based L assembly | replaces ~240 zero-fill kernels with pure movement ops; ~7-10% faster |

All multi-kernel variants fall back to v5 when N is too small or not a
multiple of the block size.

## Dispatcher

`extra/cholesky/__init__.py` exports `cholesky(A)` which picks:

- `v1` for CPU / PYTHON.
- `v5` for GPU with N < 256 or N not a multiple of 64.
- `v11` for GPU with large, block-aligned N.

## Bench

```
DEV=NV TC=1 BEAM=2 .venv/bin/python extra/cholesky/bench.py
DEV=NV TC=1 BEAM=2 VER=v5,v11,cusolver N=4096 .venv/bin/python extra/cholesky/bench.py
DEV=CPU VER=v1,numpy                          .venv/bin/python extra/cholesky/bench.py
JIT=0  DEV=NV VER=v7                          .venv/bin/python extra/cholesky/bench.py
```

Bench wraps each variant in `TinyJit` by default (set `JIT=0` to disable).
`TC=1 BEAM=2` lets tinygrad's beam search pick optimised GEMM/TRSM kernels
for the trailing-update matmuls — the first run takes a few minutes for
the search, subsequent ones hit the diskcache.

`VER=numpy` runs `np.linalg.cholesky`, `VER=cusolver` runs
`torch.linalg.cholesky` on CUDA — that's cuSOLVER's `cusolverDnSpotrf` under
the hood and is our dense-fp32 baseline on NVIDIA GPUs.

### Reference numbers (NVIDIA RTX 5090, fp32, TinyJit + `TC=1 BEAM=2`)

|        N | v5      | v10     | v11     | cuSOLVER | gap to cuSOLVER |
|---------:|--------:|--------:|--------:|---------:|-----:|
|     1024 |   71 ms |  3.1 ms |  1.5 ms |  0.49 ms |  3.1× |
|     2048 |  428 ms |  7.4 ms |  3.6 ms |  1.01 ms |  3.5× |
|     4096 | 3250 ms |   23 ms | 13.6 ms |  2.14 ms |  6.4× |
|     8192 |    —    |  107 ms |   81 ms |  7.88 ms | 10.3× |

Throughput at N=4096:  v5 ~7 GFLOPS  →  v10 ~1 TFLOPS  →  v11 ~1.7 TFLOPS  →  cuSOLVER ~10.7 TFLOPS.

Summary of the ladder of speedups vs the trivial v1 single-thread Cholesky:
- v1 → v5: ~100× (single-kernel parallelisation, 2-D thread tile).
- v5 → v11: ~240× at N=4096 (multi-kernel blocking, recursive triangular
  inverse, pad-based assembly, TinyJit + CUDA graphs).
- v11 → cuSOLVER: ~6× at N=4096. The gap is the TF32 tensor cores: cuSOLVER
  does its trailing-update matmuls on TF32 TC at >50 TFLOPS, our tinygrad
  matmuls hit ~7 TFLOPS pure fp32. ALLOW_TF32=1 puts TF32 TC into BEAM's
  action set, but BEAM's wall-clock budget hasn't picked it as a winner here.

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
  dispatch table — and tinygrad batches consecutive kernels into CUDA
  graphs automatically (you can see `JIT GRAPHing batch with N kernels`
  in the debug log).
- **BEAM works on tinygrad-emitted matmuls but not on our custom kernels.**
  `Tensor.__matmul__` produces an `Ops.REDUCE` AST that BEAM can re-tile
  with UPCAST/UNROLL/LOCAL/SWAP/THREAD/TC. Our custom Cholesky/TRSM kernels
  carry `opts_to_apply=()` to disable BEAM because BEAM's `GROUP` opt
  re-parallelises our REDUCE axes (which we use as sequential loops),
  silently breaking correctness. A BEAM-able Cholesky would need an
  algorithm that's actually reducible, which Cholesky's left/right-looking
  isn't — the j and i loops carry true dependencies.

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

## Where the remaining gap to cuSOLVER comes from

After v11 + BEAM, profiling at N=4096 (13.6 ms total GPU time) shows:
- big trailing-update matmuls reach ~7 TFLOPS (the BEAM-tiled `r_…` kernels
  hit close to fp32 peak on this GPU).
- Most time is in **the 64 small `cholesky_v5_64` calls** factoring the
  diagonal blocks (~2.7 ms) and **125 base-case 32×32 TRSM kernels** for
  the recursive inverse (~2.5 ms). These two groups together are >40% of
  GPU time and are barrier-bound, not compute-bound.
- cuSOLVER is faster mainly because cuBLAS-with-TF32 hits ~50 TFLOPS on the
  trailing matmuls (vs our 7) and the diagonal-block factor is fused
  in-shared-memory.
