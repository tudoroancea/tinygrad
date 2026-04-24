"""Benchmark driver for codegenerable Cholesky variants.

Usage:
  DEV=CPU .venv/bin/python extra/cholesky/bench.py
  DEV=NV  N=512  VER=v2  .venv/bin/python extra/cholesky/bench.py

Timing uses GlobalCounters.time_sum_s from tinygrad's profiler, which measures
just the kernel time (not Python overhead or dispatch).
"""
from __future__ import annotations
import os, importlib, time, numpy as np
from tinygrad import Tensor, GlobalCounters, Context, TinyJit
from tinygrad.device import Device

VERSIONS = ("v1", "v3", "v4", "v5", "v7", "v8", "v9", "v10", "cusolver", "numpy")
SIZES_DEFAULT = (64, 128, 256, 512, 1024)


def spd(N: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  X = rng.standard_normal((N, N)).astype(np.float32)
  return X @ X.T + N * np.eye(N, dtype=np.float32)


def load_variant(name: str):
  mod = importlib.import_module(f"cholesky_{name}")
  return mod.cholesky


MAX_N = {"v1": 256, "v2": 256}  # single-thread variants are unusably slow above this


def _run_numpy(sizes, warmup: int = 1, iters: int = 3):
  for N in sizes:
    A_np = spd(N)
    for _ in range(warmup): np.linalg.cholesky(A_np)
    ets = []
    for _ in range(iters):
      t0 = time.perf_counter(); np.linalg.cholesky(A_np); t1 = time.perf_counter()
      ets.append(t1 - t0)
    flops = (N**3) / 3.0
    best = min(ets)
    gflops = flops / best / 1e9 if best > 0 else float("nan")
    print(f"[NPY] numpy N={N:5d}  t={best*1e3:8.2f} ms  {gflops:8.2f} GFLOPS")


def _run_cusolver(sizes, warmup: int = 1, iters: int = 3):
  try:
    import torch
    assert torch.cuda.is_available()
  except Exception as e:
    print(f"(skipping cusolver: {e})")
    return
  for N in sizes:
    A_np = spd(N)
    A_cuda = torch.from_numpy(A_np).cuda()
    for _ in range(warmup): torch.linalg.cholesky(A_cuda)
    torch.cuda.synchronize()
    ets = []
    for _ in range(iters):
      t0 = time.perf_counter()
      L = torch.linalg.cholesky(A_cuda)
      torch.cuda.synchronize()
      ets.append(time.perf_counter() - t0)
    flops = (N**3) / 3.0
    best = min(ets)
    gflops = flops / best / 1e9 if best > 0 else float("nan")
    print(f"[cuSOLVER] N={N:5d}  t={best*1e3:8.2f} ms  {gflops:8.2f} GFLOPS")


def run_variant(name: str, sizes, warmup: int = 1, iters: int = 3):
  if name == "numpy":
    return _run_numpy(sizes, warmup, iters)
  if name == "cusolver":
    return _run_cusolver(sizes, warmup, iters)
  use_jit = bool(int(os.environ.get("JIT", "1")))
  chol = load_variant(name)
  limit = MAX_N.get(name, 10**9)
  for N in sizes:
    if N > limit:
      print(f"[{os.environ.get('DEV', '?')}] {name}  N={N:5d}  (skipped: above {limit})")
      continue
    A_np = spd(N)
    ref = np.linalg.cholesky(A_np)

    ets: list[float] = []
    last_L = None
    # Build a JIT wrapper so per-call Python dispatch doesn't dominate for blocked variants.
    A_fixed = Tensor(A_np).realize()
    if use_jit:
      jitted = TinyJit(lambda a: chol(a).realize())
      for _ in range(max(2, warmup)):
        jitted(A_fixed); Device[A_fixed.device].synchronize()
      for _ in range(iters):
        t0 = time.perf_counter()
        last_L = jitted(A_fixed)
        Device[A_fixed.device].synchronize()
        ets.append(time.perf_counter() - t0)
    else:
      for i in range(warmup + iters):
        A = Tensor(A_np).realize()
        Device[A.device].synchronize()
        t0 = time.perf_counter()
        with Context(DEBUG=0):
          L = chol(A).realize()
        Device[A.device].synchronize()
        t1 = time.perf_counter()
        if i >= warmup: ets.append(t1 - t0)
        last_L = L
    L_np = last_L.numpy()
    err = float(np.abs(L_np - ref).max())

    flops = (N**3) / 3.0
    best = min(ets)
    gflops = flops / best / 1e9 if best > 0 else float("nan")
    tag = os.environ.get("DEV", A_fixed.device)
    print(f"[{tag}] {name}  N={N:5d}  t={best*1e3:8.2f} ms  {gflops:8.2f} GFLOPS  err={err:.2e}")


if __name__ == "__main__":
  import sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  sizes_env = os.environ.get("N")
  sizes = (int(sizes_env),) if sizes_env else SIZES_DEFAULT
  versions = os.environ.get("VER", ",".join(VERSIONS)).split(",")
  for v in versions:
    try:
      run_variant(v, sizes)
    except ModuleNotFoundError:
      print(f"(skipping {v}: module not present)")
