#!/usr/bin/env python3
"""External reference for the report: PyTorch conv2d on the same problems.

Writes rows in the same format as scripts/bench.sh ("threads,schedule," +
the bin/conv columns), as impl "torch_cudnn" (cuDNN on the GPU) and
"torch_cpu" (PyTorch's CPU convolution, oneDNN), so plot.py can show them
next to our implementations. Requested by the proposal review: "compare
your best GPU kernel against an optimized library such as cuDNN (or
PyTorch's conv2d), so your results have an external reference point".

Same operation as our spec: F.conv2d is cross-correlation; padding=K//2 gives
zero padding and a same-size output. Single channel, batch 1.

Fairness settings:
  * TF32 disabled (on Ampere+ GPUs cuDNN would otherwise use reduced-
    precision tensor cores): all arithmetic is FP32 like ours.
  * cudnn.benchmark = True: cuDNN times its algorithms on the first call for
    each shape and keeps the fastest; the warm-up runs absorb that.
  * Stages mirror bin/conv's CUDA timing: h2d = image + filter copied to the
    device, kernel = conv2d only (CUDA events), d2h = output copied back;
    median_ms = host wall-clock for all three. That includes some Python
    overhead (tens of microseconds), so compare small problems on kernel_ms.
  * Device buffers are allocated before timing (alloc_ms), as in bin/conv.
  * cuDNN may choose Winograd/FFT algorithms whose rounding error exceeds the
    dot-product bound used by --check; err_ratio is still reported, and a
    value > 1 then means "different algorithm", not "wrong answer".

Usage:
  bench_torch.py --sizes 256 512 1024 2048 --ks 3 5 7 [--device cuda cpu]
                 [--threads 16] [--iters N] [--warmup 2] [--out file.csv] [--no-check]
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("bench_torch.py: PyTorch is not installed (pip install torch)")

# bench.sh's columns: "threads,schedule," + bin/conv's header. If the output
# file already has a header (e.g. the driver gained columns), that one is used.
DEFAULT_COLUMNS = ("threads,schedule,impl,H,W,K,bx,by,iters,median_ms,min_ms,"
                   "h2d_ms,kernel_ms,d2h_ms,alloc_ms,"
                   "max_abs_err,max_rel_err,err_ratio,nonfinite,pass").split(",")


def iters_for(n, fixed):
    """Same default as bench.sh: more samples for short runs."""
    if fixed:
        return fixed
    return 50 if n <= 512 else 20 if n <= 1024 else 10


def error_stats(got, x, w, K):
    """Error vs a float64 reference, normalized by the FP32 dot-product bound
    2*gamma(K*K)*sum|terms| (same criterion as bin/conv --check)."""
    xd, wd = x.double(), w.double()
    ref = F.conv2d(xd, wd, padding=K // 2)
    scale = F.conv2d(xd.abs(), wd.abs(), padding=K // 2)
    u = 2.0 ** -24
    n = K * K
    gamma = n * u / (1 - n * u)
    got = got.double()
    nonfinite = int((~torch.isfinite(got)).sum())
    d = (got - ref).abs()
    bound = 2 * gamma * scale
    ratio = torch.where(bound > 0, d / bound,
                        torch.where(d > 0, torch.full_like(d, float("inf")), torch.zeros_like(d)))
    mag = ref.abs()
    rel = d[mag > 1e-3] / mag[mag > 1e-3]
    err_ratio = float(ratio.max())
    return {"max_abs_err": f"{float(d.max()):.3e}",
            "max_rel_err": f"{float(rel.max()) if rel.numel() else 0.0:.3e}",
            "err_ratio": f"{err_ratio:.3e}", "nonfinite": nonfinite,
            "pass": int(nonfinite == 0 and err_ratio <= 1)}


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def bench_cuda(x, w, K, warmup, iters):
    t0 = time.perf_counter()
    x_d = torch.empty_like(x, device="cuda")
    w_d = torch.empty_like(w, device="cuda")
    out_h = torch.empty((1, 1, x.shape[2], x.shape[3]))
    torch.cuda.synchronize()
    alloc_ms = (time.perf_counter() - t0) * 1e3
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    stages = []

    def once():
        ev[0].record()
        x_d.copy_(x)
        w_d.copy_(w)
        ev[1].record()
        y = F.conv2d(x_d, w_d, padding=K // 2)
        ev[2].record()
        out_h.copy_(y)
        ev[3].record()
        ev[3].synchronize()
        stages.append((ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]),
                       ev[2].elapsed_time(ev[3])))

    times = timed(once, warmup, iters)
    st = np.median(np.array(stages[-len(times):]), axis=0)
    return times, {"h2d_ms": f"{st[0]:.4f}", "kernel_ms": f"{st[1]:.4f}",
                   "d2h_ms": f"{st[2]:.4f}", "alloc_ms": f"{alloc_ms:.4f}"}, out_h


def bench_cpu(x, w, K, threads, warmup, iters):
    torch.set_num_threads(threads)
    out = {}

    def once():
        out["y"] = F.conv2d(x, w, padding=K // 2)

    return timed(once, warmup, iters), out["y"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--device", nargs="+", default=["cuda", "cpu"], choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=0, help="default: same rule as bench.sh")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", help="append rows to this CSV (header written if new)")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    devices = [d for d in a.device if d == "cpu" or torch.cuda.is_available()]
    if "cuda" in a.device and "cuda" not in devices:
        print("bench_torch.py: CUDA not available; CPU only", file=sys.stderr)

    columns = DEFAULT_COLUMNS
    if a.out and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out) as f:
            columns = f.readline().strip().split(",")
        f = open(a.out, "a")
    else:
        f = open(a.out, "w") if a.out else sys.stdout
        print(",".join(columns), file=f, flush=True)

    rng = np.random.default_rng(759)
    for n in a.sizes:
        for K in a.ks:
            x = torch.from_numpy(rng.uniform(-1, 1, (1, 1, n, n)).astype(np.float32))
            w = torch.from_numpy(rng.uniform(-1, 1, (1, 1, K, K)).astype(np.float32))
            iters = iters_for(n, a.iters)
            runs = []
            if "cuda" in devices:
                times, st, y = bench_cuda(x, w, K, a.warmup, iters)
                runs.append(({"impl": "torch_cudnn", **st}, times, y))
            if "cpu" in devices:
                times, y = bench_cpu(x, w, K, a.threads, a.warmup, iters)
                runs.append(({"impl": "torch_cpu", "threads": a.threads}, times, y))
            for info, times, y in runs:
                r = {"H": n, "W": n, "K": K, "iters": len(times),
                     "median_ms": f"{np.median(times):.4f}", "min_ms": f"{min(times):.4f}", **info}
                if not a.no_check:
                    r.update(error_stats(y, x, w, K))
                print(",".join(str(r.get(c, "")) for c in columns), file=f, flush=True)
                print(f"  {info['impl']:12s} {n}x{n} K={K} {np.median(times):.3f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
