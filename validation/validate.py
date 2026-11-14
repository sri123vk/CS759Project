#!/usr/bin/env python3
"""External correctness check: compare bin/conv outputs against SciPy (or PyTorch).

Usage:
  validate.py DUMP_DIR [--ref auto|scipy|torch] [-v]

DUMP_DIR holds files written by `bin/conv ... --dump DUMP_DIR`:
  {H}x{W}_k{K}_in.bin           input image, H*W float32, row-major
  {H}x{W}_k{K}_k.bin            filter, K*K float32, row-major
  {H}x{W}_k{K}_{impl}_out.bin   output of one implementation, H*W float32

Every *_out.bin is checked against a float64 reference computed by a library
we did not write:
  scipy  scipy.signal.correlate2d(in, k, mode="same", boundary="fill")
  torch  torch.nn.functional.conv2d(in, k, padding=K // 2) in float64
Both compute cross-correlation with zero padding and a same-size output,
which is our spec (README, "Specification"). `auto` uses SciPy if it is
installed, else PyTorch.

Pass criterion (same as bin/conv --check and tests/test_conv.cpp): no NaN or
Inf, and at every pixel |got - ref| <= 2 * gamma(K*K) * sum|k * in|, with
gamma(n) = n*u / (1 - n*u), u = 2^-24. err_ratio is the worst error divided by
that bound, so it must be <= 1.

Before checking any files, the reference itself is checked on small cases
with known answers. That catches a library computing true convolution
(flipped filter) or centering the "same" output differently than we do.

Exit status: 0 if every output passes, 1 if any fails, 2 on bad usage or
missing/malformed files.
"""
import argparse
import os
import re
import sys

import numpy as np

OUT_RE = re.compile(r"^(\d+)x(\d+)_k(\d+)_(.+)_out\.bin$")
U = 2.0 ** -24


# ---------------------------------------------------------------------------
# References: float64 cross-correlation, zero padding, same-size output.

def ref_scipy(img, k):
    from scipy.signal import correlate2d
    return correlate2d(img, k, mode="same", boundary="fill", fillvalue=0.0)


def ref_torch(img, k):
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64))[None, None]
    w = torch.from_numpy(np.ascontiguousarray(k, dtype=np.float64))[None, None]
    return F.conv2d(x, w, padding=k.shape[0] // 2)[0, 0].numpy()


def pick_reference(name):
    """Returns (label, function) for the requested reference library."""
    candidates = ["scipy", "torch"] if name == "auto" else [name]
    for c in candidates:
        try:
            if c == "scipy":
                import scipy.signal  # noqa: F401
                import scipy
                return f"scipy {scipy.__version__} signal.correlate2d", ref_scipy
            if c == "torch":
                import torch
                return f"torch {torch.__version__} nn.functional.conv2d", ref_torch
        except ImportError:
            continue
    sys.exit(f"validate.py: no reference library available (tried {', '.join(candidates)}); "
             "pip install scipy (or torch)")


def check_reference(ref):
    """Checks the library against answers that follow directly from the spec."""
    rng = np.random.default_rng(0)

    def fail(what):
        sys.exit(f"validate.py: reference library disagrees with our spec: {what}")

    # A single tap at (i0, j0) shifts the image: out[r][c] = in[r+i0-R][c+j0-R],
    # zero where that falls outside. An asymmetric tap distinguishes
    # correlation from convolution, and a non-square image catches swapped axes.
    H, W = 6, 9
    img = rng.uniform(-1, 1, (H, W))
    for K in (3, 5):
        R = K // 2
        for i0 in range(K):
            for j0 in range(K):
                k = np.zeros((K, K))
                k[i0, j0] = 1.0
                want = np.zeros((H, W))
                for r in range(H):
                    for c in range(W):
                        y, x = r + i0 - R, c + j0 - R
                        if 0 <= y < H and 0 <= x < W:
                            want[r, c] = img[y, x]
                if not np.array_equal(ref(img, k), want):
                    fail(f"single tap at ({i0},{j0}), K={K}, image {H}x{W}")

    # Random filter against a direct loop over the spec's formula.
    for H, W, K in ((7, 5, 3), (11, 13, 7)):
        img = rng.uniform(-1, 1, (H, W))
        k = rng.uniform(-1, 1, (K, K))
        R = K // 2
        want = np.zeros((H, W))
        for r in range(H):
            for c in range(W):
                for i in range(K):
                    for j in range(K):
                        y, x = r + i - R, c + j - R
                        if 0 <= y < H and 0 <= x < W:
                            want[r, c] += k[i, j] * img[y, x]
        if not np.allclose(ref(img, k), want, rtol=0, atol=1e-12):
            fail(f"random filter, K={K}, image {H}x{W}")


# ---------------------------------------------------------------------------
# Comparing one output

def load(path, count):
    """Reads exactly `count` float32 values, or returns an error string."""
    if not os.path.exists(path):
        return None, f"missing {os.path.basename(path)}"
    size = os.path.getsize(path)
    if size != count * 4:
        return None, f"{os.path.basename(path)} is {size} bytes, expected {count * 4}"
    return np.fromfile(path, dtype=np.float32, count=count), None


def error_stats(got, ref, scale, K):
    """Mirrors compare() in include/utils.h."""
    n = K * K
    gamma = n * U / (1.0 - n * U)
    got = got.astype(np.float64)
    finite = np.isfinite(got) & np.isfinite(ref)
    nonfinite = int(np.count_nonzero(~finite))
    d = np.where(finite, np.abs(got - ref), 0.0)
    bound = 2.0 * gamma * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bound > 0, d / bound, np.where(d > 0, np.inf, 0.0))
    mag = np.abs(ref)
    big = finite & (mag > 1e-3)
    # Point at a NaN/Inf if there is one, since those fail regardless of ratio.
    worst_flat = int(np.argmax(~finite)) if nonfinite else int(np.argmax(ratio))
    worst = np.unravel_index(worst_flat, ratio.shape)
    return {
        "err_ratio": float(ratio.max()),
        "max_abs": float(d.max()),
        "max_rel": float((d[big] / mag[big]).max()) if big.any() else 0.0,
        "nonfinite": nonfinite,
        "worst": worst,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Check bin/conv --dump outputs against SciPy or PyTorch.")
    ap.add_argument("dump_dir")
    ap.add_argument("--ref", choices=["auto", "scipy", "torch"], default="auto")
    ap.add_argument("-v", action="store_true", help="also print the worst pixel of passing outputs")
    a = ap.parse_args()

    if not os.path.isdir(a.dump_dir):
        print(f"validate.py: {a.dump_dir} is not a directory", file=sys.stderr)
        return 2
    outs = sorted(f for f in os.listdir(a.dump_dir) if OUT_RE.match(f))
    if not outs:
        print(f"validate.py: no *_out.bin files in {a.dump_dir} "
              "(write them with bin/conv ... --dump DIR)", file=sys.stderr)
        return 2

    label, ref = pick_reference(a.ref)
    check_reference(ref)
    print(f"reference: {label} (float64), checked against known answers")
    print(f"{'output':40s} {'err_ratio':>10s} {'max_abs':>10s} {'max_rel':>10s} {'nonfinite':>9s}  result")

    cache = {}  # (H, W, K) -> (ref, scale); several implementations share inputs
    n_fail = n_bad = 0
    for name in outs:
        H, W, K, impl = OUT_RE.match(name).groups()
        H, W, K = int(H), int(W), int(K)
        prefix = os.path.join(a.dump_dir, f"{H}x{W}_k{K}_")

        if (H, W, K) not in cache:
            img, err = load(prefix + "in.bin", H * W)
            k, err2 = load(prefix + "k.bin", K * K) if err is None else (None, None)
            if err or err2:
                print(f"{name:40s} ERROR: {err or err2}")
                n_bad += 1
                continue
            img = img.reshape(H, W).astype(np.float64)
            k = k.reshape(K, K).astype(np.float64)
            cache[(H, W, K)] = (ref(img, k), ref(np.abs(img), np.abs(k)))
        want, scale = cache[(H, W, K)]

        got, err = load(os.path.join(a.dump_dir, name), H * W)
        if err:
            print(f"{name:40s} ERROR: {err}")
            n_bad += 1
            continue
        e = error_stats(got.reshape(H, W), want, scale, K)
        ok = e["nonfinite"] == 0 and e["err_ratio"] <= 1.0
        n_fail += not ok
        print(f"{name:40s} {e['err_ratio']:10.3e} {e['max_abs']:10.3e} {e['max_rel']:10.3e} "
              f"{e['nonfinite']:9d}  {'ok' if ok else 'FAIL'}")
        if not ok or a.v:
            r, c = e["worst"]
            print(f"{'':40s} worst pixel (r={r}, c={c}): got {got[r * W + c]:.9g}, "
                  f"want {want[r, c]:.9g}")

    print()
    if n_bad:
        print(f"{n_bad} output(s) could not be checked")
    print(f"{len(outs) - n_fail - n_bad} of {len(outs)} passed")
    return 2 if n_bad else 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
