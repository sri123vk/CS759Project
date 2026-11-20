#!/usr/bin/env python3
"""Figure and tables for the compiler-vectorization study.

Usage: plot_vec.py VEC_DIR [-o PLOTS_DIR]

Reads the output of scripts/vec_study.sh (<build>/omp.csv for each build in
builds.txt, the GCC vectorization reports and isa.txt) and writes, into
PLOTS_DIR (default: plots/<VEC_DIR name>/):

  fig_vec_speedup.png   speedup of each build over the first (reference)
                        build, per implementation, at the largest image
                        size; one panel per filter size and thread count
  vec_summary.md        the same as tables for every size, absolute times,
                        and which loops GCC vectorized at what width

Uses the colors and helpers of scripts/plot.py so the figure matches the
other report figures.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot as P  # noqa: E402  (shared style: colors, rcParams, save, md)

plt = P.plt


def builds_of(vec_dir):
    """Build names in order; the first is the reference."""
    f = vec_dir / "builds.txt"
    if f.exists():
        return f.read_text().split()
    return ["default", "native"]  # layout of the first version of vec_study.sh


def load(vec_dir, builds):
    frames = []
    for b in builds:
        f = vec_dir / b / "omp.csv"
        if not f.exists():
            sys.exit(f"missing {f}; run scripts/vec_study.sh first")
        frames.append(pd.read_csv(f).assign(build=b))
    df = pd.concat(frames, ignore_index=True)
    failed = df[df["pass"] != 1]
    # bench.sh runs seq once per (size, K) as the speedup baseline, so seq rows
    # have threads=1 only; the OpenMP rows have each thread count in THREADS.
    return df, failed


def speedup_table(df, builds):
    """Reference-build time / build time for every (impl, threads, H, K, build)."""
    keys = ["impl", "threads", "H", "W", "K"]
    t = df.pivot_table(index=keys, columns="build", values="median_ms", aggfunc="median")
    t = t.dropna(subset=builds)
    ref = builds[0]
    long = []
    for b in builds[1:]:
        long.append(pd.DataFrame({"build": b, "speedup": t[ref] / t[b]}, index=t.index))
    return pd.concat(long).reset_index()


def fig_speedup(sp, builds, out, made):
    H = sp.H.max()
    d = sp[sp.H == H]
    threads = sorted(d[d.impl != "seq"].threads.unique())
    ks = sorted(d.K.unique())
    impls = sorted(d.impl.unique(), key=lambda n: (n != "seq", n))
    others = builds[1:]
    # Builds are told apart by shade: lighter = narrower vectors.
    shades = np.linspace(0.45, 0.9, len(others)) if len(others) > 1 else [0.7]
    colors = [plt.cm.Greys(s) for s in shades]

    fig, axes = plt.subplots(len(threads), len(ks), figsize=(3.6 * len(ks), 3.0 * len(threads)),
                             sharey=True, squeeze=False)
    width = 0.8 / len(others)
    for r, t in enumerate(threads):
        for c, k in enumerate(ks):
            ax = axes[r][c]
            # seq is single-threaded; show it in every panel as the reference.
            dd = d[(d.K == k) & ((d.threads == t) | (d.impl == "seq"))]
            for n, b in enumerate(others):
                y = [dd[(dd.impl == i) & (dd.build == b)].speedup.median() for i in impls]
                x = np.arange(len(impls)) + (n - (len(others) - 1) / 2) * width
                ax.bar(x, y, width * 0.92, color=colors[n], label=b if (r, c) == (0, 0) else None)
            ax.axhline(1.0, color=P.INK2, lw=1, ls="--")
            ax.set_xticks(np.arange(len(impls)), impls, rotation=35, ha="right")
            ax.set_title(f"K={k}, {t} thread" + ("s" if t > 1 else ""))
            ax.grid(axis="x", visible=False)
        axes[r][0].set_ylabel(f"speedup over {builds[0]}")
    fig.suptitle(f"{H}×{H}: speedup of each build over the {builds[0]} build")
    P.legend_below(fig, axes)
    P.save(fig, out, "fig_vec_speedup.png", made)


WIDTH_RE = re.compile(r"^(src/[^:]+):(\d+):\d+: optimized: (loop|basic block part) vectorized using (\d+) byte vectors")


def vec_report(vec_dir, builds):
    """Widest vector used per source line, per build, plus in-order flags."""
    rows = {}
    for b in builds:
        f = vec_dir / f"vec_{b}.txt"
        for line in f.read_text().splitlines() if f.exists() else []:
            m = WIDTH_RE.match(line)
            if not m:
                continue
            key = (m.group(1)[len("src/"):], int(m.group(2)), m.group(3))
            w = int(m.group(4))
            rows.setdefault(key, {})
            rows[key][b] = max(rows[key].get(b, 0), w)
        inorder = vec_dir / f"vec_{b}_inorder.txt"
        for line in inorder.read_text().splitlines() if inorder.exists() else []:
            fname, ln = line.split(":")
            for key in rows:
                if key[0] == fname[len("src/"):] and key[1] == int(ln):
                    rows[key][f"{b}_inorder"] = True

    def fmt(r, b):
        if b not in r:
            return "no"
        s = f"{r[b]} B ({r[b] // 4} floats)"
        return s + ", in-order sum" if r.get(f"{b}_inorder") else s

    lines = ["| source:line | what | " + " | ".join(builds) + " |",
             "|---|---|" + "---|" * len(builds)]
    for (f, ln, what), r in sorted(rows.items()):
        lines.append(f"| {f}:{ln} | {what} | " + " | ".join(fmt(r, b) for b in builds) + " |")
    return "\n".join(lines)


def summary(df, sp, failed, builds, vec_dir, out):
    p = ["# Compiler vectorization study", ""]
    isa = vec_dir / "isa.txt"
    if isa.exists():
        p += ["Instruction sets each build enables on the benchmark machine:", "", "```",
              isa.read_text().strip(), "```", ""]
    if not failed.empty:
        p += ["**Correctness failures (exclude these rows):**", "",
              failed[["build", "impl", "threads", "H", "K", "err_ratio"]].to_string(index=False), ""]

    sp = sp.assign(workload=sp.H.astype(str) + "², K=" + sp.K.astype(str))
    for b in builds[1:]:
        for t in sorted(sp[sp.impl != "seq"].threads.unique()):
            d = sp[(sp.build == b) & ((sp.threads == t) | (sp.impl == "seq"))]
            tab = d.pivot_table(index="workload", columns="impl", values="speedup", aggfunc="median")
            tab = tab.reindex(sorted(tab.index, key=lambda s: (int(s.split("²")[0]), s)))
            p += [f"## Speedup of `{b}` over `{builds[0]}`, {t} thread(s)", "",
                  f"Above 1 means the {b} build is faster. seq is always single-threaded.", "",
                  P.md(tab, "{:.2f}"), ""]

    H = df.H.max()
    d = df[(df.H == H)]
    tab = d.assign(col=d.impl + " t" + d.threads.astype(str), cfg="K=" + d.K.astype(str)) \
        .pivot_table(index=["build", "cfg"], columns="col", values="median_ms", aggfunc="median")
    tab = tab.reindex(sorted(tab.index, key=lambda bc: (builds.index(bc[0]), bc[1])))
    tab.index = [f"{b}, {c}" for b, c in tab.index]
    tab.index.name = "build"
    p += [f"## Median time (ms) at {H}×{H}", "", P.md(tab, "{:.2f}"), ""]

    p += ["## Loops GCC vectorized (`-fopt-info-vec-optimized`)", "",
          "\"in-order sum\": GCC vectorized only the multiplies; the additions into the",
          "single float accumulator still run one at a time (fold-left reduction), because",
          "reordering them would change the rounding without `-ffast-math`. Such a loop",
          "gains little, and it only runs K times per pixel here.", "",
          vec_report(vec_dir, builds), ""]
    (out / "vec_summary.md").write_text("\n".join(p) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vec_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path("plots") / a.vec_dir.name
    out.mkdir(parents=True, exist_ok=True)

    builds = builds_of(a.vec_dir)
    df, failed = load(a.vec_dir, builds)
    sp = speedup_table(df, builds)
    made = []
    fig_speedup(sp, builds, out, made)
    summary(df, sp, failed, builds, a.vec_dir, out)
    print(f"{out}: " + ", ".join(made + ["vec_summary.md"]))


if __name__ == "__main__":
    main()
