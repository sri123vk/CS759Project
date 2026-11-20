#!/usr/bin/env python3
"""Figure and tables for the compiler-vectorization study.

Usage: plot_vec.py VEC_DIR [-o PLOTS_DIR]

Reads the output of scripts/vec_study.sh (default/omp.csv, native/omp.csv,
the GCC vectorization reports and isa.txt) and writes, into PLOTS_DIR
(default: plots/<VEC_DIR name>/):

  fig_vec_speedup.png   speedup of the -march=native build over the default
                        build, per implementation and filter size, at the
                        largest image size, on one thread and on all threads
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
BUILDS = ["default", "native"]


def load(vec_dir):
    frames = []
    for b in BUILDS:
        f = vec_dir / b / "omp.csv"
        if not f.exists():
            sys.exit(f"missing {f}; run scripts/vec_study.sh first")
        frames.append(pd.read_csv(f).assign(build=b))
    df = pd.concat(frames, ignore_index=True)
    failed = df[df["pass"] != 1]
    # bench.sh runs seq once per (size, K) as the speedup baseline, so seq rows
    # have threads=1 only; the OpenMP rows have each thread count in THREADS.
    return df, failed


def speedup_table(df):
    """native/default time ratio for every (impl, threads, H, K)."""
    keys = ["impl", "threads", "H", "W", "K"]
    t = df.pivot_table(index=keys, columns="build", values="median_ms", aggfunc="median")
    t = t.dropna(subset=BUILDS)
    t["speedup"] = t["default"] / t["native"]
    return t.reset_index()


def fig_speedup(sp, out, made):
    H = sp.H.max()
    d = sp[sp.H == H]
    threads = sorted(d[d.impl != "seq"].threads.unique())
    panels = [(t, f"{t} thread" + ("s" if t > 1 else "")) for t in threads]
    impls = [i for i in sorted(d.impl.unique(), key=lambda n: (n != "seq", n))]
    P.set_styles(impls)
    ks = sorted(d.K.unique())

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8),
                             sharey=True, squeeze=False)
    width = 0.8 / len(impls)
    for ax, (t, title) in zip(axes[0], panels):
        for n, impl in enumerate(impls):
            # seq is single-threaded; show it in every panel as the reference.
            rows = d[(d.impl == impl) & ((d.threads == t) | (d.impl == "seq"))]
            y = [rows[rows.K == k].speedup.median() for k in ks]
            x = np.arange(len(ks)) + (n - (len(impls) - 1) / 2) * width
            ax.bar(x, y, width * 0.92, color=P.STYLE[impl]["color"], label=impl)
        ax.axhline(1.0, color=P.INK2, lw=1, ls="--")
        ax.set_xticks(np.arange(len(ks)), [f"K={k}" for k in ks])
        ax.set_title(f"{H}×{H}, {title}")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("speedup of -march=native\nover default flags")
    P.legend_below(fig, axes)
    P.save(fig, out, "fig_vec_speedup.png", made)


WIDTH_RE = re.compile(r"^(src/[^:]+):(\d+):\d+: optimized: (loop|basic block part) vectorized using (\d+) byte vectors")


def vec_report(vec_dir):
    """Widest vector used per source line, per build, plus in-order flags."""
    rows = {}
    for b in BUILDS:
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

    lines = ["| source:line | what | default | -march=native |", "|---|---|---|---|"]
    for (f, ln, what), r in sorted(rows.items()):
        lines.append(f"| {f}:{ln} | {what} | {fmt(r, 'default')} | {fmt(r, 'native')} |")
    return "\n".join(lines)


def summary(df, sp, failed, vec_dir, out):
    p = ["# Compiler vectorization study", ""]
    isa = vec_dir / "isa.txt"
    if isa.exists():
        p += ["What `-march=native` enables on the benchmark machine:", "", "```",
              isa.read_text().strip(), "```", ""]
    if not failed.empty:
        p += ["**Correctness failures (exclude these rows):**", "",
              failed[["build", "impl", "threads", "H", "K", "err_ratio"]].to_string(index=False), ""]

    sp = sp.assign(workload=sp.H.astype(str) + "², K=" + sp.K.astype(str))
    for t in sorted(sp[sp.impl != "seq"].threads.unique()):
        d = sp[(sp.threads == t) | (sp.impl == "seq")]
        tab = d.pivot_table(index="workload", columns="impl", values="speedup", aggfunc="median")
        tab = tab.reindex(sorted(tab.index, key=lambda s: (int(s.split("²")[0]), s)))
        p += [f"## Speedup of `-march=native` over default flags, {t} thread(s)", "",
              "Above 1 means the native build is faster. seq is always single-threaded.", "",
              P.md(tab, "{:.2f}"), ""]

    H = df.H.max()
    d = df[(df.H == H)]
    tab = d.assign(col=d.impl + " t" + d.threads.astype(str), cfg="K=" + d.K.astype(str)) \
        .pivot_table(index=["build", "cfg"], columns="col", values="median_ms", aggfunc="median")
    tab.index = [f"{b}, {c}" for b, c in tab.index]
    tab.index.name = "build"
    p += [f"## Median time (ms) at {H}×{H}", "", P.md(tab, "{:.2f}"), ""]

    p += ["## Loops GCC vectorized (`-fopt-info-vec-optimized`)", "",
          "\"in-order sum\": GCC vectorized only the multiplies; the additions into the",
          "single float accumulator still run one at a time (fold-left reduction), because",
          "reordering them would change the rounding without `-ffast-math`. Such a loop",
          "gains little, and it only runs K times per pixel here.", "",
          vec_report(vec_dir), ""]
    (out / "vec_summary.md").write_text("\n".join(p) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("vec_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    out = a.out or Path("plots") / a.vec_dir.name
    out.mkdir(parents=True, exist_ok=True)

    df, failed = load(a.vec_dir)
    sp = speedup_table(df)
    made = []
    fig_speedup(sp, out, made)
    summary(df, sp, failed, a.vec_dir, out)
    print(f"{out}: " + ", ".join(made + ["vec_summary.md"]))


if __name__ == "__main__":
    main()
