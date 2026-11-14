#!/usr/bin/env python3
"""Figures and tables for the final report from one benchmark run.

Usage: plot.py RESULTS_DIR [-o PLOTS_DIR]

Reads the CSVs written by scripts/bench.sh (baseline, omp, cuda_blocks,
torch) and writes PNG figures plus summary.md (Markdown tables) into
PLOTS_DIR (default: plots/<run name>/). Experiments that weren't run are
skipped. Needs pandas and matplotlib.

  Figure                         Proposal item
  fig1_time.png                  execution time for each implementation, across image and filter sizes
  fig2_speedup.png               OpenMP and CUDA speedup relative to sequential (H1, H2)
  fig3_cuda_kernel_vs_e2e.png    CUDA kernel-only vs end-to-end speedup
  fig4_cuda_breakdown.png        where CUDA end-to-end time goes: h2d / kernel / d2h (H3)
  fig5_omp_scaling.png           OpenMP thread scaling, static vs dynamic (vs guided)
  fig6_cuda_blocks.png           CUDA thread-block configuration (H3)
  fig7_throughput.png            GFLOP/s vs the cuDNN / PyTorch reference
  fig8_fastest.png               fastest implementation per workload (stretch goal: dispatcher)
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Categorical colors in fixed order, assigned per implementation and kept
# the same in every figure; markers give a second, non-color cue.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
})

STYLE = {}


def set_styles(names):
    # seq first, then CPU parallel, then CUDA, then the external reference.
    def rank(n):
        return (0 if n.startswith("seq") else 1 if n.startswith("omp") else
                3 if n.startswith("torch") else 2, n)
    for i, n in enumerate(sorted(set(names), key=rank)):
        STYLE[n] = dict(color=SLOTS[i % 8], marker=MARKERS[i % 8],
                        linestyle="-" if i < 8 else "--")


def load(results):
    frames = []
    for f in sorted(Path(results).glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if "impl" in d.columns and not d.empty:
            frames.append(d.assign(experiment=f.stem))
    if not frames:
        sys.exit(f"no benchmark CSVs in {results}")
    df = pd.concat(frames, ignore_index=True)
    df["is_cuda"] = df.kernel_ms.notna()
    df["flops"] = 2.0 * df.K ** 2 * df.H * df.W
    # Compute-only time: the kernel for GPU rows, the whole call for CPU rows.
    df["compute_ms"] = df.kernel_ms.where(df.is_cuda, df.median_ms)
    df["gflops"] = df.flops / (df.compute_ms * 1e6)
    seq = df[df.impl == "seq"].groupby(["H", "W", "K"]).median_ms.min().rename("seq_ms")
    df = df.join(seq, on=["H", "W", "K"])
    df["speedup"] = df.seq_ms / df.median_ms
    df["speedup_kernel"] = df.seq_ms / df.compute_ms
    return df


def single_config(df):
    """One row per (impl, H, W, K): the default configuration of each
    implementation (OpenMP: most threads, static; CUDA: the default block)."""
    d = df[df.experiment.isin(["baseline", "torch"])]
    return d.sort_values("threads", na_position="first").groupby(["impl", "H", "W", "K"]).tail(1)


def size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}²" for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("image size")


def legend_below(fig, axes, extra_rows=0.0):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09 + extra_rows, 1, 1))


def save(fig, out, name, made):
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    made.append(name)


def per_k_lines(d, y, ylabel, out, name, made, hline=None, impls=None, logy=True):
    ks, sizes = sorted(d.K.unique()), sorted(d.H.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks), 3.6), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        for impl, g in d[d.K == k].groupby("impl"):
            if impls is not None and impl not in impls:
                continue
            g = g.sort_values("H")
            if g[y].notna().any():
                ax.plot(g.H, g[y], label=impl, **STYLE[impl])
        if hline is not None:
            ax.axhline(hline, color=INK2, lw=1, ls="--")
        size_axis(ax, sizes)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{k}×{k} filter")
    axes[0][0].set_ylabel(ylabel)
    legend_below(fig, axes)
    save(fig, out, name, made)


# --------------------------------------------------------------------------
# Figures

def fig_time_speedup(df, out, made):
    d = single_config(df)
    d = d[d.experiment == "baseline"]
    if d.empty:
        return
    per_k_lines(d, "median_ms", "time per call (ms, log scale)", out, "fig1_time.png", made)
    per_k_lines(d[d.impl != "seq"], "speedup", "speedup over seq (log scale)",
                out, "fig2_speedup.png", made, hline=1)


def fig_cuda_kernel_vs_e2e(df, out, made):
    d = single_config(df)
    d = d[d.is_cuda]
    if d.empty:
        return
    ks, sizes = sorted(d.K.unique()), sorted(d.H.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3.4 * len(ks), 3.6), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        for impl, g in d[d.K == k].groupby("impl"):
            g = g.sort_values("H")
            st = STYLE[impl]
            ax.plot(g.H, g.speedup_kernel, color=st["color"], marker=st["marker"],
                    label=f"{impl}: kernel only")
            ax.plot(g.H, g.speedup, color=st["color"], marker=st["marker"], ls=":",
                    markerfacecolor=SURFACE, label=f"{impl}: end-to-end")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        size_axis(ax, sizes)
        ax.set_yscale("log")
        ax.set_title(f"{k}×{k} filter")
    axes[0][0].set_ylabel("speedup over seq (log scale)")
    legend_below(fig, axes)
    save(fig, out, "fig3_cuda_kernel_vs_e2e.png", made)


def fig_cuda_breakdown(df, out, made):
    d = single_config(df)
    d = d[d.is_cuda]
    if d.empty:
        return
    impls, ks = sorted(d.impl.unique()), sorted(d.K.unique())
    stages = [("h2d_ms", "host→device copy", SLOTS[0]), ("kernel_ms", "kernel", SLOTS[1]),
              ("d2h_ms", "device→host copy", SLOTS[2])]
    fig, axes = plt.subplots(len(impls), len(ks), figsize=(3.4 * len(ks), 2.8 * len(impls)),
                             sharey=True, squeeze=False)
    for r, impl in enumerate(impls):
        for c, k in enumerate(ks):
            ax = axes[r][c]
            g = d[(d.impl == impl) & (d.K == k)].sort_values("H")
            x = np.arange(len(g))
            total = sum(g[s].values for s, _, _ in stages)
            bottom = np.zeros(len(g))
            for s, label, color in stages:
                frac = g[s].values / total * 100
                ax.bar(x, frac, bottom=bottom, width=0.75, color=color, edgecolor=SURFACE,
                       linewidth=1.5, label=label)
                bottom += frac
            ax.set_xticks(x)
            ax.set_xticklabels([f"{h}²" for h in g.H])
            ax.set_ylim(0, 100)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{impl}, {k}×{k} filter", fontsize=10)
        axes[r][0].set_ylabel("% of GPU time")
    legend_below(fig, axes)
    save(fig, out, "fig4_cuda_breakdown.png", made)


def fig_omp(df, out, made):
    d = df[(df.experiment == "omp") & df.schedule.notna()]
    if d.empty:
        return None
    impls = sorted(d.impl.unique())
    probs = sorted(d[["H", "K"]].drop_duplicates().itertuples(index=False))
    scheds = sorted(d.schedule.unique())
    threads = sorted(d.threads.unique())
    fig, axes = plt.subplots(len(impls), len(probs), figsize=(3.3 * len(probs), 3.2 * len(impls)),
                             sharey=True, squeeze=False)
    for r, impl in enumerate(impls):
        for c, (h, k) in enumerate(probs):
            ax = axes[r][c]
            for i, s in enumerate(scheds):
                g = d[(d.impl == impl) & (d.H == h) & (d.K == k) & (d.schedule == s)].sort_values("threads")
                ax.plot(g.threads, g.speedup, color=SLOTS[i], marker=MARKERS[i], label=s)
            ax.plot([1, max(threads)], [1, max(threads)], color=INK2, lw=1, ls="--", label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks(threads)
            ax.set_xticklabels([str(int(t)) for t in threads])
            ax.minorticks_off()
            ax.set_xlabel("threads")
            ax.set_title(f"{impl}: {h}², {k}×{k}", fontsize=10)
        axes[r][0].set_ylabel("speedup over seq")
    legend_below(fig, axes)
    save(fig, out, "fig5_omp_scaling.png", made)
    return d


def fig_blocks(df, out, made):
    d = df[df.experiment == "cuda_blocks"]
    if d.empty:
        return None
    d = d.assign(block=d.bx.astype(int).astype(str) + "×" + d.by.astype(int).astype(str),
                 config=d.impl + "\n" + d.H.astype(str) + "², " + d.K.astype(str) + "×" + d.K.astype(str))
    piv = d.pivot_table(index="block", columns="config", values="kernel_ms", aggfunc="median")
    rel = piv / piv.min()
    order = rel.mean(axis=1).sort_values().index
    piv, rel = piv.loc[order], rel.loc[order]
    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * rel.shape[1], 1.0 + 0.38 * rel.shape[0]))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    im = ax.imshow(np.log2(rel.values.clip(1, 4)), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for (i, j), v in np.ndenumerate(rel.values):
        if not np.isnan(v):
            ax.text(j, i, f"{piv.values[i, j]:.3g} ms\n{v:.2f}×", ha="center", va="center",
                    fontsize=7, color="white" if v > 2.1 else INK)
    ax.set_xticks(range(rel.shape[1]))
    ax.set_xticklabels(rel.columns, fontsize=8)
    ax.set_yticks(range(rel.shape[0]))
    ax.set_yticklabels(rel.index)
    ax.set_ylabel("thread block (x × y)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
    cb.ax.set_yticklabels(["1× (best)", "2×", "≥4×"])
    cb.set_label("kernel time relative to the best block")
    ax.set_title("CUDA kernel time by thread-block shape (rows sorted best first)")
    fig.tight_layout()
    save(fig, out, "fig6_cuda_blocks.png", made)
    return piv


def fig_throughput(df, out, made):
    d = single_config(df)
    if d.empty:
        return
    per_k_lines(d, "gflops", "GFLOP/s (kernel only for GPU; log scale)",
                out, "fig7_throughput.png", made)


def fastest_table(df):
    d = single_config(df)
    d = d[~d.impl.str.startswith("torch")]
    if d.empty:
        return None
    best = d.loc[d.groupby(["H", "K"]).median_ms.idxmin()]
    return best


def fig_fastest(best, out, made):
    if best is None or best.empty:
        return
    ks, sizes = sorted(best.K.unique()), sorted(best.H.unique())
    fig, ax = plt.subplots(figsize=(1.4 + 1.5 * len(sizes), 0.9 + 0.7 * len(ks)))
    for _, r in best.iterrows():
        x, y = sizes.index(r.H), ks.index(r.K)
        ax.add_patch(plt.Rectangle((x - 0.47, y - 0.45), 0.94, 0.9, color=STYLE[r.impl]["color"],
                                   alpha=0.22, lw=0))
        ax.text(x, y, f"{r.impl}\n{r.speedup:.1f}× vs seq", ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-0.5, len(ks) - 0.5)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s}²" for s in sizes])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"{k}×{k}" for k in ks])
    ax.set_xlabel("image size")
    ax.set_ylabel("filter")
    ax.grid(False)
    ax.set_title("Fastest implementation, end-to-end (our implementations only)")
    fig.tight_layout()
    save(fig, out, "fig8_fastest.png", made)


# --------------------------------------------------------------------------
# Tables (summary.md)

def md(t, fmt="{:.3g}"):
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join([str(t.index.name or "")] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in t.iterrows():
        cells = ["" if pd.isna(v) else fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in row.values]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def by_config(d, value):
    t = d.assign(config=d.H.astype(str) + "², K=" + d.K.astype(str), v=value) \
        .pivot_table(index=["H", "K", "config"], columns="impl", values="v", aggfunc="median")
    t = t.reset_index(level=["H", "K"], drop=True)
    t.index.name = "workload"
    return t.dropna(axis=1, how="all")


def summary(df, out, omp, blocks, best):
    p = ["# Benchmark results\n"]
    checked = df["pass"].notna() & (df["pass"].astype(str) != "")
    failed = df[checked & (df["pass"].astype(float) == 0)]
    p.append(f"{len(df)} measurements; {int(checked.sum())} checked against seq, "
             f"{len(failed)} failed the check.\n")

    d = single_config(df)
    if not d.empty:
        p += ["\n## Execution time, end-to-end (ms, median)\n", md(by_config(d, d.median_ms))]
        dd = d[d.impl != "seq"]
        p += ["\n## Speedup over seq, end-to-end (H1, H2)\n", md(by_config(dd, dd.speedup), "{:.2f}")]
        dc = d[d.is_cuda]
        if not dc.empty:
            p += ["\n## CUDA: kernel-only speedup over seq, and share of time spent copying\n",
                  md(by_config(dc, dc.speedup_kernel), "{:.2f}"),
                  "\nTransfer share (h2d + d2h) / end-to-end GPU time:\n",
                  md(by_config(dc, (dc.h2d_ms + dc.d2h_ms) / (dc.h2d_ms + dc.kernel_ms + dc.d2h_ms) * 100),
                     "{:.0f}%")]
        # H2: smallest image where each implementation beats seq, per filter size
        rows = {}
        for impl, g in dd.groupby("impl"):
            rows[impl] = {f"{k}×{k}": (f"{int(w.H.min())}²" if len(w) else "never")
                          for k, gk in g.groupby("K") for w in [gk[gk.speedup > 1]]}
        t = pd.DataFrame(rows).T
        t.index.name = "implementation"
        p += ["\n## Smallest image size where the implementation beats seq (H2)\n", md(t)]
    if best is not None:
        t = best.assign(cell=best.impl + " (" + best.speedup.map("{:.1f}×".format) + ")") \
            .pivot_table(index="K", columns="H", values="cell", aggfunc="first")
        t.index = [f"{k}×{k}" for k in t.index]
        t.columns = [f"{h}²" for h in t.columns]
        t.index.name = "filter"
        p += ["\n## Fastest implementation per workload (dispatcher thresholds)\n", md(t)]
    if omp is not None:
        o = omp.assign(eff=omp.speedup / omp.threads)
        t = o.assign(config=o.impl + " " + o.H.astype(str) + "² K=" + o.K.astype(str)) \
            .pivot_table(index="threads", columns=["config", "schedule"], values="speedup")
        t.columns = [f"{a} {b}" for a, b in t.columns]
        p += ["\n## OpenMP speedup over seq by thread count\n", md(t, "{:.2f}")]
        tmax = o.threads.max()
        e = o[o.threads == tmax]
        t = e.assign(config=e.impl + " " + e.H.astype(str) + "² K=" + e.K.astype(str)) \
            .pivot_table(index="config", columns="schedule", values="eff") * 100
        p += [f"\n## OpenMP parallel efficiency at {int(tmax)} threads (%)\n", md(t, "{:.0f}")]
    if blocks is not None:
        t = blocks.rename(columns=lambda c: c.replace("\n", " "))
        p += ["\n## CUDA kernel time by thread block (ms) (H3)\n", md(t)]
    if "torch_cudnn" in set(df.impl):
        dd = d[d.is_cuda]
        ref = dd[dd.impl == "torch_cudnn"].set_index(["H", "K"]).compute_ms
        ours = dd[~dd.impl.str.startswith("torch")]
        if not ours.empty:
            ours = ours.assign(rel=ours.set_index(["H", "K"]).index.map(ref) / ours.compute_ms)
            p += ["\n## Our CUDA kernels vs cuDNN: kernel-time ratio (cuDNN time / ours; >1 = ours faster)\n",
                  md(by_config(ours, ours.rel), "{:.2f}")]
    if len(failed):
        p += ["\n## Failed correctness checks\n",
              md(failed.set_index("impl")[["experiment", "H", "K", "bx", "by", "threads", "err_ratio", "nonfinite"]])]
    (out / "summary.md").write_text("\n".join(p) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "plots" / res.name
    out.mkdir(parents=True, exist_ok=True)

    df = load(res)
    set_styles(df.impl.unique())
    made = []
    fig_time_speedup(df, out, made)
    fig_cuda_kernel_vs_e2e(df, out, made)
    fig_cuda_breakdown(df, out, made)
    omp = fig_omp(df, out, made)
    blocks = fig_blocks(df, out, made)
    fig_throughput(df, out, made)
    best = fastest_table(df)
    fig_fastest(best, out, made)
    summary(df, out, omp, blocks, best)
    print(f"{out}: " + ", ".join(made + ["summary.md"]))


if __name__ == "__main__":
    main()
