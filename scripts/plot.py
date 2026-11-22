#!/usr/bin/env python3
"""Figures and tables for the final report from one benchmark run.

Usage: plot.py RESULTS_DIR [-o PLOTS_DIR]

Reads the CSVs written by scripts/bench.sh (baseline, omp, cuda_blocks,
torch, channels, torch_channels, channel_blocks) and writes PNG figures plus
summary.md (Markdown tables) into PLOTS_DIR (default: plots/<run name>/).
Experiments that weren't run are skipped. Needs pandas and matplotlib.

  Figure                         Proposal item
  fig1_time.png                  execution time for each implementation, across image and filter sizes
  fig2_speedup.png               OpenMP and CUDA speedup relative to sequential (H1, H2)
  fig3_cuda_kernel_vs_e2e.png    CUDA kernel-only vs end-to-end speedup
  fig4_cuda_breakdown.png        where CUDA end-to-end time goes: h2d / kernel / d2h (H3)
  fig5_omp_scaling.png           OpenMP thread scaling, static vs dynamic (vs guided)
  fig6_cuda_blocks.png           CUDA thread-block configuration (H3)
  fig7_throughput.png            GFLOP/s vs the cuDNN / PyTorch reference
  fig8_fastest.png               fastest implementation per workload (stretch goal: dispatcher)
  fig9_cuda_kernel_time.png      CUDA kernel times at the baseline block
  fig10_cuda_kernel_speedup.png  CUDA kernel speedup vs cuda at the same configuration
  fig11_cuda_call_speedup.png    Full-call speedup vs cuda at the same configuration
  fig12_channels_time.png        multi-channel input: time per call vs channel count
  fig13_channels_speedup.png     multi-channel input: speedup over seq vs channel count
  fig14_channels_transfer.png    multi-channel input: share of CUDA time spent copying
  (fig6_cuda_blocks.png also gets panels for the channel_blocks sweep)
"""
import argparse
import math
import re
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
SLOTS = ["#222222", "#2674bc", "#c85a17", "#16815d", "#aa3377", "#7950a5",
         "#9a6500", "#007b80", "#b33d3d", "#658125", "#5871b7", "#c05781",
         "#725348", "#6996a8", "#a46a2b", "#548c59"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "d", "p", "8", "H", "x"]
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

EXPERIMENTS = {"baseline", "omp", "cuda_blocks", "torch",
               "channels", "torch_channels", "channel_blocks"}
# Sweeps over the input-channel count. Everything else is single-channel.
CHANNEL_EXPERIMENTS = {"channels", "torch_channels", "channel_blocks"}


def set_styles(names):
    # seq first, then CPU parallel, then CUDA, then the external reference.
    def rank(n):
        return (0 if n.startswith("seq") else 1 if n.startswith("omp") else
                3 if n.startswith("torch") else 2, n)
    STYLE.clear()
    for i, n in enumerate(sorted(set(names), key=rank)):
        STYLE[n] = dict(color=SLOTS[i % len(SLOTS)], marker=MARKERS[i % len(MARKERS)],
                        linestyle="-" if i < len(SLOTS) else "--")


def load(results):
    frames = []
    seen_experiments = {}
    for f in sorted(Path(results).glob("*.csv")):
        # Attachment downloads may rename cuda_blocks.csv to cuda_blocks(1).csv.
        experiment = re.sub(r"\s*\(\d+\)$", "", f.stem)
        if experiment not in EXPERIMENTS:
            continue
        if experiment in seen_experiments:
            sys.exit(f"multiple files map to '{experiment}': {seen_experiments[experiment]} and {f.name}; "
                     "use one results file per sweep")
        seen_experiments[experiment] = f.name
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if "impl" in d.columns and not d.empty:
            frames.append(d.assign(experiment=experiment))
    if not frames:
        sys.exit(f"no benchmark CSVs in {results}")
    df = pd.concat(frames, ignore_index=True)
    required = {"H", "W", "K", "median_ms"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"benchmark CSVs are missing columns: {', '.join(sorted(missing))}")
    for col in ["H", "W", "K", "bx", "by", "threads", "median_ms", "kernel_ms",
                "h2d_ms", "d2h_ms", "err_ratio", "nonfinite", "pass"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "schedule" not in df:
        df["schedule"] = np.nan
    # Input channels (bin/conv --C). Older CSVs have no C column: one channel.
    df["C"] = pd.to_numeric(df["C"], errors="coerce").fillna(1) if "C" in df else 1.0
    # The single-channel figures group by H, W and K only. Multi-channel rows
    # belong to the channel sweeps; any elsewhere would be mixed in, so drop them.
    stray = (df["C"] != 1) & ~df.experiment.isin(CHANNEL_EXPERIMENTS)
    if stray.any():
        print(f"Skipping {int(stray.sum())} multi-channel rows outside the channel sweeps",
              file=sys.stderr)
        df = df[~stray].copy()
    invalid = (df[["H", "W", "K", "median_ms"]].isna().any(axis=1) |
               ~np.isfinite(df[["H", "W", "K"]]).all(axis=1) |
               ~np.isfinite(df["median_ms"]) | (df["median_ms"] <= 0) |
               (df[["H", "W", "K"]] <= 0).any(axis=1))
    if invalid.any():
        print(f"Skipping {int(invalid.sum())} rows with invalid workload or total time", file=sys.stderr)
        df = df[~invalid].copy()
    if df.empty:
        sys.exit("no valid benchmark timing rows")
    df["is_cuda"] = df.impl.str.startswith("cuda") | df.impl.str.startswith("torch_cudnn")
    df.loc[~np.isfinite(df.kernel_ms) | (df.kernel_ms <= 0), "kernel_ms"] = np.nan
    df["flops"] = 2.0 * df.C * df.K ** 2 * df.H * df.W
    # Compute-only time: the kernel for GPU rows, the whole call for CPU rows.
    df["compute_ms"] = df.kernel_ms.where(df.is_cuda, df.median_ms)
    df.loc[~np.isfinite(df.compute_ms) | (df.compute_ms <= 0), "compute_ms"] = np.nan
    df["gflops"] = df.flops / (df.compute_ms * 1e6)
    # Prefer the seq timing measured in the same sweep; for sweeps such as
    # cuda_blocks or torch that have no seq row of their own, use seq from the
    # baseline (C = 1) or channels sweep with the same H, W, K and C.
    seq_rows = df[(df.impl == "seq") & (df["pass"] != 0)]
    seq = seq_rows.groupby(["experiment", "H", "W", "K", "C"]).median_ms.median().rename("seq_ms")
    baseline = (seq_rows[seq_rows.experiment.isin(["baseline", "channels"])]
                .groupby(["H", "W", "K", "C"]).median_ms.median().rename("baseline_seq_ms"))
    df = df.join(seq, on=["experiment", "H", "W", "K", "C"])
    df = df.join(baseline, on=["H", "W", "K", "C"])
    df["seq_ms"] = df.seq_ms.fillna(df.baseline_seq_ms)
    df["speedup"] = df.seq_ms / df.median_ms
    df["speedup_kernel"] = df.seq_ms / df.compute_ms
    return df


def single_config(df):
    """One row per (impl, H, W, K): the default configuration of each
    implementation (OpenMP: most threads, static; CUDA: the default block)."""
    d = df[df.experiment.isin(["baseline", "torch"])]
    return d.sort_values("threads", na_position="first").groupby(["experiment", "impl", "H", "W", "K"]).tail(1)


def size_axis(ax, sizes):
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{int(h)}×{int(w)}" for h, w in sizes], rotation=20, ha="right")
    ax.set_xlabel("image size (height × width)")


def image_sizes(d):
    return sorted({(int(h), int(w)) for h, w in zip(d.H, d.W)})


def legend_below(fig, axes, extra_rows=0.0):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if labels:
        ncol = min(len(labels), 3)
        fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        margin = min(0.36, 0.06 + 0.035 * math.ceil(len(labels) / ncol) + extra_rows)
    else:
        margin = 0.02 + extra_rows
    fig.tight_layout(rect=(0, margin, 1, 1))


def save(fig, out, name, made):
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    made.append(name)


def per_k_lines(d, y, ylabel, out, name, made, hline=None, impls=None, logy=True):
    d = d[d[y].notna()].copy()
    if logy:
        d = d[d[y] > 0]
    if impls is not None:
        d = d[d.impl.isin(impls)]
    if d.empty:
        return
    ks, sizes = sorted(d.K.unique()), image_sizes(d)
    position = {size: i for i, size in enumerate(sizes)}
    fig, axes = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 5.5), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        for impl, g in d[d.K == k].groupby("impl"):
            g = g.sort_values(["H", "W"])
            if g[y].notna().any():
                x = [position[(int(h), int(w))] for h, w in zip(g.H, g.W)]
                ax.plot(x, g[y], label=impl, **STYLE[impl])
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
    d = d[d.is_cuda & d.speedup.notna() & d.speedup_kernel.notna()]
    if d.empty:
        return
    ks, sizes = sorted(d.K.unique()), image_sizes(d)
    position = {size: i for i, size in enumerate(sizes)}
    fig, axes = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 6), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        for impl, g in d[d.K == k].groupby("impl"):
            g = g.sort_values(["H", "W"])
            st = STYLE[impl]
            x = [position[(int(h), int(w))] for h, w in zip(g.H, g.W)]
            ax.plot(x, g.speedup_kernel, color=st["color"], marker=st["marker"],
                    label=f"{impl}: kernel only")
            ax.plot(x, g.speedup, color=st["color"], marker=st["marker"], ls=":",
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
            g = d[(d.impl == impl) & (d.K == k)].sort_values(["H", "W"])
            x = np.arange(len(g))
            total = sum(g[s].values for s, _, _ in stages)
            total = np.where(np.isfinite(total) & (total > 0), total, np.nan)
            bottom = np.zeros(len(g))
            for s, label, color in stages:
                frac = g[s].values / total * 100
                ax.bar(x, frac, bottom=bottom, width=0.75, color=color, edgecolor=SURFACE,
                       linewidth=1.5, label=label)
                bottom += frac
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(h)}×{int(w)}" for h, w in zip(g.H, g.W)], rotation=20, ha="right")
            ax.set_ylim(0, 100)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{impl}, {k}×{k} filter", fontsize=10)
        axes[r][0].set_ylabel("% of sum of stage medians")
    legend_below(fig, axes)
    save(fig, out, "fig4_cuda_breakdown.png", made)


def fig_omp(df, out, made):
    d = df[(df.experiment == "omp") & df.schedule.notna() & df.threads.notna()].copy()
    if d.empty:
        return None
    # Parallel scaling compares an implementation against its OWN 1-thread
    # run, with the same schedule and workload. seq may use a different
    # algorithm/vectorization, so seq/omp_time is not parallel efficiency.
    keys = ["impl", "H", "W", "K", "schedule"]
    one = d[d.threads == 1].groupby(keys).median_ms.median().rename("one_thread_ms")
    d = d.join(one, on=keys)
    d["parallel_speedup"] = d.one_thread_ms / d.median_ms
    if d.parallel_speedup.notna().sum() == 0:
        print("Skipping OpenMP scaling: no matching one-thread runs", file=sys.stderr)
        return None
    impls = sorted(d.impl.unique())
    probs = sorted(d[["H", "W", "K"]].drop_duplicates().itertuples(index=False, name=None))
    scheds = sorted(d.schedule.unique())
    threads = sorted(d.threads.unique())
    fig, axes = plt.subplots(len(impls), len(probs), figsize=(3.3 * len(probs), 3.2 * len(impls)),
                             sharey=True, squeeze=False)
    for r, impl in enumerate(impls):
        for c, (h, w, k) in enumerate(probs):
            ax = axes[r][c]
            for i, s in enumerate(scheds):
                g = d[(d.impl == impl) & (d.H == h) & (d.W == w) & (d.K == k) & (d.schedule == s)].sort_values("threads")
                ax.plot(g.threads, g.parallel_speedup, color=SLOTS[i], marker=MARKERS[i], label=s)
            ax.plot([1, max(threads)], [1, max(threads)], color=INK2, lw=1, ls="--", label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_xticks(threads)
            ax.set_xticklabels([str(int(t)) for t in threads])
            ax.minorticks_off()
            ax.set_xlabel("threads")
            ax.set_title(f"{impl}: {int(h)}×{int(w)}, {int(k)}×{int(k)}", fontsize=10)
        axes[r][0].set_ylabel("speedup vs same variant, 1 thread")
    legend_below(fig, axes)
    save(fig, out, "fig5_omp_scaling.png", made)
    return d


def fig_blocks(df, out, made):
    """Block-shape sweeps: cuda_blocks (C = 1) and channel_blocks (C > 1)."""
    d = df[df.experiment.isin(["cuda_blocks", "channel_blocks"]) & df.is_cuda &
           df.bx.notna() & df.by.notna() &
           np.isfinite(df.kernel_ms) & (df.kernel_ms > 0)].copy()
    if d.empty:
        return None
    d["block"] = d.bx.astype(int).astype(str) + "×" + d.by.astype(int).astype(str)
    workloads = sorted(d[["C", "H", "W", "K"]].drop_duplicates().itertuples(index=False, name=None))
    cols = min(2, len(workloads))
    rows = math.ceil(len(workloads) / cols)
    n_blocks = d.block.nunique()
    row_h = max(4.3, 0.22 * n_blocks + 1.2)
    fig, axes = plt.subplots(rows, cols, figsize=(8.0 * cols, row_h * rows), squeeze=False)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    im = None
    for ax, (c, h, w, k) in zip(axes.flat, workloads):
        g = d[(d.C == c) & (d.H == h) & (d.W == w) & (d.K == k)]
        piv = g.pivot_table(index="block", columns="impl", values="kernel_ms", aggfunc="median")
        block_order = sorted(piv.index, key=lambda s: tuple(map(int, s.split("×"))))
        impl_order = sorted(piv.columns)
        piv = piv.loc[block_order, impl_order]
        rel = piv / piv.min()
        im = ax.imshow(np.log2(rel.values.clip(1, 4)), cmap=cmap, vmin=0, vmax=2, aspect="auto")
        for (i, j), value in np.ndenumerate(rel.values):
            if not np.isnan(value):
                ax.text(j, i, f"{piv.values[i, j]:.3g}\n{value:.2f}×", ha="center", va="center",
                        fontsize=6.5, color="white" if value > 2.1 else INK)
        ax.set_xticks(range(len(impl_order)))
        ax.set_xticklabels([name.replace("cuda_", "") for name in impl_order], rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(block_order)))
        ax.set_yticklabels(block_order, fontsize=8)
        ax.set_ylabel("block (x × y)")
        c_label = "" if c == 1 else f", {int(c)} input channels"
        ax.set_title(f"{int(h)}×{int(w)} image, {int(k)}×{int(k)} filter{c_label}")
        ax.grid(False)
    for ax in list(axes.flat)[len(workloads):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.12, top=0.95, wspace=0.25, hspace=0.35)
    cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.25, 0.015, 0.55]), ticks=[0, 1, 2])
    cb.ax.set_yticklabels(["1× (best)", "2×", "≥4×"])
    cb.set_label("time / fastest block for the same variant and workload")
    fig.suptitle("CUDA kernel time by block shape: cells show ms and ratio", fontsize=12)
    save(fig, out, "fig6_cuda_blocks.png", made)
    return d


def fig_throughput(df, out, made):
    d = single_config(df)
    if d.empty:
        return
    per_k_lines(d, "gflops", "nominal GFLOP/s (GPU kernel; CPU call)",
                out, "fig7_throughput.png", made)


def cuda_vs_direct(df):
    """Compare only rows with the same sweep, image, filter, and block."""
    d = single_config(df)
    d = d[(d.experiment == "baseline") & d.impl.str.startswith("cuda") &
          d.bx.notna() & d.by.notna()].copy()
    keys = ["experiment", "H", "W", "K", "bx", "by"]
    refs = (d[d.impl == "cuda"].groupby(keys)[["median_ms", "kernel_ms"]].median()
            .rename(columns={"median_ms": "cuda_call_ms", "kernel_ms": "cuda_kernel_ms"}))
    d = d.join(refs, on=keys)
    d["kernel_speedup_vs_cuda"] = d.cuda_kernel_ms / d.kernel_ms
    d["call_speedup_vs_cuda"] = d.cuda_call_ms / d.median_ms
    return d


def fig_cuda_variants(df, out, made):
    d = cuda_vs_direct(df)
    if d.empty:
        return d
    per_k_lines(d, "kernel_ms", "kernel time (ms; CUDA-event median)",
                out, "fig9_cuda_kernel_time.png", made)
    per_k_lines(d, "kernel_speedup_vs_cuda", "kernel speedup vs cuda (>1 is faster)",
                out, "fig10_cuda_kernel_speedup.png", made, hline=1, logy=False)
    per_k_lines(d, "call_speedup_vs_cuda", "full-call speedup vs cuda (>1 is faster)",
                out, "fig11_cuda_call_speedup.png", made, hline=1, logy=False)
    return d


def fastest_table(df):
    d = single_config(df)
    d = d[~d.impl.str.startswith("torch")]
    if d.empty:
        return None
    best = d.loc[d.groupby(["H", "W", "K"]).median_ms.idxmin()]
    return best


def fig_fastest(best, out, made):
    if best is None or best.empty:
        return
    ks, sizes = sorted(best.K.unique()), image_sizes(best)
    fig, ax = plt.subplots(figsize=(1.4 + 1.5 * len(sizes), 0.9 + 0.7 * len(ks)))
    for _, r in best.iterrows():
        x, y = sizes.index((int(r.H), int(r.W))), ks.index(r.K)
        ax.add_patch(plt.Rectangle((x - 0.47, y - 0.45), 0.94, 0.9, color=STYLE[r.impl]["color"],
                                   alpha=0.22, lw=0))
        label = f"{r.speedup:.1f}× vs seq" if pd.notna(r.speedup) else "no seq timing"
        ax.text(x, y, f"{r.impl}\n{label}", ha="center", va="center", fontsize=8)
    ax.set_xlim(-0.5, len(sizes) - 0.5)
    ax.set_ylim(-0.5, len(ks) - 0.5)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{h}×{w}" for h, w in sizes])
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

def md(t, fmt="{:.4f}"):
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join([str(t.index.name or "")] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in t.iterrows():
        cells = ["" if pd.isna(v) else fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in row.values]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def by_config(d, value):
    config = d.H.astype(int).astype(str) + "×" + d.W.astype(int).astype(str) + ", K=" + d.K.astype(int).astype(str)
    t = d.assign(config=config, v=value).pivot_table(index=["H", "W", "K", "config"], columns="impl", values="v", aggfunc="median")
    t = t.reset_index(level=["H", "W", "K"], drop=True)
    t.index.name = "workload"
    return t.dropna(axis=1, how="all")


def summary(df, out, omp, blocks, best, failed, results_dir, cuda_comparison, channels=None):
    p = ["# Benchmark results\n"]
    p.append(f"{len(df)} usable timing rows; {len(failed)} explicitly failed rows excluded from the plots.\n")
    p.append("A `pass=1` field alone does not prove `--check` was enabled; consult the benchmark command or log.\n")
    p.append("Where available, speedup vs seq uses the sequential timing from the same sweep and workload; "
             "CUDA-only sweeps fall back to the matching baseline sequential timing.\n")
    sysinfo = Path(results_dir) / "sysinfo.txt"
    if sysinfo.is_file():
        text = sysinfo.read_text(encoding="utf-8", errors="replace")
        if "permission denied" in text.lower():
            p.append("Hardware metadata is incomplete: sysinfo.txt reports a permission error.\n")
    if "torch_cudnn" not in set(df.impl):
        p.append("No cuDNN benchmark rows were found; no cuDNN comparison is reported.\n")
    p.append("Nominal GFLOP/s uses 2 × H × W × K² operations; border handling can reduce actual work.\n")

    d = single_config(df)
    if not d.empty:
        p += ["\n## Execution time, end-to-end (ms, median)\n", md(by_config(d, d.median_ms))]
        dd = d[d.impl != "seq"]
        p += ["\n## Speedup over seq, end-to-end (H1, H2)\n", md(by_config(dd, dd.speedup), "{:.2f}")]
        dc = d[d.is_cuda]
        if not dc.empty:
            p += ["\n## CUDA: kernel-only speedup over seq, and share of time spent copying\n",
                  md(by_config(dc, dc.speedup_kernel), "{:.2f}"),
                  "\nTransfer share (h2d + d2h) / sum of the three stage medians; "
                  "these medians are not necessarily the median full-call time:\n",
                  md(by_config(dc, (dc.h2d_ms + dc.d2h_ms) / (dc.h2d_ms + dc.kernel_ms + dc.d2h_ms) * 100),
                     "{:.0f}%")]
        # H2: smallest image where each implementation beats seq, per filter size
        if not cuda_comparison.empty:
            p += ["\n## CUDA kernel time at the baseline block (ms)\n",
                  md(by_config(cuda_comparison, cuda_comparison.kernel_ms)),
                  "\n## CUDA kernel speedup vs cuda at the same image, filter, and block (>1 is faster)\n",
                  md(by_config(cuda_comparison, cuda_comparison.kernel_speedup_vs_cuda), "{:.2f}")]
        def first_tested_win(group):
            with_ref = group[group.seq_ms.notna()]
            if with_ref.empty:
                return "no seq reference"
            wins = with_ref[with_ref.speedup > 1].sort_values(["H", "W"])
            if wins.empty:
                return "none observed"
            return f"{int(wins.iloc[0].H)}×{int(wins.iloc[0].W)}"
        rows = {}
        for impl, g in dd.groupby("impl"):
            rows[impl] = {f"{int(k)}×{int(k)}": first_tested_win(gk) for k, gk in g.groupby("K")}
        t = pd.DataFrame(rows).T
        t.index.name = "implementation"
        p += ["\n## Smallest tested image size where the implementation beats seq\n", md(t)]
    if best is not None:
        label = best.speedup.map(lambda x: f"{x:.1f}×" if pd.notna(x) else "no seq timing")
        t = best.assign(cell=best.impl + " (" + label + ")", size=best.H.astype(int).astype(str) + "×" + best.W.astype(int).astype(str)) \
            .pivot_table(index="K", columns="size", values="cell", aggfunc="first")
        t.index = [f"{k}×{k}" for k in t.index]
        t.index.name = "filter"
        p += ["\n## Fastest tested implementation per baseline workload\n", md(t)]
    if omp is not None:
        o = omp.assign(eff=omp.parallel_speedup / omp.threads)
        p += ["\n## OpenMP parallel speedup vs the same variant at one thread\n"]
        for (impl, h, w, k), g in o.groupby(["impl", "H", "W", "K"]):
            t = g.pivot_table(index="threads", columns="schedule", values="parallel_speedup", aggfunc="median")
            p += [f"\n### {impl}: {int(h)}×{int(w)}, K={int(k)}\n", md(t, "{:.2f}")]
        tmax = o.threads.max()
        e = o[o.threads == tmax]
        t = e.assign(config=e.impl + " " + e.H.astype(int).astype(str) + "×" + e.W.astype(int).astype(str) + " K=" + e.K.astype(int).astype(str)) \
            .pivot_table(index="config", columns="schedule", values="eff") * 100
        p += [f"\n## OpenMP parallel efficiency at {int(tmax)} threads (%)\n", md(t, "{:.0f}")]
    if blocks is not None:
        p += ["\n## CUDA kernel time by thread block (ms)\n"]
        for (c, h, w, k), g in blocks.groupby(["C", "H", "W", "K"]):
            t = g.pivot_table(index="block", columns="impl", values="kernel_ms", aggfunc="median")
            t = t.loc[sorted(t.index, key=lambda s: tuple(map(int, s.split("×"))))]
            c_label = "" if c == 1 else f", C={int(c)}"
            p += [f"\n### {int(h)}×{int(w)}, K={int(k)}{c_label}\n", md(t)]
            p += ["\nRelative to the fastest block for each kernel (1.00 = fastest):\n",
                  md(t / t.min(), "{:.2f}")]
    if "torch_cudnn" in set(df.impl):
        dd = d[d.is_cuda]
        ref = dd[dd.impl == "torch_cudnn"].groupby(["H", "W", "K"]).compute_ms.median()
        ours = dd[~dd.impl.str.startswith("torch")]
        if not ours.empty:
            ours = ours.assign(rel=ours.set_index(["H", "W", "K"]).index.map(ref) / ours.compute_ms)
            p += ["\n## Our CUDA kernels vs cuDNN: kernel-time ratio (cuDNN time / ours; >1 = ours faster)\n",
                  md(by_config(ours, ours.rel), "{:.2f}")]
    p += channel_summary(channels)
    if len(failed):
        p += ["\n## Failed correctness checks\n",
              md(failed.set_index("impl")[["experiment", "H", "K", "C", "bx", "by", "threads", "err_ratio", "nonfinite"]])]
    (out / "summary.md").write_text("\n".join(p) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Multi-channel input (channels and torch_channels sweeps)

def channel_rows(df):
    """One row per (impl, H, W, K, C) from the channel sweeps (not the block sweep)."""
    d = df[df.experiment.isin(["channels", "torch_channels"])]
    return d.sort_values("threads", na_position="first").groupby(
        ["experiment", "impl", "H", "W", "K", "C"]).tail(1)


def channel_grid(d, y, ylabel, out, name, made, hline=None, logy=True, title=None):
    """Rows: image sizes. Columns: filter sizes. x: channel count (log 2)."""
    d = d[d[y].notna() & np.isfinite(d[y])]
    if logy:
        d = d[d[y] > 0]
    if d.empty:
        return
    sizes, ks = image_sizes(d), sorted(d.K.unique())
    cs = sorted(int(c) for c in d.C.unique())
    fig, axes = plt.subplots(len(sizes), len(ks), figsize=(4.3 * len(ks), 3.3 * len(sizes)),
                             sharex=True, squeeze=False)
    for r, (h, w) in enumerate(sizes):
        for c, k in enumerate(ks):
            ax = axes[r][c]
            for impl, g in d[(d.H == h) & (d.W == w) & (d.K == k)].groupby("impl"):
                g = g.sort_values("C")
                ax.plot(g.C, g[y], label=impl, **STYLE[impl])
            if hline is not None:
                ax.axhline(hline, color=INK2, lw=1, ls="--")
            ax.set_xscale("log", base=2)
            ax.set_xticks(cs)
            ax.set_xticklabels([str(v) for v in cs])
            ax.minorticks_off()
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{h}×{w} image, {int(k)}×{int(k)} filter", fontsize=10)
            if r == len(sizes) - 1:
                ax.set_xlabel("input channels (C)")
        axes[r][0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title, fontsize=12)
    legend_below(fig, axes)
    save(fig, out, name, made)


def fig_channels(df, out, made):
    d = channel_rows(df)
    if d.empty:
        return d
    channel_grid(d, "median_ms", "time per call (ms, log)", out, "fig12_channels_time.png", made,
                 title="Multi-channel input: end-to-end time per call")
    channel_grid(d[d.impl != "seq"], "speedup", "speedup over seq (log)", out,
                 "fig13_channels_speedup.png", made, hline=1,
                 title="Multi-channel input: end-to-end speedup over seq at the same C")
    dc = d[d.is_cuda].copy()
    dc["copy_share"] = (dc.h2d_ms + dc.d2h_ms) / (dc.h2d_ms + dc.kernel_ms + dc.d2h_ms) * 100
    channel_grid(dc, "copy_share", "% of CUDA time copying", out, "fig14_channels_transfer.png", made,
                 logy=False, title="Multi-channel input: share of each CUDA call spent on copies")
    return d


def by_channel_config(d, value):
    config = (d.H.astype(int).astype(str) + "×" + d.W.astype(int).astype(str) +
              ", K=" + d.K.astype(int).astype(str) + ", C=" + d.C.astype(int).astype(str))
    t = d.assign(config=config, v=value).pivot_table(
        index=["H", "W", "K", "C", "config"], columns="impl", values="v", aggfunc="median")
    t = t.reset_index(level=["H", "W", "K", "C"], drop=True)
    t.index.name = "workload"
    return t.dropna(axis=1, how="all")


def channel_summary(d):
    if d is None or d.empty:
        return []
    p = ["\n## Multi-channel input: execution time, end-to-end (ms, median)\n",
         "FLOPs scale with C: 2 × C × K² × H × W. The OpenMP versions are single-channel and not included.\n",
         md(by_channel_config(d, d.median_ms))]
    dd = d[d.impl != "seq"]
    if not dd.empty:
        p += ["\n## Multi-channel input: speedup over seq at the same C, end-to-end\n",
              md(by_channel_config(dd, dd.speedup), "{:.2f}")]
    dc = d[d.is_cuda]
    if not dc.empty:
        p += ["\n## Multi-channel input: CUDA kernel-only speedup over seq\n",
              md(by_channel_config(dc, dc.speedup_kernel), "{:.2f}"),
              "\n## Multi-channel input: share of CUDA time spent copying (h2d + d2h)\n",
              md(by_channel_config(dc, (dc.h2d_ms + dc.d2h_ms) / (dc.h2d_ms + dc.kernel_ms + dc.d2h_ms) * 100),
                 "{:.0f}%")]
        if "torch_cudnn" in set(dc.impl):
            ref = dc[dc.impl == "torch_cudnn"].groupby(["H", "W", "K", "C"]).compute_ms.median()
            ours = dc[~dc.impl.str.startswith("torch")]
            if not ours.empty:
                ours = ours.assign(rel=ours.set_index(["H", "W", "K", "C"]).index.map(ref) / ours.compute_ms)
                p += ["\n## Multi-channel input: our CUDA kernels vs cuDNN (cuDNN kernel time / ours; >1 = ours faster)\n",
                      md(by_channel_config(ours, ours.rel), "{:.2f}")]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    res = Path(a.results)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent.parent / "plots" / res.name
    out.mkdir(parents=True, exist_ok=True)

    all_rows = load(res)
    failed = all_rows[all_rows["pass"] == 0].copy()
    df = all_rows[all_rows["pass"] != 0].copy()
    if df.empty:
        sys.exit("no usable rows remain after excluding failed checks")
    set_styles(df.impl.unique())
    made = []
    # Single-channel figures use the C = 1 sweeps; the channel sweeps get
    # their own figures, and fig_blocks takes both block sweeps.
    single = df[~df.experiment.isin(CHANNEL_EXPERIMENTS)].copy()
    fig_time_speedup(single, out, made)
    fig_cuda_kernel_vs_e2e(single, out, made)
    fig_cuda_breakdown(single, out, made)
    omp = fig_omp(single, out, made)
    blocks = fig_blocks(df, out, made)
    fig_throughput(single, out, made)
    best = fastest_table(single)
    fig_fastest(best, out, made)
    cuda_comparison = fig_cuda_variants(single, out, made)
    channels = fig_channels(df, out, made)
    summary(single, out, omp, blocks, best, failed, res, cuda_comparison, channels)
    print(f"{out}: " + ", ".join(made + ["summary.md"]))


if __name__ == "__main__":
    main()
