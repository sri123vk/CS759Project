#!/usr/bin/env python3
"""Plot all CUDA variants from a results directory, CSV, or Slurm log.

  python plot_cuda_blocks.py results/bench_123 --metric kernel_ms
  python plot_cuda_blocks.py results/bench_123/cuda_blocks.csv --metric median_ms
  python plot_cuda_blocks.py conv_variants_123.out -o plots

Slurm progress lines contain only median_ms. For kernel_ms use a CSV.
"""
import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CUDA_ORDER = ["cuda", "cuda_opt", "cuda_const", "cuda_tiled_const", "cuda_multiout",
              "cuda_multiout_const", "cuda_multiout_tiled", "cuda_multiout_tiled_const"]
COLORS = ["#2674bc", "#c85a17", "#16815d", "#aa3377", "#7950a5", "#9a6500", "#007b80", "#b33d3d"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def order_impls(names):
    return sorted(names, key=lambda n: (CUDA_ORDER.index(n) if n in CUDA_ORDER else len(CUDA_ORDER), n))


def sweep_from_name(path):
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    return stem if stem in ("baseline", "cuda_blocks") else None


def parse_csv(path, metric, sweep, records, counts):
    sweep = sweep or sweep_from_name(path)
    if sweep is None:
        raise ValueError(f"Cannot identify {path.name}; supply --sweep baseline or --sweep cuda_blocks.")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"impl", "H", "W", "K", "bx", "by", metric} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, 2):
            impl = (row.get("impl") or "").strip()
            if not impl.startswith("cuda"):
                continue
            status = (row.get("pass") or "").strip().lower()
            if status in ("0", "0.0", "false"):
                counts["failed"] += 1
                continue
            if not status:
                counts["unverified"] += 1
            try:
                h, w, k, bx, by = (int(row[v]) for v in ("H", "W", "K", "bx", "by"))
                value = float(row[metric])
                if min(h, w, k, bx, by) <= 0 or not math.isfinite(value) or value <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                counts["invalid"] += 1
                print(f"Skipping invalid {metric} or configuration at {path.name}:{line_number}", file=sys.stderr)
                continue
            records[(sweep, h, w, k, bx, by, impl)].append(value)
            counts["rows"] += 1


def parse_log(path, metric, requested_sweep, records, counts):
    if metric != "median_ms":
        raise ValueError("Slurm progress lines contain only median_ms. Use a benchmark CSV for kernel_ms.")
    section = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        text = line.strip()
        if text.startswith("== "):
            section = text[3:].strip()
            continue
        if section not in ("baseline", "cuda_blocks") or (requested_sweep and section != requested_sweep):
            continue
        parts = text.split()
        if not parts or parts[-1] != "ms" or "--impl" not in parts:
            continue
        try:
            impl = parts[parts.index("--impl") + 1]
            if not impl.startswith("cuda"):
                continue
            if "--size" in parts:
                h = w = int(parts[parts.index("--size") + 1])
            else:
                h, w = int(parts[parts.index("--H") + 1]), int(parts[parts.index("--W") + 1])
            k = int(parts[parts.index("--K") + 1])
            b = parts.index("--block")
            bx, by, value = int(parts[b + 1]), int(parts[b + 2]), float(parts[-2])
            if min(h, w, k, bx, by) <= 0 or not math.isfinite(value) or value <= 0:
                raise ValueError
        except (ValueError, IndexError):
            counts["invalid"] += 1
            print(f"Skipping non-numeric block or timing at {path.name}:{line_number}", file=sys.stderr)
            continue
        records[(section, h, w, k, bx, by, impl)].append(value)
        counts["rows"] += 1
        counts["unverified"] += 1


def make_page(pdf, data, impls, sweep, group, styles, metric):
    if sweep == "baseline":
        k, bx, by = group
        keys = [key for key in data if key[0] == sweep and (key[3], key[4], key[5]) == group]
        categories = sorted({(key[1], key[2]) for key in keys})
        category = lambda key: (key[1], key[2])
        title, xlabel = f"Baseline: {k}×{k} filter, {bx}×{by} block", "Image size (height × width)"
    else:
        h, w, k = group
        keys = [key for key in data if key[0] == sweep and (key[1], key[2], key[3]) == group]
        categories = sorted({(key[4], key[5]) for key in keys}, key=lambda b: (b[0] * b[1], b[0], b[1]))
        category = lambda key: (key[4], key[5])
        title, xlabel = f"Block sweep: {h}×{w} image, {k}×{k} filter", "Block shape (x × y)"
    positions = {value: i for i, value in enumerate(categories)}
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for impl in impls:
        points = sorted((positions[category(key)], data[key]) for key in keys if key[6] == impl)
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], label=impl,
                    linewidth=1.8, markersize=6, **styles[impl])
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([f"{a}×{b}" for a, b in categories], rotation=35 if len(categories) > 4 else 0,
                       ha="right" if len(categories) > 4 else "center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Kernel time (ms; CUDA-event median)" if metric == "kernel_ms"
                  else "Full-call time (ms; wall-clock median)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.tight_layout(rect=(0, 0.20 if len(labels) > 4 else 0.13, 1, 1))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Benchmark results directory, one CSV, or a Slurm output log")
    ap.add_argument("-o", "--output-dir", default="cuda_median_plots")
    ap.add_argument("--metric", choices=("median_ms", "kernel_ms"), default="median_ms")
    ap.add_argument("--sweep", choices=("baseline", "cuda_blocks"), help="Select a sweep or label a renamed CSV")
    ap.add_argument("--impls", nargs="+", help="Only plot these CUDA implementations")
    args = ap.parse_args()
    path = Path(args.input)
    records, counts = defaultdict(list), Counter()
    try:
        if path.is_dir():
            files = sorted(p for p in path.glob("*.csv") if sweep_from_name(p))
            if args.sweep:
                files = [p for p in files if sweep_from_name(p) == args.sweep]
            if not files:
                raise ValueError(f"No baseline.csv or cuda_blocks.csv found in {path}")
            for file in files:
                parse_csv(file, args.metric, sweep_from_name(file), records, counts)
        elif path.suffix.lower() == ".csv":
            parse_csv(path, args.metric, args.sweep, records, counts)
        else:
            parse_log(path, args.metric, args.sweep, records, counts)
    except (OSError, ValueError) as error:
        sys.exit(f"Cannot read results: {error}")
    if not records:
        sys.exit("No valid CUDA timing rows found.")
    repeated = sum(len(values) - 1 for values in records.values() if len(values) > 1)
    data = {key: median(values) for key, values in records.items()}
    all_impls = order_impls({key[6] for key in data})
    impls = [name for name in all_impls if not args.impls or name in args.impls]
    if not impls:
        sys.exit(f"No selected implementations found. Available: {', '.join(all_impls)}")
    styles = {name: {"color": COLORS[i % len(COLORS)], "marker": MARKERS[i % len(MARKERS)],
                     "linestyle": "-" if i < len(COLORS) else "--"} for i, name in enumerate(all_impls)}
    selected = {key: value for key, value in data.items() if key[6] in impls}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "cuda_median_comparison" if args.metric == "median_ms" else "cuda_kernel_comparison"
    csv_path, pdf_path = out / f"{prefix}.csv", out / f"{prefix}.pdf"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sweep", "H", "W", "K", "bx", "by", "impl", "metric", "time_ms",
                         "cuda_time_ms", "speedup_vs_cuda", "time_reduction_pct_vs_cuda"])
        for key in sorted(selected):
            sweep, h, w, k, bx, by, impl = key
            value = selected[key]
            base = data.get((sweep, h, w, k, bx, by, "cuda"))
            writer.writerow([sweep, h, w, k, bx, by, impl, args.metric, f"{value:.9g}",
                             f"{base:.9g}" if base is not None else "",
                             f"{base / value:.9g}" if base is not None else "",
                             f"{100 * (base - value) / base:.9g}" if base is not None else ""])
    baseline_groups = sorted({(k[3], k[4], k[5]) for k in selected if k[0] == "baseline"})
    block_groups = sorted({(k[1], k[2], k[3]) for k in selected if k[0] == "cuda_blocks"})
    with PdfPages(pdf_path) as pdf:
        for group in baseline_groups:
            make_page(pdf, selected, impls, "baseline", group, styles, args.metric)
        for group in block_groups:
            make_page(pdf, selected, impls, "cuda_blocks", group, styles, args.metric)
    print(f"Loaded {counts['rows']} CUDA rows; implementations: {', '.join(impls)}")
    if counts["failed"] or counts["invalid"]:
        print(f"Skipped {counts['failed']} explicitly failed rows and {counts['invalid']} invalid rows.")
    if counts["unverified"]:
        print(f"{counts['unverified']} rows have no recorded check status.")
    if repeated:
        print(f"Aggregated {repeated} duplicate entries using the median of reported medians.")
    print(f"Created {len(baseline_groups) + len(block_groups)} plot pages")
    print(f"PDF: {pdf_path}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
