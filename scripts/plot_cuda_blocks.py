import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib

# Save plots without opening a graphical window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


parser = argparse.ArgumentParser(
    description="Compare CUDA median times from a Slurm output file."
)

parser.add_argument(
    "log_file",
    help="Path to the Slurm output file",
)

parser.add_argument(
    "-o",
    "--output-dir",
    default="cuda_median_plots",
    help="Directory for the PDF and comparison CSV",
)

args = parser.parse_args()
log_path = Path(args.log_file)
output_dir = Path(args.output_dir)

try:
    lines = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()
except OSError as error:
    sys.exit(f"Could not read {log_path}: {error}")


# Key: (sweep, image size, K, block x, block y)
# Value: {"cuda": [times], "cuda_opt": [times]}
records = defaultdict(lambda: {"cuda": [], "cuda_opt": []})

counts = Counter()
skipped = 0
section = None

for line in lines:
    text = line.strip()

    # bench.sh prints headings such as:
    # == baseline
    # == cuda_blocks
    if text.startswith("== "):
        section = text[3:].strip()
        continue

    if section not in ("baseline", "cuda_blocks"):
        continue

    parts = text.split()

    # Benchmark progress lines end with a time followed by "ms".
    if len(parts) < 2 or parts[-1] != "ms":
        continue

    if "--impl" not in parts:
        continue

    try:
        impl = parts[parts.index("--impl") + 1]

        if impl not in ("cuda", "cuda_opt"):
            continue

        size = int(parts[parts.index("--size") + 1])
        filter_size = int(parts[parts.index("--K") + 1])

        block_index = parts.index("--block")
        block_x = int(parts[block_index + 1])
        block_y = int(parts[block_index + 2])

        elapsed_ms = float(parts[-2])

        if (
            min(size, filter_size, block_x, block_y) <= 0
            or not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
        ):
            raise ValueError("Invalid configuration or timing")

    except (ValueError, IndexError):
        skipped += 1
        continue

    key = (section, size, filter_size, block_x, block_y)
    records[key][impl].append(elapsed_ms)
    counts[impl] += 1


print(
    f"Timing lines found: cuda={counts['cuda']}, "
    f"cuda_opt={counts['cuda_opt']}"
)

if skipped:
    print(
        f"Skipped {skipped} lines with missing or non-numeric "
        "size, filter, block, or timing values."
    )


# Match cuda and cuda_opt using exactly the same configuration.
matched = []
unmatched = 0

for key, values in sorted(records.items()):
    if not values["cuda"] or not values["cuda_opt"]:
        unmatched += 1
        continue

    sweep, size, filter_size, block_x, block_y = key

    # If the same configuration occurs more than once in the same
    # sweep, use the median of its reported median times.
    cuda_ms = median(values["cuda"])
    opt_ms = median(values["cuda_opt"])

    matched.append(
        (
            sweep,
            size,
            filter_size,
            block_x,
            block_y,
            cuda_ms,
            opt_ms,
        )
    )


if unmatched:
    print(
        f"Skipped {unmatched} configurations that did not have "
        "both cuda and cuda_opt results."
    )

if not matched:
    sys.exit(
        "No matching results found. Check that the log contains "
        "'== baseline' or '== cuda_blocks' sections with both "
        "implementations and numeric block dimensions."
    )


output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "matched_cuda_medians.csv"
pdf_path = output_dir / "cuda_median_comparison.pdf"


# Save the numerical comparison.
with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "sweep",
            "size",
            "K",
            "bx",
            "by",
            "cuda_median_ms",
            "cuda_opt_median_ms",
            "speedup_cuda_over_cuda_opt",
        ]
    )

    for sweep, size, k, bx, by, cuda_ms, opt_ms in matched:
        speedup = cuda_ms / opt_ms if opt_ms > 0 else None

        writer.writerow(
            [
                sweep,
                size,
                k,
                bx,
                by,
                f"{cuda_ms:.4f}",
                f"{opt_ms:.4f}",
                f"{speedup:.3f}" if speedup is not None else "",
            ]
        )


# Baseline pages: compare image sizes for each filter and block.
baseline_groups = defaultdict(list)

# Block-sweep pages: compare block shapes for each image and filter.
block_groups = defaultdict(list)

for sweep, size, k, bx, by, cuda_ms, opt_ms in matched:
    if sweep == "baseline":
        baseline_groups[(k, bx, by)].append(
            (size, cuda_ms, opt_ms)
        )

    elif sweep == "cuda_blocks":
        block_groups[(size, k)].append(
            (bx, by, cuda_ms, opt_ms)
        )


page_count = 0

with PdfPages(pdf_path) as pdf:

    # Plot the baseline sweep.
    for (k, bx, by), rows in sorted(baseline_groups.items()):
        rows.sort(key=lambda row: row[0])

        sizes = [str(row[0]) for row in rows]
        cuda_times = [row[1] for row in rows]
        opt_times = [row[2] for row in rows]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(sizes, cuda_times, marker="o", label="cuda")
        ax.plot(sizes, opt_times, marker="o", label="cuda_opt")

        ax.set_xlabel("Image size N (N × N)")
        ax.set_ylabel("Median full-call time (ms)")
        ax.set_title(f"Baseline: K={k}, block={bx}×{by}")

        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        page_count += 1

    # Plot the CUDA block-size sweep.
    for (size, k), rows in sorted(block_groups.items()):
        rows.sort(key=lambda row: (row[0] * row[1], row[0], row[1]))

        labels = [f"{row[0]}×{row[1]}" for row in rows]
        cuda_times = [row[2] for row in rows]
        opt_times = [row[3] for row in rows]

        positions = list(range(len(rows)))
        bar_width = 0.38

        fig, ax = plt.subplots(
            figsize=(max(9, len(rows) * 0.8), 5.5)
        )

        ax.bar(
            [x - bar_width / 2 for x in positions],
            cuda_times,
            width=bar_width,
            label="cuda",
        )

        ax.bar(
            [x + bar_width / 2 for x in positions],
            opt_times,
            width=bar_width,
            label="cuda_opt",
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")

        ax.set_xlabel("Block shape (bx × by)")
        ax.set_ylabel("Median full-call time (ms)")
        ax.set_title(f"Block sweep: image={size}×{size}, K={k}")

        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend()

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        page_count += 1


print(f"Compared {len(matched)} matched configurations.")
print(f"Created {page_count} plot pages.")
print(f"PDF: {pdf_path}")
print(f"CSV: {csv_path}")
