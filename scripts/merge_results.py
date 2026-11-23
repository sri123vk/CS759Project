#!/usr/bin/env python3
"""Splice a partial rerun into an earlier benchmark run.

Usage: merge_results.py BASE RERUN OUT [--impl PREFIX ...]

Copies BASE to OUT, then, in every CSV, replaces the rows whose implementation
starts with one of the PREFIXes (default: omp) with the matching rows from
RERUN. Everything else in BASE (seq, CUDA, PyTorch, ...) is kept as it was, so
speedups in plot.py still use BASE's seq timings. Rows of other
implementations in RERUN are ignored.

Example: only the OpenMP versions changed since results/euler-127842, so
rerun them alone and combine:

    scripts/merge_results.py results/euler-127842 results/euler-NNNNNN-omp \\
        results/euler-127842+omp-NNNNNN

Both runs should come from the same node type with the same CPU count
(compare their sysinfo.txt, which is combined into OUT's). Standard library
only, so it runs anywhere bench.sh does.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else (None, [])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base")
    ap.add_argument("rerun")
    ap.add_argument("out")
    ap.add_argument("--impl", nargs="+", default=["omp"], metavar="PREFIX",
                    help="implementation-name prefixes to replace (default: omp)")
    a = ap.parse_args()
    base, rerun, out = Path(a.base), Path(a.rerun), Path(a.out)
    prefixes = tuple(a.impl)
    for d in (base, rerun):
        if not d.is_dir():
            sys.exit(f"error: {d} is not a directory")
    if out.exists():
        sys.exit(f"error: {out} already exists; remove it or pick another name")

    shutil.copytree(base, out)

    for new_csv in sorted(rerun.glob("*.csv")):
        new_header, new_rows = read(new_csv)
        if new_header is None:
            continue
        impl_col = new_header.index("impl")
        new_rows = [r for r in new_rows if r[impl_col].startswith(prefixes)]
        target = out / new_csv.name
        if target.exists():
            header, rows = read(target)
            if header != new_header:
                sys.exit(f"error: {new_csv.name}: columns differ between {base} and {rerun}\n"
                         f"  {header}\n  {new_header}")
            kept = [r for r in rows if not r[impl_col].startswith(prefixes)]
        else:
            header, rows, kept = new_header, [], []
        dropped = len(rows) - len(kept)
        with open(target, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(kept + new_rows)
        print(f"{new_csv.name}: kept {len(kept)} rows, replaced {dropped} with {len(new_rows)}",
              file=sys.stderr)

    # Logs and system info: keep BASE's, append RERUN's under a heading.
    for name in ("sysinfo.txt", "errors.log", "skipped.log"):
        src = rerun / name
        if src.exists() and src.stat().st_size:
            with open(out / name, "a") as f:
                f.write(f"\n## rerun of {', '.join(p + '*' for p in prefixes)} "
                        f"from {rerun}\n")
                f.write(src.read_text())
    (out / "MERGED.txt").write_text(
        f"base:  {base}\nrerun: {rerun}\n"
        f"rows for implementations starting with {', '.join(prefixes)} come from the rerun; "
        f"all other rows from the base.\n")
    print(f"merged -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
