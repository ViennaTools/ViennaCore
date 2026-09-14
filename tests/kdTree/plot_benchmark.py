#!/usr/bin/env python3
"""Plot KDTreeBenchmark CSV output (requires matplotlib)."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "results" / "kdtree_benchmark.csv",
    )
    parser.add_argument(
        "--output-prefix", type=Path, help="Output path without .png/.svg suffix"
    )
    args = parser.parse_args()
    lines = args.csv_file.read_text().splitlines()
    comments = [line.removeprefix("# ") for line in lines if line.startswith("#")]
    rows = list(csv.DictReader(line for line in lines if line and not line.startswith("#")))
    if not rows:
        parser.error("CSV contains no benchmark results.")
    rows.sort(key=lambda row: int(row["points"]))
    points = [int(row["points"]) for row in rows]
    query_counts = {int(row["queries"]) for row in rows}
    if any(count <= 0 for count in points) or len(query_counts) != 1:
        parser.error("Point counts must be positive and query counts must be constant.")
    query_count = query_counts.pop()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    for axis, operation, title in zip(
        axes, ("build", "lookup"), ("Build", f"Radius search · {query_count:,} queries")
    ):
        for backend, color, marker in (
            ("KDTree", "#2563eb", "o"),
            ("NFKDTree", "#e07818", "s"),
        ):
            times = [float(row[f"{backend}_{operation}_min_ms"]) for row in rows]
            if any(not math.isfinite(time) or time <= 0 for time in times):
                parser.error("Times must be finite and positive for logarithmic plots.")
            axis.loglog(points, times, label=backend, color=color, marker=marker, linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Points in tree")
        axis.set_ylabel("Minimum time (ms)")
        axis.set_xticks(points)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(True, which="major", alpha=0.3)
        axis.grid(True, which="minor", alpha=0.1)
        axis.legend(frameon=False)

    details = [
        line for line in comments
        if line.startswith(("Radius:", "Maximum build threads:", "Build threads:"))
    ]
    fig.suptitle("KD-tree benchmark · minimum of 10 runs", fontsize=15)
    if details:
        fig.supxlabel("\n".join(details), fontsize=9)
    prefix = args.output_prefix or args.csv_file.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        output = Path(f"{prefix}.{extension}")
        fig.savefig(output, dpi=180)
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
