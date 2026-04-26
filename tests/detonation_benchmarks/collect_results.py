"""Collect and summarize benchmark results into a comparison table.

Usage:
    python collect_results.py results/replicator_sweep_20260421_120000/

Reads metrics.json from each run directory and produces a markdown table
and CSV file summarizing all runs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def collect_metrics(results_dir: Path) -> list[dict[str, Any]]:
    """Walk the results directory and collect all metrics.json files."""
    rows = []
    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        with open(metrics_file) as f:
            metrics = json.load(f)
        # Derive model and strategy from the path structure: results/<exp>/<model>/<strategy>/
        parts = metrics_file.relative_to(results_dir).parts
        if len(parts) >= 3:
            metrics["model"] = parts[-3] if len(parts) >= 3 else "unknown"
            metrics["strategy"] = parts[-2] if len(parts) >= 2 else "unknown"
        rows.append(metrics)
    return rows


def to_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Convert metrics rows into a markdown table."""
    if not rows:
        return "No results found."

    columns = ["model", "strategy", "total_time_sec", "train_iters"]
    # Add any extra columns present in all rows
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    for key in sorted(all_keys):
        if key not in columns:
            columns.append(key)

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"

    lines = [header, sep]
    for row in rows:
        cells = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def to_csv_file(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write metrics to a CSV file."""
    if not rows:
        return
    columns = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--format", choices=["markdown", "csv", "both"], default="both")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = collect_metrics(results_dir)

    if not rows:
        print(f"No results found in {results_dir}")
        return

    if args.format in ("markdown", "both"):
        md = to_markdown_table(rows)
        md_path = results_dir / "results_summary.md"
        with open(md_path, "w") as f:
            f.write(f"# Benchmark Results: {results_dir.name}\n\n")
            f.write(md)
            f.write("\n")
        print(md)
        print(f"\nMarkdown saved to {md_path}")

    if args.format in ("csv", "both"):
        csv_path = results_dir / "results_summary.csv"
        to_csv_file(rows, csv_path)
        print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
