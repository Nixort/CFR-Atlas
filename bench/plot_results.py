#!/usr/bin/env python3
"""Render CFR-Atlas benchmark raw CSV into a summary CSV and comparison chart."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib.pyplot as plt


COLORS = {
    "full_kv": "#64748b",
    "cfr_cold": "#0f766e",
    "cfr_hot": "#2563eb",
}
LABELS = {
    "full_kv": "Full K/V",
    "cfr_cold": "CFR cold",
    "cfr_hot": "CFR hot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path, help="raw CSV emitted by bench-cfr")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="PNG chart path")
    parser.add_argument("--summary", type=pathlib.Path, required=True, help="derived CSV path")
    return parser.parse_args()


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"benchmark CSV contains no data rows: {path}")
    expected = {"method", "median_ms", "full_kv_bytes", "cfr_resident_kv_bytes", "max_abs_diff"}
    missing = expected.difference(rows[0])
    if missing:
        raise ValueError(f"benchmark CSV misses columns: {', '.join(sorted(missing))}")
    return rows


def write_summary(rows: list[dict[str, str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "median_ms",
        "min_ms",
        "max_ms",
        "contexts_per_second",
        "full_kv_bytes",
        "resident_kv_bytes",
        "resident_memory_reduction_x",
        "speed_ratio_vs_full_kv",
        "hot_hits_per_run",
        "cold_regenerations_per_run",
        "max_abs_diff",
    ]
    baseline = next(row for row in rows if row["method"] == "full_kv")
    baseline_ms = float(baseline["median_ms"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            resident = int(row["cfr_resident_kv_bytes"])
            full = int(row["full_kv_bytes"])
            writer.writerow({
                "method": row["method"],
                "median_ms": row["median_ms"],
                "min_ms": row["min_ms"],
                "max_ms": row["max_ms"],
                "contexts_per_second": row["contexts_per_second"],
                "full_kv_bytes": full,
                "resident_kv_bytes": resident,
                "resident_memory_reduction_x": f"{full / resident:.2f}",
                "speed_ratio_vs_full_kv": f"{baseline_ms / float(row['median_ms']):.3f}",
                "hot_hits_per_run": row["hot_hits_per_run"],
                "cold_regenerations_per_run": row["cold_regenerations_per_run"],
                "max_abs_diff": row["max_abs_diff"],
            })


def render(rows: list[dict[str, str]], output: pathlib.Path) -> None:
    methods = [row["method"] for row in rows]
    labels = [LABELS.get(method, method) for method in methods]
    colors = [COLORS.get(method, "#475569") for method in methods]
    runtimes = [float(row["median_ms"]) for row in rows]
    resident_mib = [int(row["cfr_resident_kv_bytes"]) / (1024 * 1024) for row in rows]
    errors = [float(row["max_abs_diff"]) for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=180, constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("#f8fafc")
        axis.spines[["top", "right"]].set_visible(False)

    runtime_bars = axes[0].bar(labels, runtimes, color=colors, width=0.62)
    axes[0].set_title("Reference workload runtime", loc="left", weight="bold", color="#0f172a")
    axes[0].set_ylabel("Median wall time (ms)")
    for bar, value in zip(runtime_bars, runtimes, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9, color="#0f172a")

    memory_bars = axes[1].bar(labels, resident_mib, color=colors, width=0.62)
    axes[1].set_title("Active resident K/V scope", loc="left", weight="bold", color="#0f172a")
    axes[1].set_ylabel("MiB (full K/V vs CFR scratch + hot cache)")
    for bar, value in zip(memory_bars, resident_mib, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color="#0f172a")

    context = rows[0]["context_tokens"]
    head_dim = rows[0]["head_dim"]
    page_tokens = rows[0]["page_tokens"]
    runs = rows[0]["runs"]
    exact = max(errors) == 0.0
    fig.suptitle(
        f"CFR-Atlas · Tiny Shakespeare reference workload\n"
        f"context={context} · head_dim={head_dim} · page_tokens={page_tokens} · median of {runs} runs · exact={exact}",
        x=0.08,
        ha="left",
        fontsize=13,
        weight="bold",
        color="#0f172a",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    write_summary(rows, args.summary)
    render(rows, args.output)


if __name__ == "__main__":
    main()
