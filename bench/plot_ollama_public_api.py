#!/usr/bin/env python3
"""Render a chart from bench_ollama_public_api.py raw measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    arguments = parser.parse_args()
    raw_path = arguments.input_dir / "ollama_public_api_raw.csv"
    summary_path = arguments.input_dir / "ollama_public_api_summary.json"
    with raw_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError("raw benchmark CSV has no measurement rows")

    run_labels = [f"run {row['run']}" for row in rows]
    prompt_ms = [float(row["prompt_ms"]) for row in rows]
    generation_ms = [float(row["generation_ms"]) for row in rows]
    total_ms = [float(row["total_ms"]) for row in rows]
    overhead_ms = [max(0.0, total - prompt - generation) for total, prompt, generation in zip(total_ms, prompt_ms, generation_ms)]
    prompt_rate = [float(row["prompt_tokens_per_second"]) for row in rows]
    generation_rate = [float(row["generated_tokens_per_second"]) for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (latency_axis, rate_axis) = plt.subplots(1, 2, figsize=(12, 5.4))
    figure.subplots_adjust(bottom=0.22, top=0.84, wspace=0.25)
    figure.suptitle(
        f"Ollama public-API benchmark — {summary['model']} ({summary['runs']} measured runs)",
        y=0.96,
        fontsize=15,
        fontweight="bold",
    )

    latency_axis.bar(run_labels, prompt_ms, color="#2563eb", label="prompt evaluation")
    latency_axis.bar(run_labels, generation_ms, bottom=prompt_ms, color="#f97316", label="generation")
    latency_axis.bar(
        run_labels,
        overhead_ms,
        bottom=[prompt + generation for prompt, generation in zip(prompt_ms, generation_ms)],
        color="#94a3b8",
        label="other service time",
    )
    latency_axis.set_title("Measured server time")
    latency_axis.set_ylabel("milliseconds")
    latency_axis.legend(loc="upper right", frameon=True)

    index = list(range(len(rows)))
    width = 0.38
    rate_axis.bar([value - width / 2 for value in index], prompt_rate, width=width, color="#2563eb", label="prompt")
    rate_axis.bar([value + width / 2 for value in index], generation_rate, width=width, color="#f97316", label="generation")
    rate_axis.set_xticks(index, run_labels)
    rate_axis.set_title("Observed token rate")
    rate_axis.set_ylabel("tokens / second")
    rate_axis.legend(loc="upper right", frameon=True)
    figure.text(
        0.5,
        0.09,
        f"Mean total: {summary['means']['total_ms']:.1f} ms | mean wall: {summary['means']['wall_ms']:.1f} ms | "
        f"mean prompt: {summary['means']['prompt_tokens_per_second']:.1f} tok/s | "
        f"mean generation: {summary['means']['generated_tokens_per_second']:.1f} tok/s",
        ha="center",
        fontsize=9,
        color="#334155",
    )
    figure.text(
        0.5,
        0.035,
        "Public Ollama generation only; not a CFR-Atlas exact K/V or virtual-memory measurement.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    output_path = arguments.input_dir / "ollama_public_api.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
