#!/usr/bin/env python3
"""Render the real Transformers-to-CFR benchmark only from saved raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse input artifact locations and result stems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfr-raw", type=Path, required=True)
    parser.add_argument("--transformers-raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--chart", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required CSV artifact with a non-empty schema."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def summarize(values: list[float]) -> dict[str, float]:
    """Calculate a compact robust summary."""
    return {
        "runs": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    """Create report JSON and a publication-ready chart."""
    args = parse_args()
    cfr_rows = read_csv(args.cfr_raw)
    transformer_rows = read_csv(args.transformers_raw)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    transformer = {
        mode: summarize([float(row["seconds"]) for row in transformer_rows if row["mode"] == mode])
        for mode in ("full_prefill", "page_replay")
    }
    cfr = {
        scenario: summarize([float(row["elapsed_us"]) / 1000.0 for row in cfr_rows if row["scenario"] == scenario])
        for scenario in ("cfr_cold", "cfr_hot_recent_page")
    }
    if not all(summary["runs"] for summary in transformer.values()) or not all(
        summary["runs"] for summary in cfr.values()
    ):
        raise ValueError("one or more required benchmark scenarios are missing")

    cfr_conformance = max(float(row["output_max_abs_diff"]) for row in cfr_rows)
    direct_python_conformance = max(float(row["direct_python_max_abs_diff"]) for row in cfr_rows)
    latest_hot = next(row for row in cfr_rows if row["scenario"] == "cfr_hot_recent_page")
    latest_cold = next(row for row in cfr_rows if row["scenario"] == "cfr_cold")
    full_kv_bytes = int(manifest["selected_kv_bytes_f32"])
    page_kv_bytes = int(manifest["selected_page_kv_bytes_f32"])

    summary = {
        "schema": 1,
        "scope": "real Qwen2.5-0.5B K/V page replay plus CFR one-layer/one-KV-head folded-attention bridge",
        "model": manifest["model"],
        "resolved_revision": manifest["resolved_revision"],
        "context_tokens": manifest["context_tokens"],
        "page_tokens": manifest["page_tokens"],
        "target": {
            "layer": manifest["layer"],
            "kv_head": manifest["kv_head"],
            "query_head": manifest["query_head"],
            "head_dim": manifest["head_dim"],
        },
        "transformers_wall_seconds": transformer,
        "cfr_folded_attention_milliseconds": cfr,
        "page_replay_to_full_prefill_median_ratio": (
            transformer["page_replay"]["median"] / transformer["full_prefill"]["median"]
        ),
        "conformance": {
            "declared_model_atol": manifest["max_abs_tolerance"],
            "all_layers_k_max_abs_diff": manifest["page_replay_all_layers_k_max_abs_diff"],
            "all_layers_v_max_abs_diff": manifest["page_replay_all_layers_v_max_abs_diff"],
            "page_replay_final_logit_max_abs_diff": manifest["page_replay_final_logit_max_abs_diff"],
            "incremental_cache_final_logit_max_abs_diff": manifest[
                "incremental_cache_final_logit_max_abs_diff"
            ],
            "python_direct_attention_max_abs_diff": direct_python_conformance,
            "cfr_to_direct_attention_max_abs_diff": cfr_conformance,
        },
        "selected_head_kv_memory": {
            "full_kv_bytes": full_kv_bytes,
            "cfr_cold_resident_bytes": int(latest_cold["hot_cache_bytes"]),
            "cfr_hot_resident_bytes": int(latest_hot["hot_cache_bytes"]),
            "cold_scratch_bytes": page_kv_bytes,
            "hot_scratch_bytes": page_kv_bytes,
            "full_to_cold_scratch_ratio": full_kv_bytes / page_kv_bytes,
            "full_to_hot_resident_ratio": full_kv_bytes / int(latest_hot["hot_cache_bytes"]),
            "cold_regenerations": int(latest_cold["cold_regenerations"]),
            "hot_regenerations": int(latest_hot["cold_regenerations"]),
            "hot_hits": int(latest_hot["hot_hits"]),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)
    colors = ["#355C7D", "#C06C84"]

    time_labels = ["Full\nprefill", "All-page\nreplay"]
    time_values = [
        transformer["full_prefill"]["median"],
        transformer["page_replay"]["median"],
    ]
    time_axis = axes[0]
    bars = time_axis.bar(time_labels, time_values, color=colors, width=0.58)
    time_axis.set_title("Real Qwen2.5 model wall time")
    time_axis.set_ylabel("seconds; median of 3 runs")
    time_axis.set_ylim(0.0, max(time_values) * 1.24)
    for bar, value in zip(bars, time_values, strict=True):
        time_axis.text(bar.get_x() + bar.get_width() / 2, value + max(time_values) * 0.035, f"{value:.3f}s", ha="center", va="bottom", fontweight="bold")
    time_axis.text(
        0.5,
        -0.27,
        f"All-page replay = {summary['page_replay_to_full_prefill_median_ratio']:.2f}× full prefill.\n"
        "Replay computes eight causal prefixes; it is not the folded-attention time.",
        ha="center",
        va="top",
        transform=time_axis.transAxes,
        fontsize=9,
    )

    memory_axis = axes[1]
    memory_labels = ["Full\nK/V", "CFR cold\nscratch", "CFR hot\nresident"]
    memory_values = [full_kv_bytes / 1024.0, page_kv_bytes / 1024.0, int(latest_hot["hot_cache_bytes"]) / 1024.0]
    memory_bars = memory_axis.bar(memory_labels, memory_values, color=["#355C7D", "#F67280", "#6C5B7B"], width=0.58)
    memory_axis.set_title("Selected real-model K/V head")
    memory_axis.set_ylabel("KiB of f32 K/V")
    memory_axis.set_ylim(0.0, max(memory_values) * 1.24)
    for bar, value in zip(memory_bars, memory_values, strict=True):
        memory_axis.text(bar.get_x() + bar.get_width() / 2, value + max(memory_values) * 0.035, f"{value:.0f} KiB", ha="center", va="bottom", fontweight="bold")
    memory_axis.text(
        0.5,
        -0.27,
        "CFR cold retains no K/V page; it reuses one 64-token scratch page.\n"
        "Hot keeps the newest page: one hit and seven regenerations per measured request.",
        ha="center",
        va="top",
        transform=memory_axis.transAxes,
        fontsize=9,
    )

    figure.suptitle("CFR-Atlas real Transformers K/V conformance benchmark", fontsize=15, fontweight="bold")
    args.chart.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.chart, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
