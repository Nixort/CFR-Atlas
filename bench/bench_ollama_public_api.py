#!/usr/bin/env python3
"""Benchmark documented non-streaming Ollama generation through its public API.

This harness measures a real model's public generation path. It does not measure
CFR-Atlas virtual-K/V execution because standard Ollama endpoints do not expose
the per-layer K/V tensors or deterministic page replay required by KvRegenerator.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:0.5b"

CONTEXT_PARAGRAPH = """CFR-Atlas virtualizes historical attention keys and values as deterministic pages. A conformant backend must replay the same token history, absolute positions, positional policy, K/V-head mapping, and storage rounding as its conventional stored-KV path. Cache residency may change latency and resident bytes, but it must not change the K/V values consumed by causal attention. The public Ollama API can report model information and run generation, but it does not publish per-layer K/V tensors or a page-replay interface. Therefore an Ollama public-API integration must keep virtual-K/V execution disabled until a sidecar exposes the required contract."""
PROMPT = "\n\n".join([CONTEXT_PARAGRAPH] * 12) + "\n\nIn exactly three concise sentences, explain the correctness boundary."


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {body}") from error
    except URLError as error:
        raise RuntimeError(f"cannot reach Ollama at {base_url}: {error}") from error


def get_json(base_url: str, path: str) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}{path}", headers={"accept": "application/json"}, method="GET"
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise RuntimeError(f"cannot reach Ollama at {base_url}: {error}") from error


def milliseconds(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else nanoseconds / 1_000_000.0


def rate(tokens: int | None, duration_ns: int | None) -> float | None:
    if tokens is None or duration_ns is None or duration_ns <= 0:
        return None
    return tokens / (duration_ns / 1_000_000_000.0)


def generate_once(base_url: str, model: str, case: str) -> tuple[dict[str, Any], float]:
    prompt = f"[benchmark-case:{case}]\n{PROMPT}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": 2048,
            "num_predict": 64,
            "temperature": 0,
            "seed": 42,
        },
    }
    started = time.perf_counter_ns()
    response = post_json(base_url, "/api/generate", payload)
    wall_ns = time.perf_counter_ns() - started
    if not response.get("done"):
        raise RuntimeError("Ollama generate response did not report done=true")
    return response, wall_ns / 1_000_000.0


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be at least one")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    tags = get_json(arguments.base_url, "/api/tags")
    names = {item.get("name") for item in tags.get("models", [])}
    if arguments.model not in names:
        available = ", ".join(sorted(name for name in names if name)) or "none"
        raise RuntimeError(f"model {arguments.model!r} is not installed; available: {available}")

    model_summary = next(item for item in tags["models"] if item.get("name") == arguments.model)
    model_record = post_json(arguments.base_url, "/api/show", {"model": arguments.model})
    (arguments.output_dir / "ollama_public_api_model_record.json").write_text(
        json.dumps(
            {"tags_model_summary": model_summary, "show_response": model_record},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # Warm-up performs model load and is intentionally excluded from measured rows.
    # Each case has a distinct leading marker so Ollama cannot reuse the prior full prompt prefix.
    generate_once(arguments.base_url, arguments.model, "warmup")

    rows: list[dict[str, Any]] = []
    for run in range(1, arguments.runs + 1):
        response, wall_ms = generate_once(arguments.base_url, arguments.model, f"measurement-{run}")
        rows.append(
            {
                "run": run,
                "prompt_case": f"measurement-{run}",
                "model": response.get("model", arguments.model),
                "wall_ms": round(wall_ms, 3),
                "total_ms": round(milliseconds(response.get("total_duration")) or 0.0, 3),
                "load_ms": round(milliseconds(response.get("load_duration")) or 0.0, 3),
                "prompt_tokens": response.get("prompt_eval_count"),
                "prompt_ms": round(milliseconds(response.get("prompt_eval_duration")) or 0.0, 3),
                "prompt_tokens_per_second": round(
                    rate(response.get("prompt_eval_count"), response.get("prompt_eval_duration")) or 0.0, 3
                ),
                "generated_tokens": response.get("eval_count"),
                "generation_ms": round(milliseconds(response.get("eval_duration")) or 0.0, 3),
                "generated_tokens_per_second": round(
                    rate(response.get("eval_count"), response.get("eval_duration")) or 0.0, 3
                ),
                "response_characters": len(response.get("response", "")),
            }
        )

    raw_path = arguments.output_dir / "ollama_public_api_raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "benchmark": "ollama_public_api_generation",
        "model": arguments.model,
        "model_identity": {
            "digest": model_summary.get("digest"),
            "size_bytes": model_summary.get("size"),
            "format": model_summary.get("details", {}).get("format"),
            "family": model_summary.get("details", {}).get("family"),
            "quantization_level": model_summary.get("details", {}).get("quantization_level"),
        },
        "base_url": arguments.base_url,
        "runs": arguments.runs,
        "warmup_excluded": True,
        "prompt_policy": {
            "prompt_tokens_observed": rows[0]["prompt_tokens"],
            "num_ctx": 2048,
            "num_predict": 64,
            "temperature": 0,
            "seed": 42,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "means": {
            field: round(mean([float(row[field]) for row in rows]), 3)
            for field in (
                "wall_ms",
                "total_ms",
                "load_ms",
                "prompt_ms",
                "prompt_tokens_per_second",
                "generation_ms",
                "generated_tokens_per_second",
            )
        },
        "exact_kv_status": "unavailable_through_public_api",
        "interpretation": (
            "This is an end-to-end measurement of documented Ollama public generation. "
            "It is not a CFR-Atlas virtual-K/V or exact-attention benchmark."
        ),
    }
    summary_path = arguments.output_dir / "ollama_public_api_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"raw results: {raw_path}")
    print(f"model record: {arguments.output_dir / 'ollama_public_api_model_record.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
