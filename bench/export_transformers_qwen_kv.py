#!/usr/bin/env python3
"""Export real Qwen2.5 K/V evidence for the CFR-Atlas exactness benchmark.

The script deliberately keeps the model runtime outside the Rust core. It loads a
real Hugging Face causal-language model, uses a public corpus for token input,
materializes a conventional full K/V baseline, and independently recreates every
page by replaying the matching causal token prefix. It writes only the selected
layer/K/V-head values that the Rust benchmark needs, plus a machine-readable
manifest and raw timing samples.

The exported tensors are transient benchmark input. `bench/data/` is ignored so
large model-derived artifacts are never committed. Derived raw CSV/JSON and the
human-readable report live under `results/` after a verified run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers import __version__ as transformers_version
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_CORPUS = Path("bench/data/tinyshakespeare.txt")
DEFAULT_OUTPUT = Path("bench/data/transformers_qwen2_5_0_5b")


def parse_args() -> argparse.Namespace:
    """Parse reproducible benchmark parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--page-tokens", type=int, default=64)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kv-head", type=int, default=0)
    parser.add_argument("--query-head", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--atol",
        type=float,
        default=5.0e-3,
        help="maximum f32 absolute conformance difference across K/V and logits",
    )
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    """Raise a clear error for an invalid benchmark parameter."""
    if not condition:
        raise ValueError(message)


def save_f32(path: Path, tensor: torch.Tensor) -> None:
    """Write a tensor as portable contiguous little-endian f32 values."""
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    values.astype("<f4", copy=False).tofile(path)


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return a finite f32-oriented maximum absolute difference."""
    return float((left.float() - right.float()).abs().max().item())


def timed(callable_: Any, runs: int, mode: str, context_tokens: int, page_tokens: int) -> list[dict[str, Any]]:
    """Measure one deterministic CPU operation repeatedly and return raw rows."""
    rows: list[dict[str, Any]] = []
    for run in range(1, runs + 1):
        started = time.perf_counter()
        callable_()
        seconds = time.perf_counter() - started
        rows.append(
            {
                "mode": mode,
                "run": run,
                "seconds": f"{seconds:.9f}",
                "context_tokens": context_tokens,
                "page_tokens": page_tokens,
            }
        )
    return rows


def main() -> None:
    """Execute model-backed K/V export and self-conformance validation."""
    args = parse_args()
    require(args.context_tokens > 0, "context-tokens must be positive")
    require(args.page_tokens > 0, "page-tokens must be positive")
    require(args.context_tokens % args.page_tokens == 0, "context must divide exactly into pages")
    require(args.runs > 0, "runs must be positive")
    require(args.atol > 0.0 and math.isfinite(args.atol), "atol must be finite and positive")
    require(args.threads > 0, "threads must be positive")
    require(args.corpus.is_file(), f"corpus does not exist: {args.corpus}")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = args.corpus.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    encoded = tokenizer(corpus, add_special_tokens=False, return_tensors="pt").input_ids
    require(encoded.shape[1] >= args.context_tokens, "corpus tokenization is shorter than context")
    input_ids = encoded[:, : args.context_tokens].contiguous()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )
    model.eval()
    config = model.config
    require(config.model_type == "qwen2", f"expected Qwen2 model, got {config.model_type}")
    require(0 <= args.layer < config.num_hidden_layers, "layer is outside model topology")
    require(0 <= args.kv_head < config.num_key_value_heads, "kv-head is outside model topology")
    require(0 <= args.query_head < config.num_attention_heads, "query-head is outside model topology")
    group_size = config.num_attention_heads // config.num_key_value_heads
    require(args.query_head // group_size == args.kv_head, "query-head must map to selected kv-head")

    captured_query: list[torch.Tensor] = []
    attention = model.model.layers[args.layer].self_attn

    def capture_query(_module: torch.nn.Module, positional: tuple[Any, ...], keywords: dict[str, Any]) -> None:
        hidden_states = keywords.get("hidden_states")
        if hidden_states is None:
            hidden_states = positional[0]
        position_embeddings = keywords["position_embeddings"]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query_states = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        captured_query.append(query_states[0, args.query_head, -1].detach().clone())

    hook = attention.register_forward_pre_hook(capture_query, with_kwargs=True)
    try:
        with torch.inference_mode():
            baseline = model(input_ids=input_ids, use_cache=True, return_dict=True)
    finally:
        hook.remove()
    require(len(captured_query) == 1, "query hook did not observe exactly one selected-layer forward")

    baseline_cache = baseline.past_key_values
    baseline_k = baseline_cache.key_cache[args.layer][0, args.kv_head].detach().clone()
    baseline_v = baseline_cache.value_cache[args.layer][0, args.kv_head].detach().clone()
    query = captured_query[0]
    scale = 1.0 / math.sqrt(float(attention.head_dim))
    scores = torch.matmul(query, baseline_k.transpose(0, 1)) * scale
    baseline_attention = torch.matmul(torch.softmax(scores, dim=-1), baseline_v)
    baseline_logits = baseline.logits[:, -1, :].detach().clone()

    replay_k_pages: list[torch.Tensor] = []
    replay_v_pages: list[torch.Tensor] = []
    max_k_diff = 0.0
    max_v_diff = 0.0
    max_all_layers_k_diff = 0.0
    max_all_layers_v_diff = 0.0
    replay_logits: torch.Tensor | None = None

    def replay_all_pages(capture: bool) -> None:
        nonlocal max_k_diff, max_v_diff, max_all_layers_k_diff, max_all_layers_v_diff, replay_logits
        local_k_pages: list[torch.Tensor] = []
        local_v_pages: list[torch.Tensor] = []
        local_max_k = 0.0
        local_max_v = 0.0
        local_all_k = 0.0
        local_all_v = 0.0
        local_logits: torch.Tensor | None = None
        with torch.inference_mode():
            for end in range(args.page_tokens, args.context_tokens + 1, args.page_tokens):
                page_output = model(input_ids=input_ids[:, :end], use_cache=True, return_dict=True)
                page_cache = page_output.past_key_values
                start = end - args.page_tokens
                for layer_index in range(config.num_hidden_layers):
                    local_all_k = max(
                        local_all_k,
                        max_abs_diff(
                            page_cache.key_cache[layer_index][:, :, start:end, :],
                            baseline_cache.key_cache[layer_index][:, :, start:end, :],
                        ),
                    )
                    local_all_v = max(
                        local_all_v,
                        max_abs_diff(
                            page_cache.value_cache[layer_index][:, :, start:end, :],
                            baseline_cache.value_cache[layer_index][:, :, start:end, :],
                        ),
                    )
                selected_k = page_cache.key_cache[args.layer][0, args.kv_head, start:end, :].detach().clone()
                selected_v = page_cache.value_cache[args.layer][0, args.kv_head, start:end, :].detach().clone()
                local_k_pages.append(selected_k)
                local_v_pages.append(selected_v)
                local_max_k = max(local_max_k, max_abs_diff(selected_k, baseline_k[start:end, :]))
                local_max_v = max(local_max_v, max_abs_diff(selected_v, baseline_v[start:end, :]))
                local_logits = page_output.logits[:, -1, :].detach().clone()
        if capture:
            replay_k_pages[:] = local_k_pages
            replay_v_pages[:] = local_v_pages
            max_k_diff = local_max_k
            max_v_diff = local_max_v
            max_all_layers_k_diff = local_all_k
            max_all_layers_v_diff = local_all_v
            replay_logits = local_logits

    replay_all_pages(capture=True)
    require(replay_logits is not None, "page replay did not produce final logits")
    replay_k = torch.cat(replay_k_pages, dim=0)
    replay_v = torch.cat(replay_v_pages, dim=0)
    require(max_k_diff <= args.atol, f"selected K page replay mismatch: {max_k_diff}")
    require(max_v_diff <= args.atol, f"selected V page replay mismatch: {max_v_diff}")
    require(max_all_layers_k_diff <= args.atol, f"all-layer K page replay mismatch: {max_all_layers_k_diff}")
    require(max_all_layers_v_diff <= args.atol, f"all-layer V page replay mismatch: {max_all_layers_v_diff}")
    logit_max_abs_diff = max_abs_diff(replay_logits, baseline_logits)
    require(
        logit_max_abs_diff <= args.atol,
        f"model final-logit mismatch after page replay: {logit_max_abs_diff}",
    )

    incremental_cache = DynamicCache()
    incremental_logits: torch.Tensor | None = None
    with torch.inference_mode():
        for index in range(args.context_tokens):
            incremental_output = model(
                input_ids=input_ids[:, index : index + 1],
                past_key_values=incremental_cache,
                use_cache=True,
                return_dict=True,
            )
            incremental_cache = incremental_output.past_key_values
            incremental_logits = incremental_output.logits[:, -1, :].detach().clone()
    require(incremental_logits is not None, "incremental cache did not produce logits")
    incremental_logit_max_abs_diff = max_abs_diff(incremental_logits, baseline_logits)
    require(
        incremental_logit_max_abs_diff <= args.atol,
        f"incremental cache final-logit mismatch: {incremental_logit_max_abs_diff}",
    )

    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=True, return_dict=True)
        replay_all_pages(capture=False)
    timing_rows = timed(
        lambda: model(input_ids=input_ids, use_cache=True, return_dict=True),
        args.runs,
        "full_prefill",
        args.context_tokens,
        args.page_tokens,
    )
    timing_rows.extend(
        timed(
            lambda: replay_all_pages(capture=False),
            args.runs,
            "page_replay",
            args.context_tokens,
            args.page_tokens,
        )
    )

    save_f32(args.output_dir / "query_f32le.bin", query)
    save_f32(args.output_dir / "baseline_k_f32le.bin", baseline_k)
    save_f32(args.output_dir / "baseline_v_f32le.bin", baseline_v)
    save_f32(args.output_dir / "replayed_k_f32le.bin", replay_k)
    save_f32(args.output_dir / "replayed_v_f32le.bin", replay_v)
    save_f32(args.output_dir / "baseline_attention_f32le.bin", baseline_attention)

    with (args.output_dir / "timing_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "schema": 1,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(config, "_commit_hash", None),
        "model_type": config.model_type,
        "parameter_count": parameter_count,
        "context_tokens": args.context_tokens,
        "page_tokens": args.page_tokens,
        "layer": args.layer,
        "kv_head": args.kv_head,
        "query_head": args.query_head,
        "head_dim": attention.head_dim,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "attention_scale": scale,
        "dtype": "float32",
        "max_abs_tolerance": args.atol,
        "attn_implementation": "eager",
        "corpus": str(args.corpus),
        "corpus_bytes": args.corpus.stat().st_size,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "cpu_threads": args.threads,
        "page_replay_selected_k_max_abs_diff": max_k_diff,
        "page_replay_selected_v_max_abs_diff": max_v_diff,
        "page_replay_all_layers_k_max_abs_diff": max_all_layers_k_diff,
        "page_replay_all_layers_v_max_abs_diff": max_all_layers_v_diff,
        "page_replay_final_logit_max_abs_diff": logit_max_abs_diff,
        "incremental_cache_final_logit_max_abs_diff": incremental_logit_max_abs_diff,
        "selected_kv_bytes_f32": int(baseline_k.numel() * baseline_k.element_size() * 2),
        "selected_page_kv_bytes_f32": int(args.page_tokens * attention.head_dim * 4 * 2),
        "all_layers_kv_bytes_f32": int(
            config.num_hidden_layers
            * config.num_key_value_heads
            * args.context_tokens
            * attention.head_dim
            * 4
            * 2
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
