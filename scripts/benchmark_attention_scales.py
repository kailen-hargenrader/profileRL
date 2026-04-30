#!/usr/bin/env python3
"""
Benchmark scaled_dot_product_attention at multiple scales.

- Batch size 8, effective single-head attention (Q/K/V last dim = d_model).
- Grid: d_model in {16, 32, 64, 128} × seq_len in {64, 128, 256, 512, 1024}.
- Times 100 forward passes and 100 backward passes (backward timed only).
- Records CUDA memory allocated after one forward (before backward).

Requires CUDA.

Use `--compile` to wrap attention in nn.Module and benchmark after torch.compile(model).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "basics"))

from basics.model import scaled_dot_product_attention  # noqa: E402


class AttentionModule(nn.Module):
    """Thin wrapper so attention can be passed to torch.compile(model)."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return scaled_dot_product_attention(q, k, v, mask)

D_MODELS = (16, 32, 64, 128)
SEQ_LENGTHS = (64, 128, 256, 512, 1024)
BATCH_SIZE = 8
NUM_FORWARD = 100
NUM_BACKWARD = 100
WARMUP = 5


def causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    seq = torch.arange(sequence_length, device=device)
    qi = seq.view(1, sequence_length, 1)
    kj = seq.view(1, 1, sequence_length)
    return qi >= kj


def benchmark_one(
    d_model: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    model: nn.Module,
) -> dict[str, float | str | int | None]:
    mask = causal_mask(seq_len, device)

    # --- Forward (no grad): warmup + timed ---
    q_fwd = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype)
    k_fwd = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype)
    v_fwd = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(q_fwd, k_fwd, v_fwd, mask)
            torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        forward_ms: list[float] = []
        for _ in range(NUM_FORWARD):
            start_ev.record()
            _ = model(q_fwd, k_fwd, v_fwd, mask)
            end_ev.record()
            torch.cuda.synchronize()
            forward_ms.append(start_ev.elapsed_time(end_ev))

    # --- Memory after forward (graph built), before backward ---
    torch.cuda.reset_peak_memory_stats()
    q_b = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k_b = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v_b = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    out_once = model(q_b, k_b, v_b, mask)
    torch.cuda.synchronize()
    mem_before_backward = torch.cuda.memory_allocated()
    peak_after_forward = torch.cuda.max_memory_allocated()

    grad_template = torch.randn_like(out_once)

    # --- Backward: warmup then time backward only ---
    for _ in range(WARMUP):
        q_w = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        k_w = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        v_w = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        o_w = model(q_w, k_w, v_w, mask)
        torch.cuda.synchronize()
        o_w.backward(torch.randn_like(o_w))
        torch.cuda.synchronize()

    backward_ms: list[float] = []
    start_b = torch.cuda.Event(enable_timing=True)
    end_b = torch.cuda.Event(enable_timing=True)
    q_b = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    k_b = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    v_b = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
    for _ in range(NUM_BACKWARD):
        out_b = model(q_b, k_b, v_b, mask)
        torch.cuda.synchronize()
        start_b.record()
        out_b.backward(grad_template)
        end_b.record()
        torch.cuda.synchronize()
        backward_ms.append(start_b.elapsed_time(end_b))
        q_b.grad = None
        k_b.grad = None
        v_b.grad = None

    return {
        "d_model": d_model,
        "seq_len": seq_len,
        "forward_mean_ms": statistics.mean(forward_ms),
        "forward_std_ms": statistics.stdev(forward_ms) if len(forward_ms) > 1 else 0.0,
        "backward_mean_ms": statistics.mean(backward_ms),
        "backward_std_ms": statistics.stdev(backward_ms) if len(backward_ms) > 1 else 0.0,
        "mem_before_backward_bytes": mem_before_backward,
        "peak_after_forward_bytes": peak_after_forward,
        "status": "ok",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark attention at multiple scales.")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Wrap attention in nn.Module and run torch.compile(model) before benchmarking.",
    )
    return p.parse_args()


def _lookup_grid(rows: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(r["d_model"]), int(r["seq_len"])): r for r in rows}


def print_pivot_tables(rows: list[dict]) -> None:
    """Print metrics as grids: rows = head embedding d_model, cols = sequence length."""
    lk = _lookup_grid(rows)
    row_label = "d_model"

    def cell_ok(r: dict, fmt: Callable[[dict], str]) -> str:
        if r.get("status") != "ok":
            return "OOM" if r.get("status") == "OOM" else "err"
        return fmt(r)

    col_w = max(12, max(len(str(s)) for s in SEQ_LENGTHS) + 2)
    row_head_w = max(len(row_label), max(len(str(d)) for d in D_MODELS)) + 2

    def print_table(title: str, fmt: Callable[[dict], str]) -> None:
        print()
        print(title)
        hdr = row_label.ljust(row_head_w) + "".join(str(s).rjust(col_w) for s in SEQ_LENGTHS)
        print(hdr)
        print("-" * len(hdr))
        for d in D_MODELS:
            parts = [str(d).ljust(row_head_w)]
            for s in SEQ_LENGTHS:
                r = lk.get((d, s))
                if r is None:
                    parts.append("---".rjust(col_w))
                else:
                    parts.append(cell_ok(r, fmt).rjust(col_w))
            print("".join(parts))

    print_table(
        "Forward time — mean ms per pass (100 samples)",
        lambda r: f"{r['forward_mean_ms']:.4f}",
    )
    print_table(
        "Backward time — mean ms per pass (100 backward() calls; forward untimed)",
        lambda r: f"{r['backward_mean_ms']:.4f}",
    )
    print_table(
        "Memory before backward — MiB (torch.cuda.memory_allocated after forward)",
        lambda r: f"{r['mem_before_backward_bytes'] / (1024 * 1024):.2f}",
    )

    print()
    print("Std dev (ms) — forward σ")
    hdr = row_label.ljust(row_head_w) + "".join(str(s).rjust(col_w) for s in SEQ_LENGTHS)
    print(hdr)
    print("-" * len(hdr))
    for d in D_MODELS:
        parts = [str(d).ljust(row_head_w)]
        for s in SEQ_LENGTHS:
            r = lk.get((d, s))
            if r is None or r.get("status") != "ok":
                parts.append(("OOM" if r and r.get("status") == "OOM" else "---").rjust(col_w))
            else:
                parts.append(f"{r['forward_std_ms']:.4f}".rjust(col_w))
        print("".join(parts))

    print()
    print("Std dev (ms) — backward σ")
    hdr = row_label.ljust(row_head_w) + "".join(str(s).rjust(col_w) for s in SEQ_LENGTHS)
    print(hdr)
    print("-" * len(hdr))
    for d in D_MODELS:
        parts = [str(d).ljust(row_head_w)]
        for s in SEQ_LENGTHS:
            r = lk.get((d, s))
            if r is None or r.get("status") != "ok":
                parts.append(("OOM" if r and r.get("status") == "OOM" else "---").rjust(col_w))
            else:
                parts.append(f"{r['backward_std_ms']:.4f}".rjust(col_w))
        print("".join(parts))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda")
    # Prime CUDA + autograd/cuBLAS so the first timed backward is not special-cased.
    _x = torch.randn(8, 8, device=device, dtype=dtype, requires_grad=True)
    (_x * _x).sum().backward()
    torch.cuda.synchronize()
    del _x

    model: nn.Module = AttentionModule().to(device)
    if args.compile:
        model = torch.compile(model)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {dtype}")
    print(f"torch.compile: {args.compile}")
    print(f"batch_size={BATCH_SIZE}, heads=1 (implicit), forward_passes={NUM_FORWARD}, backward_passes={NUM_BACKWARD}")
    n_configs = len(D_MODELS) * len(SEQ_LENGTHS)
    print(f"Running {n_configs} configurations…")
    print()

    rows: list[dict] = []
    done = 0
    for d_model in D_MODELS:
        for seq_len in SEQ_LENGTHS:
            torch.cuda.empty_cache()
            try:
                row = benchmark_one(d_model, seq_len, device, dtype, model)
                rows.append(row)
                done += 1
                print(f"\r  [{done}/{n_configs}] done", end="", flush=True)
            except RuntimeError as e:
                done += 1
                print(f"\r  [{done}/{n_configs}] done", end="", flush=True)
                if "out of memory" in str(e).lower():
                    rows.append(
                        {
                            "d_model": d_model,
                            "seq_len": seq_len,
                            "status": "OOM",
                            "error": str(e),
                        }
                    )
                else:
                    traceback.print_exc()
                    rows.append({"d_model": d_model, "seq_len": seq_len, "status": "error", "error": str(e)})
    print()

    print_pivot_tables(rows)


if __name__ == "__main__":
    main()
