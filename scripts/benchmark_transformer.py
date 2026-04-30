#!/usr/bin/env python3
"""
Benchmarking script for the BasicsTransformerLM model.

Times forward and backward passes separately using torch.cuda.Event for accurate GPU timing.

Use `--compile` to run torch.compile(model) on the LM before warmup and timing.

With `--compile`, use `--compile-burnin-steps` (default 50 per phase) to run untimed
forwards (and forward+backward if `--backward`) so Dynamo/Inductor finish before
warmup and measured steps.
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "basics"))

from basics.model import BasicsTransformerLM
from basics.nn_utils import softmax
from jaxtyping import Float, Bool
from torch import Tensor
import math
from einops import einsum
from torch.optim import AdamW

import torch.cuda.nvtx as nvtx
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:

    """Scaled dot-product attention with NVTX ranges.
    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    
    with nvtx.range("applying mask"):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

import basics.model
basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM model")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=128, help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta value")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Benchmark parameters
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of benchmark steps")
    parser.add_argument("--backward", action="store_true", help="Time backward passes as well")
    parser.add_argument("--autocast", action="store_true", help="Use autocast")
    parser.add_argument("--model_size", choices=["small", "medium", "large"], default=None, help="Model size")
    parser.add_argument("--mem_profile", action="store_true", help="Use memory profiler")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile before warmup and benchmarking.",
    )
    parser.add_argument(
        "--compile-burnin-steps",
        type=int,
        default=-1,
        help="Untimed iterations so torch.compile can finish before warmup/timing. "
        "-1 means auto: 50 per phase when --compile (forward-only then forward+backward if --backward), else 0.",
    )

    return parser.parse_args()


def resolve_compile_burnin_steps(args: argparse.Namespace) -> int:
    if not args.compile:
        return 0
    if args.compile_burnin_steps >= 0:
        return args.compile_burnin_steps
    return 50


def compile_burnin(
    model: nn.Module,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_fn: nn.Module,
    n: int,
) -> None:
    """Untimed steps so torch.compile builds forward (and backward) graphs before warmup."""
    if n <= 0:
        return

    print(f"\nCompile burn-in: {n} untimed forward-only steps (torch.no_grad)…")
    model.train()
    with torch.no_grad():
        for _ in range(n):
            if args.autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
            torch.cuda.synchronize()

    if args.backward:
        print(f"Compile burn-in: {n} untimed forward + backward steps…")
        for _ in range(n):
            model.zero_grad()
            if args.autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
            else:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
            loss.backward()
            torch.cuda.synchronize()

    print("Compile burn-in complete.")

def get_model_spec(model_size):
    if model_size == "small":
        return dict(d_model=512, d_ff=2048, num_layers=8, num_heads=8)
    elif model_size == "medium":
        return dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12)
    elif model_size == "large":
        return dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16)
    else:
        raise ValueError(f"Invalid model size: {model_size}")

def print_stats(name, times_ms):
    """Print timing statistics in table format."""
    if not times_ms:
        return None
    
    mean = statistics.mean(times_ms)
    stdev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    min_time = min(times_ms)
    max_time = max(times_ms)
    
    return {
        "name": name,
        "mean": mean,
        "stdev": stdev,
        "min": min_time,
        "max": max_time,
    }


def main():
    args = parse_args()
    if args.model_size is not None:
        model_spec = get_model_spec(args.model_size)
        args.d_model = model_spec["d_model"]
        args.num_layers = model_spec["num_layers"]
        args.num_heads = model_spec["num_heads"]
        args.d_ff = model_spec["d_ff"]

    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")
    
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Validate model config
    if args.d_model % args.num_heads != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads})")
    
    # Initialize model
    print("\nInitializing model...")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model = model.to(device)
    model.train()

    num_params_m = model.get_num_params() / 1e6

    if args.compile:
        print("Compiling model with torch.compile…")
        model = torch.compile(model)

    compile_burnin_n = resolve_compile_burnin_steps(args)
    print(f"torch.compile: {args.compile}")
    if args.compile:
        print(f"compile burn-in steps per phase: {compile_burnin_n}")
    print(f"Model parameters: {num_params_m:.2f}M (non-embedding)")
    
    # Generate random batch
    print("\nGenerating random batch...")
    input_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    target_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")

    compile_burnin(model, args, input_ids, target_ids, loss_fn, compile_burnin_n)

    # Warm-up phase
    print(f"\nWarming up with {args.warmup_steps} steps...")
    with torch.no_grad():
        for _ in range(args.warmup_steps):
            if args.autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids)
            else:
                _ = model(input_ids)
            torch.cuda.synchronize()
    
    print("Warm-up complete.")
    
    # Benchmark forward pass
    print(f"\nBenchmarking forward pass ({args.num_steps} steps)...")
    forward_times = []
    
    for _ in range(args.num_steps):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        if args.mem_profile:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        with nvtx.range("forward pass"):
            if args.autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
            else:
                logits = model(input_ids)
        end_event.record()
        if args.mem_profile:
            torch.cuda.memory._dump_snapshot("profiling/memory_snapshot_forward.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        forward_times.append(elapsed)
    
    forward_stats = print_stats("Forward Pass", forward_times)
    
    # Benchmark backward pass
    backward_stats = None
    if args.backward:
        print(f"\nBenchmarking backward pass ({args.num_steps} steps)...")
        backward_times = []
        
        for _ in range(args.num_steps):
            model.zero_grad()
            
            # Forward pass (not timed as part of backward benchmark)
            if args.autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
            else:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
       
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            if args.mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            start_event.record()
            with nvtx.range("backward pass"):
                loss.backward()
            end_event.record()
            if args.mem_profile:
                torch.cuda.memory._dump_snapshot("profiling/memory_snapshot_backward.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)

            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            backward_times.append(elapsed)
        
        backward_stats = print_stats("Backward Pass", backward_times)
    
    # Print results table
    print("\n" + "="*70)
    print(f"{'Metric':<25} {'Mean ± Std (ms)':<20} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("="*70)
    
    if forward_stats:
        mean_std = f"{forward_stats['mean']:.4f} ± {forward_stats['stdev']:.4f}"
        print(f"{'Forward Pass':<25} {mean_std:<20} {forward_stats['min']:<10.4f} {forward_stats['max']:<10.4f}")
    
    if backward_stats:
        mean_std = f"{backward_stats['mean']:.4f} ± {backward_stats['stdev']:.4f}"
        print(f"{'Backward Pass':<25} {mean_std:<20} {backward_stats['min']:<10.4f} {backward_stats['max']:<10.4f}")
    
    print("="*70)

    if args.backward:
        print("\nRunning one full training step (forward, backward, optimizer)...")
        optimizer = AdamW(model.parameters(), lr=1e-3)
        if args.mem_profile:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        with nvtx.range("full training step"):
            model.zero_grad()
            if args.autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
            else:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, args.vocab_size), target_ids.view(-1))
            with nvtx.range("full training step - backward pass"):
                loss.backward()
            optimizer.step()
        if args.mem_profile:
            torch.cuda.memory._dump_snapshot("profiling/memory_snapshot_full.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

        print("Full training step complete.")
        print("="*70)
    
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
