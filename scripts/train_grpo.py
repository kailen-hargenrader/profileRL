#!/usr/bin/env python3
"""
Fine-tune a causal LM with GRPO on GSM8K-style prompts using alignment.grpo.train_grpo.

Expects outputs with <answer>...</answer> (see alignment.prompts.DIRECT_PROMPT_TEMPLATE).
Reward uses alignment.rewards.answer_tag_reward_fn.

OOM / memory:
  Peak VRAM scales with (prompt_batch_size / num_gpus) * group_size when using DDP —
  still linear in group_size. Long CoT + large max_new_tokens widens padded sequences.
  Mitigations: torchrun + DDP (this script), --gradient-checkpointing,
  --generation-microbatch-size, --forward-microbatch-size, lower group_size / prompt_batch_size / max_new_tokens.
  Validation: --eval-every-steps with --eval-split (default test) and --max-eval-examples logs val/mean_reward (rank 0).

Multi-GPU: launch with torchrun, e.g.
  torchrun --standalone --nproc_per_node=2 scripts/train_grpo.py ...
  Global rollout batch size must be divisible by WORLD_SIZE and by group_size; local rollouts must also divide by group_size.

Effective batch size scales with WORLD_SIZE; consider scaling learning rate linearly with WORLD_SIZE.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alignment.eval import DEFAULT_MODEL_NAME, build_prompts, get_prompt_template, load_gsm8k_examples
from alignment.grpo import train_grpo
from alignment.rewards import answer_tag_reward_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO fine-tuning on GSM8K with optional DDP.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--max-examples", type=int, default=None, help="Cap training examples after loading split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cot", action="store_true", help="Use chain-of-thought prompt template.")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument(
        "--rollout-batch-size",
        type=int,
        default=32,
        help="Global number of completion sequences per GRPO step (divisible by WORLD_SIZE and group_size).",
    )
    p.add_argument("--num-steps", type=int, default=60, dest="n_grpo_steps")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-new-tokens", type=int, default=256, dest="sampling_max_tokens")
    p.add_argument("--min-new-tokens", type=int, default=4, dest="sampling_min_tokens")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--cliprange", type=float, default=1.0)
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    p.add_argument("--no-normalize-by-std", action="store_true")
    p.add_argument("--generation-microbatch-size", type=int, default=8)
    p.add_argument("--forward-microbatch-size", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-entity", type=str, default="khargenr-california-institute-of-technology-caltech")
    p.add_argument("--wandb-project", type=str, default="eecs-148b-hw2-grpo")
    p.add_argument("--checkpoint-dir", type=Path, default="checkpoints/withNorm")
    p.add_argument("--checkpoint-every-steps", type=int, default=5)
    p.add_argument("--eval-every-steps", type=int, default=5)
    p.add_argument("--eval-split", type=str, default="test")
    p.add_argument("--max-eval-examples", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        if distributed:
            raise RuntimeError("Multi-GPU training requires CUDA.")

    if distributed:
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0

    set_seed(args.seed + rank)

    template = get_prompt_template(use_cot=args.cot)
    examples = load_gsm8k_examples(args.train_split)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    questions = [ex["question"] for ex in examples]
    train_prompts = build_prompts(questions, template)
    train_gts = [ex["ground_truth"] for ex in examples]

    val_prompts_list: list[str] | None = None
    val_gts_list: list[str] | None = None
    if args.eval_every_steps is not None:
        val_examples = load_gsm8k_examples(args.eval_split)[: args.max_eval_examples]
        val_prompts_list = build_prompts([ex["question"] for ex in val_examples], template)
        val_gts_list = [ex["ground_truth"] for ex in val_examples]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only LMs must use left padding when batching variable-length prompts for generate().
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(use_reentrant=False)

    train_batch_size = args.rollout_batch_size
    if train_batch_size < args.group_size:
        raise ValueError("rollout_batch_size must be >= group_size")
    if train_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if train_batch_size % world_size != 0:
        raise ValueError("rollout_batch_size must be divisible by WORLD_SIZE")

    local_rollout = train_batch_size // world_size
    if local_rollout % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "(rollout_batch_size / WORLD_SIZE) must be divisible by gradient_accumulation_steps; "
            f"got local_rollout={local_rollout}, gradient_accumulation_steps={args.gradient_accumulation_steps}"
        )

    model.train()
    policy = model
    if distributed:
        policy = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    if args.wandb and rank == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )

    try:
        train_grpo(
            policy,
            tokenizer,
            answer_tag_reward_fn,
            train_prompts,
            train_gts,
            optimizer,
            n_grpo_steps=args.n_grpo_steps,
            advantage_eps=args.advantage_eps,
            rollout_batch_size=args.rollout_batch_size,
            group_size=args.group_size,
            sampling_temperature=args.temperature,
            sampling_top_p=args.top_p,
            sampling_min_tokens=args.sampling_min_tokens,
            sampling_max_tokens=args.sampling_max_tokens,
            epochs_per_rollout_batch=args.epochs_per_rollout_batch,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cliprange=args.cliprange,
            normalize_by_std=not args.no_normalize_by_std,
            device=device,
            generation_microbatch_size=args.generation_microbatch_size,
            forward_microbatch_size=args.forward_microbatch_size,
            use_wandb=args.wandb,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            eval_every_steps=args.eval_every_steps,
            validation_prompts=val_prompts_list,
            validation_ground_truths=val_gts_list,
            max_validation_examples=args.max_eval_examples,
        )

        if args.checkpoint_dir is not None and rank == 0:
            final_dir = Path(args.checkpoint_dir) / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
    finally:
        if args.wandb and rank == 0:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
