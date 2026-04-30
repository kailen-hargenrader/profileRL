from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _get_response_log_probs_batched(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    forward_microbatch_size: int,
) -> Tensor:
    """Score log-probs in chunks along batch to cap activation memory; drop chunk outputs between steps."""
    b = input_ids.shape[0]
    if b == 0:
        raise ValueError("empty batch")
    forward_microbatch_size = max(1, forward_microbatch_size)
    chunks: list[Tensor] = []
    for start in range(0, b, forward_microbatch_size):
        end = min(start + forward_microbatch_size, b)
        sl_ids = input_ids[start:end]
        sl_lab = labels[start:end]
        out = get_response_log_probs(model, sl_ids, sl_lab, return_token_entropy=False)
        chunks.append(out["log_probs"])
        del out
    return torch.cat(chunks, dim=0)

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs for causal LM: input_ids drop the final token, labels are next-token targets, response_mask marks label positions over the generated answer."""

    pad_id = tokenizer.pad_token_id
    full_sequences = [
        tokenizer.encode(prompt, add_special_tokens=False)
        + tokenizer.encode(output, add_special_tokens=False)
        for prompt, output in zip(prompt_strs, output_strs, strict=True)
    ]
    max_lm_len = max(len(seq) - 1 for seq in full_sequences)

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for prompt, output, full in zip(prompt_strs, output_strs, full_sequences, strict=True):
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        response_len = len(tokenizer.encode(output, add_special_tokens=False))
        lm_len = len(full) - 1
        pad_len = max_lm_len - lm_len
        input_rows.append(full[:-1] + [pad_id] * pad_len)
        label_rows.append(full[1:] + [pad_id] * pad_len)
        mask_rows.append(
            [False] * (prompt_len - 1) + [True] * response_len + [False] * pad_len
        )

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "response_mask": torch.tensor(mask_rows, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    out = model(input_ids)
    logits = out.logits
    # Fused NLL: avoids materializing a full [B, T, V] log_softmax (large VRAM at long T and big V).
    b, t, v = logits.shape
    log_probs = -F.cross_entropy(
        logits.reshape(-1, v),
        labels.reshape(-1),
        reduction="none",
    ).reshape(b, t)

    result: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result



def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    weighted = tensor * mask
    if dim is None:
        return weighted.sum() / normalize_constant
    return weighted.sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("len(rollout_responses) must be divisible by group_size")

    raw_rewards = torch.tensor(
        [
            reward_fn(resp, gt)["reward"]
            for resp, gt in zip(rollout_responses, repeated_ground_truths, strict=True)
        ],
        dtype=torch.float32,
    )
    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        normalized = centered / (grouped.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    else:
        normalized = centered
    advantages = normalized.reshape(-1)

    metadata: dict[str, float] = {
        "mean_reward": float(raw_rewards.mean().item()),
    }
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss.

    Reference ``old_log_probs`` must be detached (typically from ``torch.no_grad()`` scoring);
    they are detached again here so gradients never flow through the behavior policy.
    """
    old_log_probs = old_log_probs.detach()
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    adv = advantages.unsqueeze(-1) if advantages.dim() == 1 else advantages
    broadcast_advantages = adv.expand_as(policy_log_probs)
    loss = -torch.minimum(ratios * broadcast_advantages, clipped_ratios * broadcast_advantages)
    metadata = {
        "ratios": ratios,
        "clipped_ratios": clipped_ratios,
    }
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
    *,
    ddp_model: DDP | None = None,
    sync_gradients: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss.

    When using DDP with gradient accumulation, pass ``ddp_model`` and set ``sync_gradients=False``
    on non-final microbatches to avoid all-reducing every backward.
    """
    per_token_loss, clip_metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask = response_mask.to(dtype=per_token_loss.dtype)
    masked_loss = per_token_loss * mask
    per_example_loss = masked_loss.sum(dim=1) / mask.sum(dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    if ddp_model is not None and not sync_gradients:
        with ddp_model.no_sync():
            loss.backward()
    else:
        loss.backward()

    metadata: dict[str, Tensor] = dict(clip_metadata)
    metadata["per_token_loss"] = per_token_loss
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
    use_wandb: bool = False,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    if use_wandb:
        wandb.log({
            "prompts": prompts,
            "responses": responses,
            "ground_truths": ground_truths,
            "reward_infos": reward_infos,
            "token_entropies": token_entropies,
        })
    return []


def _generate_rollout_responses(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts_flat: list[str],
    device: torch.device,
    generation_microbatch_size: int,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    sampling_temperature: float,
    sampling_top_p: float,
) -> list[str]:
    """Sample completions for a flat list of prompts (already expanded by group_size per prompt)."""
    gen_model = _unwrap_model(model)
    was_training = model.training
    gen_model.eval()
    responses: list[str] = []
    pad_id = tokenizer.pad_token_id
    generation_microbatch_size = max(1, generation_microbatch_size)
    tokenizer.padding_side = "left"
    with torch.no_grad():
        for start in range(0, len(prompts_flat), generation_microbatch_size):
            chunk = prompts_flat[start : start + generation_microbatch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            )
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            prefix_len = enc["input_ids"].shape[1]
            generated = gen_model.generate(
                **enc,
                max_new_tokens=sampling_max_tokens,
                min_new_tokens=sampling_min_tokens,
                do_sample=True,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                pad_token_id=pad_id,
            )
            for seq in generated:
                new_tokens = seq[prefix_len:]
                responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
            del enc, generated
    if was_training:
        model.train()
    return responses


def _validation_mean_reward(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    val_prompts: Sequence[str],
    val_ground_truths: Sequence[str],
    device: torch.device,
    generation_microbatch_size: int,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    sampling_temperature: float,
    sampling_top_p: float,
) -> float:
    """Single-sample mean reward on validation prompts (caller runs on rank 0 only)."""
    prompts_list = list(val_prompts)
    gts = list(val_ground_truths)
    if len(prompts_list) == 0:
        return 0.0
    texts = _generate_rollout_responses(
        model=model,
        tokenizer=tokenizer,
        prompts_flat=prompts_list,
        device=device,
        generation_microbatch_size=generation_microbatch_size,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
    )
    total = 0.0
    for resp, gt in zip(texts, gts, strict=True):
        total += float(reward_fn(resp, gt)["reward"])
    return total / len(texts)


def train_grpo(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    optimizer: torch.optim.Optimizer,
    *,
    n_grpo_steps: int = 8,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    device: torch.device | str | None = None,
    generation_microbatch_size: int = 8,
    forward_microbatch_size: int = 8,
    use_wandb: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_every_steps: int | None = None,
    eval_every_steps: int | None = None,
    validation_prompts: Sequence[str] | None = None,
    validation_ground_truths: Sequence[str] | None = None,
    max_validation_examples: int | None = None,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5.

    Distributed (torchrun): ``rollout_batch_size`` and ``train_batch_size`` are **global** counts
    of completion sequences. Each rank handles ``* // world_size``, which must divide evenly.
    Each rank keeps full GRPO groups (``group_size``) locally. Checkpoints and validation logging
    should run only on rank 0 (handled here).

    Reference policy log-probs are computed under ``torch.no_grad()`` and never receive gradients.
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddp_model: DDP | None = model if isinstance(model, DDP) else None

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rollout_batch_size % group_size != 0:
        raise ValueError(f"rollout_batch_size ({rollout_batch_size}) must be divisible by group_size ({group_size})")
    if rollout_batch_size % world_size != 0:
        raise ValueError(
            f"rollout_batch_size ({rollout_batch_size}) must be divisible by WORLD_SIZE ({world_size})"
        )
    if train_batch_size % world_size != 0:
        raise ValueError(
            f"train_batch_size ({train_batch_size}) must be divisible by WORLD_SIZE ({world_size})"
        )
    local_rollout = rollout_batch_size // world_size
    local_train = train_batch_size // world_size
    if local_rollout % group_size != 0:
        raise ValueError(
            f"local rollout ({local_rollout}) must be divisible by group_size ({group_size})"
        )
    if train_batch_size != rollout_batch_size:
        raise ValueError("train_batch_size must equal rollout_batch_size for on-policy GRPO in this implementation")
    if local_train % gradient_accumulation_steps != 0:
        raise ValueError(
            f"local_train ({local_train}) must be divisible by gradient_accumulation_steps ({gradient_accumulation_steps})"
        )

    micro_train_batch_size = local_train // gradient_accumulation_steps
    n_prompts_global = rollout_batch_size // group_size
    n_prompts_local = local_rollout // group_size

    prompts_list = list(prompts)
    gts_list = list(ground_truths)
    if len(prompts_list) != len(gts_list):
        raise ValueError("prompts and ground_truths must have the same length")
    n_examples = len(prompts_list)
    if n_examples == 0:
        raise ValueError("no training examples")

    last_mean_reward = 0.0

    for step in range(n_grpo_steps):
        global_base = step * n_prompts_global
        indices = [
            (global_base + rank * n_prompts_local + i) % n_examples for i in range(n_prompts_local)
        ]

        batch_prompts = [prompts_list[i] for i in indices]
        batch_gts = [gts_list[i] for i in indices]

        prompts_flat: list[str] = []
        gts_flat: list[str] = []
        for p, gt in zip(batch_prompts, batch_gts, strict=True):
            prompts_flat.extend([p] * group_size)
            gts_flat.extend([gt] * group_size)

        rollout_responses = _generate_rollout_responses(
            model=model,
            tokenizer=tokenizer,
            prompts_flat=prompts_flat,
            device=dev,
            generation_microbatch_size=generation_microbatch_size,
            sampling_min_tokens=sampling_min_tokens,
            sampling_max_tokens=sampling_max_tokens,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
        )
        del prompts_flat

        advantages, raw_rewards, rew_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=gts_flat,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )
        last_mean_reward = rew_meta["mean_reward"]

        reward_infos = [reward_fn(r, gt) for r, gt in zip(rollout_responses, gts_flat, strict=True)]
        if use_wandb and rank == 0:
            log_generations(
                prompts=batch_prompts,
                responses=rollout_responses,
                ground_truths=gts_flat,
                reward_infos=reward_infos,
                use_wandb=True,
            )
        del reward_infos, raw_rewards

        prompt_strs_rep: list[str] = []
        for p in batch_prompts:
            prompt_strs_rep.extend([p] * group_size)
        tok = tokenize_prompt_and_output(
            prompt_strs=prompt_strs_rep,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        del rollout_responses, gts_flat

        input_ids = tok["input_ids"].to(dev, non_blocking=True)
        labels = tok["labels"].to(dev, non_blocking=True)
        response_mask = tok["response_mask"].to(dev, non_blocking=True)
        del tok

        advantages_dev = advantages.to(dev, non_blocking=True)
        del advantages

        with torch.no_grad():
            old_log_probs = _get_response_log_probs_batched(
                model=model,
                input_ids=input_ids,
                labels=labels,
                forward_microbatch_size=forward_microbatch_size,
            )
        old_log_probs = old_log_probs.detach()

        num_microbatches = local_rollout // micro_train_batch_size
        if num_microbatches != gradient_accumulation_steps:
            raise RuntimeError("internal: microbatch count mismatch")

        for _epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad(set_to_none=True)
            for mb in range(num_microbatches):
                sl = slice(mb * micro_train_batch_size, (mb + 1) * micro_train_batch_size)
                ids_sl = input_ids[sl]
                lab_sl = labels[sl]
                mask_sl = response_mask[sl]
                adv_sl = advantages_dev[sl]
                old_sl = old_log_probs[sl]

                policy_log_probs = _get_response_log_probs_batched(
                    model=model,
                    input_ids=ids_sl,
                    labels=lab_sl,
                    forward_microbatch_size=forward_microbatch_size,
                )

                sync_gradients = mb == num_microbatches - 1
                loss_mb, _meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mask_sl,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=adv_sl,
                    old_log_probs=old_sl,
                    cliprange=cliprange,
                    ddp_model=ddp_model,
                    sync_gradients=sync_gradients,
                )
                del policy_log_probs, loss_mb, _meta, ids_sl, lab_sl, mask_sl, adv_sl, old_sl

            optimizer.step()

        del old_log_probs, input_ids, labels, response_mask, advantages_dev

        if dist.is_initialized():
            dist.barrier()

        metrics_tensor = torch.tensor([last_mean_reward], device=dev, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= world_size
        agg_mean_reward = float(metrics_tensor.item())
        del metrics_tensor

        if use_wandb and rank == 0:
            wandb.log({"train/mean_reward": agg_mean_reward, "train/step": step})

        if checkpoint_dir is not None and checkpoint_every_steps is not None:
            if rank == 0 and (step + 1) % checkpoint_every_steps == 0:
                ckpt_root = Path(checkpoint_dir)
                ckpt_root.mkdir(parents=True, exist_ok=True)
                save_dir = ckpt_root / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                gen_ref = _unwrap_model(model)
                gen_ref.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

        if (
            eval_every_steps is not None
            and validation_prompts is not None
            and validation_ground_truths is not None
            and (step + 1) % eval_every_steps == 0
        ):
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                vp = list(validation_prompts)
                vg = list(validation_ground_truths)
                if max_validation_examples is not None:
                    vp = vp[:max_validation_examples]
                    vg = vg[:max_validation_examples]
                val_reward = _validation_mean_reward(
                    model=model,
                    tokenizer=tokenizer,
                    reward_fn=reward_fn,
                    val_prompts=vp,
                    val_ground_truths=vg,
                    device=dev,
                    generation_microbatch_size=generation_microbatch_size,
                    sampling_min_tokens=sampling_min_tokens,
                    sampling_max_tokens=sampling_max_tokens,
                    sampling_temperature=sampling_temperature,
                    sampling_top_p=sampling_top_p,
                )
                if use_wandb:
                    wandb.log({"val/mean_reward": val_reward, "train/step": step})
            if dist.is_initialized():
                dist.barrier()

    return {"mean_reward": last_mean_reward, "n_grpo_steps": n_grpo_steps, "n_prompts_global": n_prompts_global}

