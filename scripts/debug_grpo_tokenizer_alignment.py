#!/usr/bin/env python3
"""Compare tokenizer paths used by GRPO training vs defaults used during rollout encoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alignment.eval import DEFAULT_MODEL_NAME, build_prompts
from alignment.prompts import DIRECT_PROMPT_TEMPLATE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--question", type=str, default="If Mary has 3 apples and buys 2, how many apples does she have?")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = build_prompts([args.question], DIRECT_PROMPT_TEMPLATE)
    probe = prompts[0]
    ids_plain = tokenizer.encode(probe, add_special_tokens=False)

    batch_default = tokenizer(
        [probe],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    batch_aligned = tokenizer(
        [probe],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    row_default = batch_default["input_ids"][0].tolist()
    row_aligned = batch_aligned["input_ids"][0].tolist()

    payload = {
        "sessionId": "27fd8b",
        "timestamp": int(time.time() * 1000),
        "hypothesisId": "H1",
        "location": "scripts/debug_grpo_tokenizer_alignment.py:main",
        "message": "tokenizer_plain_vs_batch_modes",
        "runId": "dbg-token-smoke",
        "data": {
            "model": args.model,
            "len_encode_plain": len(ids_plain),
            "len_batch_default_pad": len(row_default),
            "len_batch_aligned_pad": len(row_aligned),
            "plain_equals_row_aligned_unpad_list": row_aligned == ids_plain,
            "plain_equals_row_default_unpad_list": row_default == ids_plain,
        },
    }

    log_path = REPO_ROOT / ".cursor" / "debug-27fd8b.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload) + "\n")

    print(json.dumps(payload["data"], indent=2))


if __name__ == "__main__":
    main()
