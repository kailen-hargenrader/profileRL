"""Ensure batched rollout encoding matches tokenize_prompt_and_output-style encode."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer


def _unpad_row(input_ids: torch.Tensor, attention_mask: torch.Tensor, row: int) -> list[int]:
    mask = attention_mask[row].bool()
    return input_ids[row][mask].tolist()


@pytest.mark.parametrize(
    "model_name,prompts",
    [
        (
            "gpt2",
            [
                "Please answer with ONLY the answer enclosed in <answer> </answer> tags.\nQuestion: hi?\n",
                "Please answer with ONLY the answer enclosed in <answer> </answer> tags.\nQuestion: hello world?\n",
            ],
        ),
    ],
)
def test_batched_tokenizer_add_special_tokens_false_matches_encode_rows(model_name: str, prompts: list[str]) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    for i, prompt in enumerate(prompts):
        expected = tokenizer.encode(prompt, add_special_tokens=False)
        actual = _unpad_row(encoded["input_ids"], encoded["attention_mask"], i)
        assert actual == expected, (model_name, i, len(actual), len(expected))

