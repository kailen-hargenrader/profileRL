from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
import json

from vllm import LLM
from vllm.sampling_params import SamplingParams
from datasets import load_dataset

from . import r1_zero_reward_fn
from . import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE: int = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    ds = load_dataset("gsm8k", "main", split=split)
    examples: list[dict[str, Any]] = []
    for row in ds:
        question = row["question"].strip()
        answer = row["answer"].strip()
        if "####" in answer:
            ground_truth = answer.split("####")[-1].strip()
        else:
            ground_truth = answer
        examples.append(
            {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
            }
        )
    return examples


def build_prompts(questions: Sequence[str], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=question) for question in questions]


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    ground_truths: Sequence[str],
    eval_sampling_params: SamplingParams,
) -> dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    Args:
        vllm_model: The VLLM model to evaluate.
        reward_fn: A function to compute rewards for each generation.
        prompts: A list of prompts to evaluate.
        ground_truths: A list of ground truths to evaluate.
        eval_sampling_params: The sampling parameters to use for evaluation.
    """
    prompts_list = list(prompts)
    gt_list = list(ground_truths)
    assert len(gt_list) == len(prompts_list), "ground_truths length must match prompts length"

    outputs = vllm_model.generate(prompts_list, eval_sampling_params)
    generations = [req.outputs[0].text for req in outputs]
    results = {}
    for i, (gen, gt) in enumerate(zip(generations, gt_list, strict=True)):
        results[i] = {}
        results[i]["question"] = prompts_list[i]
        results[i]["ground_truth"] = gt
        results[i]["response"] = gen
        for key, value in reward_fn(gen, gt).items():
            results[i][key] = value
    return results

def run_baseline(llm: LLM, output_path: Path | None, prompt_template: str, verbose: bool = False, max_evals: int | None = None) -> dict[str, Any]:
    examples = load_gsm8k_examples("test")
    if max_evals is not None:
        examples = examples[:max_evals]
    questions = [ex["question"] for ex in examples]
    prompts = build_prompts(questions, prompt_template)
    ground_truths = [ex["ground_truth"] for ex in examples]
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    results = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        ground_truths,
        sampling_params,
    )

    metrics = {}
    for key, value in results.items():
        for k, v in value.items():
            if isinstance(v, (int, float)):
                metrics["average_" + k] = metrics.get("average_" + k, 0.0) + v
           
    
    metrics = {k: v / len(results) for k, v in metrics.items()}

    if verbose:
        for i, value in results.items():
            print(f"{'='*40}")
            print(f"Prompt {i+1}:\n{value['question']}\n")
            print("-" * 10)
            print(f"Response {i+1}:\n{value['response']}")
   
        print(metrics)
    
    output = {
        "metrics": metrics,
        "results": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    return output


def run_direct_baseline(llm: LLM, output_path: Path | None, verbose: bool = False, max_evals: int | None = None) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    run_baseline(llm, output_path, DIRECT_PROMPT_TEMPLATE, verbose, max_evals)


def run_cot_baseline(llm: LLM, output_path: Path | None, verbose: bool = False, max_evals: int | None = None) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    run_baseline(llm, output_path, COT_PROMPT_TEMPLATE, verbose, max_evals)


def run_self_consistency_baseline(llm: LLM, output_path: Path | None, k: int = 5, verbose: bool = False, max_evals: int | None = None) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    results = run_baseline(llm, None, COT_PROMPT_TEMPLATE, verbose, max_evals)["results"]
    for i in range(k - 1):
        new_results = run_baseline(llm, None, COT_PROMPT_TEMPLATE, verbose, max_evals)["results"]

        #update scores
        for key, value in new_results.items():
            for key2, value2 in value.items():
                if isinstance(value2, (int, float)):
                    results[key][key2] = results[key][key2] + value2
    
    metrics = {}
    for key, value in results.items():
            for key2, value2 in value.items():
                if isinstance(value2, (int, float)):
                    results[key][key2] = float(value2) / k
                    metrics[f"average_{key2}"] = metrics.get(f"average_{key2}", 0.0) + round(results[key][key2], 1)
    
    metrics = {k: v / len(results) for k, v in metrics.items()}

    output = {
        "metrics": metrics,
        "results": results,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    return output


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


if __name__ == "__main__":
    llm = LLM(model=DEFAULT_MODEL_NAME)
    run_self_consistency_baseline(llm, Path("results/self_consistency_baseline.json"), k=5, verbose=False, max_evals=None)