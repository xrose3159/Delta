#!/usr/bin/env python3
"""
Offline High-Entropy Sum (HES) scoring for existing Chain-of-Thought (CoT) completions.

This script computes the HES metric per sample by:
  1) Concatenating prompt + completion as a single input string.
  2) Running vLLM in teacher-forced mode using the "max_tokens=0" trick with
     prompt_logprobs enabled to obtain token-level log-prob distributions over the
     input text (no new tokens are generated).
  3) Slicing log-prob distributions to the completion span only.
  4) Computing token entropies for the completion tokens and summing those in the
     top p percentile (highest entropy), yielding the HES score.

Usage example:
  python hes_scorer.py \
      --model_path meta-llama/Llama-2-7b-hf \
      --dataset_path /path/to/dataset.jsonl \
      --output_path /path/to/scored_dataset.jsonl \
      --percentile_cutoff 0.005 \
      --tensor_parallel_size 1

Notes:
  - HES depends on high-quality token-level log-prob distributions. We request a
    very large "logprobs" count (e.g., 32000) to approximate a near-full
    distribution when feasible.
  - For long inputs, this can be memory intensive. Consider smaller batches if
    you run into memory constraints.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except Exception as e:  # pragma: no cover - import-time configuration errors
    raise RuntimeError(
        "Failed to import vLLM. Please install vllm (pip install vllm) and ensure CUDA is available if using GPU."
    ) from e


class HESScorer:
    """Compute High-Entropy Sum (HES) scores for CoT completions using vLLM.

    The HES metric is defined as the sum of token entropies for the top p percentile
    of the most uncertain (highest-entropy) tokens within the completion span.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        percentile_cutoff: float = 0.005,
        max_model_len: int = None,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        """
        Initialize the HES scorer with a vLLM engine.
        
        Args:
            model_path: Path to the Hugging Face model directory
            tensor_parallel_size: Number of GPUs for tensor parallelism
            percentile_cutoff: Percentile p for selecting high-entropy tokens (e.g., 0.005 = top 0.5%)
            max_model_len: Maximum model sequence length (None = auto-detect)
            gpu_memory_utilization: GPU memory utilization fraction
        """
        print(f"Initializing vLLM engine with model: {model_path}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        
        # Initialize vLLM engine with configurable settings
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        # Keep a reference to the tokenizer used by vLLM to ensure identical tokenization.
        self.tokenizer = self.llm.get_tokenizer()

        if not (0.0 < percentile_cutoff < 1.0):
            raise ValueError("percentile_cutoff must be in (0, 1). Example: 0.005 for top 0.5%.")
        self.percentile_cutoff = percentile_cutoff
        print(f"HES percentile cutoff: {percentile_cutoff} (top {percentile_cutoff * 100}%)")

        # Request logprobs per position to approximate token distribution.
        # We enable prompt_logprobs to obtain distributions over the input tokens.
        # Note: vLLM v1 limits max logprobs to 20, so we use that maximum.
        # Note: vLLM v1 requires max_tokens >= 1, so we set it to 1 and ignore generated tokens.
        self.sampling_params = SamplingParams(
            max_tokens=1,  # minimum allowed by vLLM v1; we ignore generated tokens
            logprobs=20,   # max allowed by vLLM v1
            prompt_logprobs=20,  # max allowed by vLLM v1
        )
        print("Initialization complete!\n")

    @torch.no_grad()
    def calculate_hes_for_batch(
        self,
        batch_prompts: Sequence[str],
        batch_completions: Sequence[str],
    ) -> List[float]:
        """Compute HES scores for a batch.

        Steps:
          - Concatenate prompt + completion to form the full input per sample.
          - Run vLLM.generate with max_tokens=0 and prompt_logprobs, capturing token-level
            log-prob dictionaries for the input tokens.
          - Slice the prompt_logprobs to the completion span using tokenizer-aligned indices.
          - Compute per-token entropy and aggregate the top p percentile.

        Returns a list of HES scores, one per sample.
        """
        if len(batch_prompts) != len(batch_completions):
            raise ValueError("batch_prompts and batch_completions must have the same length")

        # Prepare full inputs. No additional separator is added by default.
        full_texts: List[str] = [p + c for p, c in zip(batch_prompts, batch_completions)]

        # Tokenize prompts alone to locate the completion span start for each input.
        prompt_token_lengths: List[int] = [
            len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in batch_prompts
        ]

        # Request prompt logprobs over the provided full texts.
        outputs = self.llm.generate(full_texts, self.sampling_params, use_tqdm=False)

        hes_scores: List[float] = []
        for i, out in enumerate(outputs):
            # vLLM returns per-position logprob distributions for the input tokens when prompt_logprobs=True.
            # out.prompt_logprobs: List[Optional[Dict[str, TokenLogprob]]]
            prompt_logprobs_list = getattr(out, "prompt_logprobs", None)
            if prompt_logprobs_list is None:
                # Defensive fallback: if not present, HES cannot be computed.
                hes_scores.append(0.0)
                continue

            # Determine the starting index of completion tokens within the full input tokenization.
            completion_start = prompt_token_lengths[i]

            # Safety: If completion is empty, score is 0.
            if len(batch_completions[i]) == 0:
                hes_scores.append(0.0)
                continue

            # Iterate over completion token positions only.
            token_entropies: List[float] = []
            for pos in range(completion_start, len(prompt_logprobs_list)):
                token_dist = prompt_logprobs_list[pos]
                if not token_dist:
                    # Some positions may not contain a distribution (e.g., first input token).
                    continue

                # token_dist maps token string -> TokenLogprob (with .logprob attribute)
                # Collect log-probabilities; numerical stability is handled by softmax on log-values.
                logprob_values: List[float] = []
                for v in token_dist.values():
                    # v.logprob is expected to be a float (natural log-prob)
                    logprob_values.append(float(v.logprob))

                if len(logprob_values) == 0:
                    continue

                logprob_tensor = torch.tensor(logprob_values, dtype=torch.float32)
                probs_tensor = F.softmax(logprob_tensor, dim=-1)
                entropy = torch.distributions.Categorical(probs=probs_tensor).entropy().item()
                token_entropies.append(entropy)

            # Aggregate top p percentile entropies
            if len(token_entropies) == 0:
                hes_scores.append(0.0)
                continue

            entropies_np = np.asarray(token_entropies, dtype=np.float32)

            # Compute percentile threshold for top p fraction
            # Example: p=0.005 -> threshold at 99.5th percentile
            threshold_percent = (1.0 - self.percentile_cutoff) * 100.0
            threshold = float(np.percentile(entropies_np, threshold_percent))

            # Select entropies >= threshold. Ensure at least one token contributes.
            selected = entropies_np[entropies_np >= threshold]
            if selected.size == 0:
                # Fallback to selecting the single maximum entropy token.
                selected = entropies_np[np.argmax(entropies_np)][None]

            hes_score = float(np.sum(selected, dtype=np.float64))
            hes_scores.append(hes_score)

        return hes_scores


def stream_jsonl(path: str) -> Tuple[int, List[Dict]]:
    """Count lines and return a buffered list of dicts lazily if needed.

    This helper first counts lines to drive a tqdm progress bar. It does not load the
    entire dataset into memory during scoring (we still process in batches).
    """
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    return total, []


def read_jsonl_in_batches(
    path: str,
    batch_size: int,
    prompt_key: str,
    completion_key: str,
):
    """Yield batches of (records, prompts, completions) from a JSONL file.

    - records: List[original JSON objects]
    - prompts: List[str]
    - completions: List[str]
    """
    buffer_records: List[Dict] = []
    buffer_prompts: List[str] = []
    buffer_completions: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get(prompt_key, "")
            completion = obj.get(completion_key, "")

            buffer_records.append(obj)
            buffer_prompts.append(prompt)
            buffer_completions.append(completion)

            if len(buffer_records) >= batch_size:
                yield buffer_records, buffer_prompts, buffer_completions
                buffer_records, buffer_prompts, buffer_completions = [], [], []

    if buffer_records:
        yield buffer_records, buffer_prompts, buffer_completions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute High-Entropy Sum (HES) scores for CoT completions using vLLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path or name of the Hugging Face model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to input JSONL dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save JSONL with added 'hes_score'.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="JSON key for the prompt (default: 'prompt').")
    parser.add_argument(
        "--completion_key", type=str, default="completion", help="JSON key for the completion (default: 'completion')."
    )
    parser.add_argument(
        "--percentile_cutoff",
        type=float,
        default=0.005,
        help="Top p fraction for entropy aggregation (e.g., 0.005 => top 0.5%).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism (default: 1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for scoring (default: 8). Reduce if you hit OOM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model sequence length (None = auto-detect).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (default: 0.9).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Print configuration
    print("=" * 80)
    print("HES Score Calculator - Configuration")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Input dataset: {args.dataset_path}")
    print(f"Output dataset: {args.output_path}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Percentile cutoff: {args.percentile_cutoff}")
    print("=" * 80)
    print()

    scorer = HESScorer(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        percentile_cutoff=args.percentile_cutoff,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    total_lines, _ = stream_jsonl(args.dataset_path)
    
    # 逐个样本处理，避免单个样本失败导致整个任务崩溃
    failed_count = 0
    success_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fout:
        with tqdm(total=total_lines, desc="Scoring (HES)") as pbar:
            for records, prompts, completions in read_jsonl_in_batches(
                args.dataset_path, args.batch_size, args.prompt_key, args.completion_key
            ):
                # 逐个样本处理
                for obj, prompt, completion in zip(records, prompts, completions):
                    try:
                        # 截断过长的输入（避免 OOM）
                        max_chars = 32000  # 约 8000 tokens
                        if len(prompt) + len(completion) > max_chars:
                            # 保留 prompt，截断 completion
                            remaining = max_chars - len(prompt)
                            if remaining > 0:
                                completion = completion[:remaining]
                            else:
                                prompt = prompt[:max_chars // 2]
                                completion = completion[:max_chars // 2]
                        
                        hes_scores = scorer.calculate_hes_for_batch([prompt], [completion])
                        hes_score = hes_scores[0] if hes_scores else 0.0
                        success_count += 1
                    except Exception as e:
                        print(f"Warning: HES calculation failed for sample {obj.get('id', 'unknown')}: {e}")
                        hes_score = 0.0
                        failed_count += 1
                    
                    obj["hes_score"] = float(hes_score)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    fout.flush()  # 立即写入，避免崩溃丢失数据
                
                pbar.update(len(records))
    
    print(f"HES scoring completed: {success_count} success, {failed_count} failed")


if __name__ == "__main__":
    main()


