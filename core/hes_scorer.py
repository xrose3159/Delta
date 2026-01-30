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
except Exception as e:
    raise RuntimeError(
        "Failed to import vLLM. Please install vllm (pip install vllm) and ensure CUDA is available if using GPU."
    ) from e


class HESScorer:









        Steps:
          - Concatenate prompt + completion to form the full input per sample.
          - Run vLLM.generate with max_tokens=0 and prompt_logprobs, capturing token-level
            log-prob dictionaries for the input tokens.
          - Slice the prompt_logprobs to the completion span using tokenizer-aligned indices.
          - Compute per-token entropy and aggregate the top p percentile.

        Returns a list of HES scores, one per sample.
        if len(batch_prompts) != len(batch_completions):
            raise ValueError("batch_prompts and batch_completions must have the same length")


        full_texts: List[str] = [p + c for p, c in zip(batch_prompts, batch_completions)]


        prompt_token_lengths: List[int] = [
            len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in batch_prompts
        ]


        outputs = self.llm.generate(full_texts, self.sampling_params, use_tqdm=False)

        hes_scores: List[float] = []
        for i, out in enumerate(outputs):


            prompt_logprobs_list = getattr(out, "prompt_logprobs", None)
            if prompt_logprobs_list is None:

                hes_scores.append(0.0)
                continue


            completion_start = prompt_token_lengths[i]


            if len(batch_completions[i]) == 0:
                hes_scores.append(0.0)
                continue


            token_entropies: List[float] = []
            for pos in range(completion_start, len(prompt_logprobs_list)):
                token_dist = prompt_logprobs_list[pos]
                if not token_dist:

                    continue



                logprob_values: List[float] = []
                for v in token_dist.values():

                    logprob_values.append(float(v.logprob))

                if len(logprob_values) == 0:
                    continue

                logprob_tensor = torch.tensor(logprob_values, dtype=torch.float32)
                probs_tensor = F.softmax(logprob_tensor, dim=-1)
                entropy = torch.distributions.Categorical(probs=probs_tensor).entropy().item()
                token_entropies.append(entropy)


            if len(token_entropies) == 0:
                hes_scores.append(0.0)
                continue

            entropies_np = np.asarray(token_entropies, dtype=np.float32)



            threshold_percent = (1.0 - self.percentile_cutoff) * 100.0
            threshold = float(np.percentile(entropies_np, threshold_percent))


            selected = entropies_np[entropies_np >= threshold]
            if selected.size == 0:

                selected = entropies_np[np.argmax(entropies_np)][None]

            hes_score = float(np.sum(selected, dtype=np.float64))
            hes_scores.append(hes_score)

        return hes_scores


def stream_jsonl(path: str) -> Tuple[int, List[Dict]]:




    - records: List[original JSON objects]
    - prompts: List[str]
    - completions: List[str]
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


    failed_count = 0
    success_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fout:
        with tqdm(total=total_lines, desc="Scoring (HES)") as pbar:
            for records, prompts, completions in read_jsonl_in_batches(
                args.dataset_path, args.batch_size, args.prompt_key, args.completion_key
            ):

                for obj, prompt, completion in zip(records, prompts, completions):
                    try:

                        max_chars = 32000
                        if len(prompt) + len(completion) > max_chars:

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
                    fout.flush()

                pbar.update(len(records))

    print(f"HES scoring completed: {success_count} success, {failed_count} failed")


if __name__ == "__main__":
    main()
