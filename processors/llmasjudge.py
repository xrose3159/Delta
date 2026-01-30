import json
import os
import re
import argparse
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections import defaultdict
from transformers import AutoTokenizer

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_json(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            return data['data']
        raise ValueError(f"Expected a list at the root of {file_path}, got dict without 'data' list.")

    raise ValueError(f"Unsupported JSON structure in {file_path}: expected list or dict containing 'data'.")

def load_data(file_path: str) -> List[Dict[str, Any]]:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.jsonl':
        return load_jsonl(file_path)
    if ext == '.json':
        return load_json(file_path)

    raise ValueError(f"Unsupported input file extension '{ext}'. Expected '.jsonl' or '.json'.")

def save_data(data: List[Dict[str, Any]], file_path: str):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.jsonl':
        save_jsonl(data, file_path)
    elif ext == '.json':
        save_json(data, file_path)
    else:
        raise ValueError(f"Unsupported output file extension '{ext}'. Expected '.jsonl' or '.json'.")

def create_comparison_prompt(question:str, solution: str, response: str) -> str:
    prompt = f"""You are an expert evaluator. Compare the reference and generated answers only for semantic correctness and factual agreement.

## TASK
Determine whether the two answers express the same correct solution. Focus on meaning, correctness, and final results rather than wording or format.

## EVALUATION GUIDELINES
- **EQUIVALENT**: same conclusion or final answer, no substantive factual differences.
- **DIFFERENT**: conflicting conclusions, missing required reasoning, or any factual mistake in the generated answer.

## INPUT QUESTION
{question}

## REFERENCE ANSWER
{solution}

## GENERATED ANSWER
{response}

## OUTPUT INSTRUCTIONS
Respond in the following two-line format (no extra text):
Analysis: <concise reasoning>
JUDGMENT: <EQUIVALENT or DIFFERENT>"""

    return prompt


def _extract_boxed_content(text: str) -> list:
    results = []
    i = 0
    while i < len(text):
        start = text.find('\\boxed{', i)
        if start == -1:
            break
        brace_start = start + 7  
        brace_count = 1
        j = brace_start
        while j < len(text) and brace_count > 0:
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
            j += 1
        if brace_count == 0:
            results.append(text[brace_start:j-1])
            i = j
        else:
            i = brace_start
    return results

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_PHRASE_PATTERNS = [
    re.compile(r"\banswer\s*:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\bthe answer is\s*:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"\bthe answer is\s+([^\n]+)", re.IGNORECASE),
]


def _normalize_extracted_text(text: str) -> str:
    return text.strip().rstrip(". ").strip()


def extract_solution_text(solution: Any) -> str:
    if not isinstance(solution, str):
        return str(solution)

    boxed_matches = _extract_boxed_content(solution)
    if boxed_matches:
        return _normalize_extracted_text(boxed_matches[-1])

    answer_tag_match = ANSWER_TAG_PATTERN.search(solution)
    if answer_tag_match:
        return _normalize_extracted_text(answer_tag_match.group(1))

    for pattern in ANSWER_PHRASE_PATTERNS:
        phrase_match = pattern.search(solution)
        if phrase_match:
            return _normalize_extracted_text(phrase_match.group(1))

    return solution.strip()


def extract_response_text(response: Any) -> str:
    if not isinstance(response, str):
        return str(response)

    trimmed = response.strip()
    if not trimmed:
        return trimmed

    extracted = extract_solution_text(trimmed)
    return extracted or trimmed


def extract_judgment_batch(responses: List[str]) -> List[str]:
    judgments: List[str] = []
    for response in responses:
        parsed = 'unknown'
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            match = JUDGMENT_PATTERN.match(stripped)
            if match:
                parsed = match.group(1).lower()
                break
        judgments.append(parsed)

    return judgments

def prepare_batch_prompts(
    data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer
) -> Tuple[List[str], List[Tuple[int, int]]]:
    all_prompts = []
    all_indices = []

    for data_idx, item in enumerate(data):
        question = item.get('problem', '')
        answer = item.get('answer', '')
        if isinstance(answer, str) and answer.strip():
            solution = answer.strip()
        else:
            solution = extract_solution_text(item.get('solution', ''))

        raw_response = item.get('predict', item.get('model_prediction', ''))

        response = extract_response_text(raw_response)
        if response and solution:
            prompt = create_comparison_prompt(question, solution, response)
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            all_prompts.append(prompt)
            all_indices.append(data_idx)

    return all_prompts, all_indices

def batch_evaluate(
    data: List[Dict[str, Any]],
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer
) -> List[Dict[str, Any]]:
    all_prompts, all_indices = prepare_batch_prompts(data, tokenizer)

    print(f"🔥 Starting batch generation for {len(all_prompts)} prompts...", flush=True)
    outputs = llm.generate(all_prompts, sampling_params)

    print(f"✅ Generation completed, processing outputs...", flush=True)

    outputs_list = list(outputs)
    print(f"✅ Converted {len(outputs_list)} outputs to list", flush=True)


    responses = []
    for i, output in enumerate(outputs_list):
        try:
            text = output.outputs[0].text
            responses.append(text)
        except (IndexError, AttributeError) as e:
            print(f"⚠️  Warning: Failed to extract text from output {i}: {e}", flush=True)
            responses.append("")

    print(f"✅ Extracted {len(responses)} responses, parsing judgments...", flush=True)
    judgments = extract_judgment_batch(responses)

    for data_idx, response, judgment in zip(all_indices, responses, judgments):
        if judgment == 'equivalent':
            score = True
        elif judgment == 'different':
            score = False
        else:
            score = None
        data[data_idx]['match_analysis'] = response
        data[data_idx]['matched'] = score

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='math_with_rollouts.jsonl')
    parser.add_argument('--output_file', type=str, default='math_with_failscores.jsonl')

    parser.add_argument('--model', type=str, default='/path/to/user/Qwen3-4B-Instruct-2507')
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)

    parser.add_argument('--generation_size', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_model_len', type=int, default=70000, help='Maximum context length for Judge model')

    args = parser.parse_args()

    data = load_data(args.input_file)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len 
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.generation_size,
    )

    data = batch_evaluate(
        data,
        llm,
        sampling_params,
        tokenizer
    )

    import os
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_data(data, args.output_file)

def evaluate_predictions_with_judge(
    predictions_path: str,
    eval_records: List[Dict[str, Any]],
    model_path: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    temperature: float = 0.6,
    max_tokens: int = 8192,
    max_model_len: int = 70000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
