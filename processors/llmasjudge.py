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
    """
    加载 JSON 数组格式的文件。
    注意：函数名保留为 load_jsonl 以保持兼容性，但实际只支持 JSON 数组格式。
    """
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
JUDGMENT: <EQUIVALENT or DIFFERENT>
"""

    return prompt

JUDGMENT_PATTERN = re.compile(r"^judgment\s*:\s*(equivalent|different)\s*$", re.IGNORECASE)

# 修复：使用更智能的方法匹配 \boxed{} 中的内容，支持嵌套括号（如 \frac{1}{18}）
def _extract_boxed_content(text: str) -> list:
    """提取所有 \\boxed{} 中的内容，正确处理嵌套括号"""
    results = []
    i = 0
    while i < len(text):
        # 查找 \boxed{
        start = text.find('\\boxed{', i)
        if start == -1:
            break
        # 从 { 开始计数括号
        brace_start = start + 7  # len('\\boxed{') = 7
        brace_count = 1
        j = brace_start
        while j < len(text) and brace_count > 0:
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
            j += 1
        if brace_count == 0:
            # 成功找到匹配的 }
            results.append(text[brace_start:j-1])
            i = j
        else:
            # 没有找到匹配的 }，跳过这个 \boxed{
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

    # 使用新的嵌套括号匹配函数
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
        
        # 支持多种字段名：predict（vLLM输出）或 model_prediction（旧格式）
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
    
    # 🔑 关键修复：立即将 outputs 转换为列表，避免懒加载导致的阻塞
    # vLLM 的 outputs 可能是生成器或包含未完成的异步操作
    outputs_list = list(outputs)
    print(f"✅ Converted {len(outputs_list)} outputs to list", flush=True)
    
    # 提取响应文本
    responses = []
    for i, output in enumerate(outputs_list):
        try:
            text = output.outputs[0].text
            responses.append(text)
        except (IndexError, AttributeError) as e:
            print(f"⚠️  Warning: Failed to extract text from output {i}: {e}", flush=True)
            responses.append("")  # 使用空字符串作为占位符
    
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

    parser.add_argument('--model', type=str, default='/mnt/dhwfile/raise/user/zhuyun/Qwen3-4B-Instruct-2507') # /share/wulijun/panzhuoshi/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5
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
        max_model_len=args.max_model_len  # 使用可配置的上下文长度
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
    """使用 LLM-as-Judge 评估预测结果
    
    Args:
        predictions_path: LLaMA-Factory 生成的 predictions.jsonl 路径
        eval_records: 评估数据集记录列表（ProblemRecord 转换的字典）
        model_path: Judge 模型路径
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU 内存利用率
        temperature: 采样温度
        max_tokens: 最大生成 token 数
        max_model_len: 模型允许的最大上下文长度（token 数）
    
    Returns:
        (评估后的数据, 统计信息字典)
    """
    # 1. 加载预测结果
    predictions = load_jsonl(predictions_path)
    
    # 2. 构建 pid/id 到 record 的映射（支持质量检测和正式评估两种场景）
    pid_to_record = {}
    for record in eval_records:
        # 优先使用 pid，如果没有则使用 id（转为字符串）
        key = record.get('pid') or str(record.get('id', ''))
        if key:
            pid_to_record[key] = record
    
    # 3. 构建评估数据格式
    eval_data = []
    for pred in predictions:
        # 从预测中获取 pid（支持三种情况）
        # 情况1：预测中有 _metadata.original_pid 字段（质量检测时）
        # 情况2：预测中直接有 pid 字段
        # 情况3：预测中有 id 字段（正式评估时）
        if '_metadata' in pred and 'original_pid' in pred['_metadata']:
            original_pid = pred['_metadata']['original_pid']
            attempt = pred['_metadata'].get('attempt', 1)
        elif 'pid' in pred:
            original_pid = pred['pid']
            attempt = 1
        elif 'id' in pred:
            original_pid = str(pred['id'])
            attempt = 1
        else:
            # 如果都没有，尝试从 eval_records 匹配（按顺序）
            if len(eval_data) < len(eval_records):
                original_pid = eval_records[len(eval_data)].get('pid', '')
                attempt = 1
            else:
                print(f"⚠️  Warning: Cannot determine pid for prediction {len(eval_data)}")
                continue
        
        # 获取对应的 record
        if original_pid not in pid_to_record:
            print(f"⚠️  Warning: PID {original_pid} not found in eval_records")
            continue
        
        record = pid_to_record[original_pid]
        
        eval_item = {
            'problem': record.get('question', ''),
            'answer': record.get('answer', ''),
            'model_prediction': pred.get('predict', pred.get('model_prediction', '')),  # 支持两种字段名
            'id': original_pid,  # 🔑 统一使用 'id' 字段，与输入数据一致
            'attempt': attempt,  # 保留 attempt 编号
            'category_id': record.get('category_id', 0),
            'category_name': record.get('category_name', 'Unknown'),
        }
        eval_data.append(eval_item)
    
    # 3. 初始化 LLM Judge 之前清理 GPU 显存
    print(f"🤖 Initializing LLM Judge: {model_path}")
    print(f"💾 GPU Memory Utilization: {gpu_memory_utilization}")
    # import torch
    # # 禁用 torch.compile 以避免 Triton 编译错误
    # torch._dynamo.config.disable = True
    # torch._dynamo.config.suppress_errors = True
    # print("✅ Disabled torch.compile to avoid Triton compilation errors")
    
    # # 强制清理 GPU 显存
    # try:
    #     import gc
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    #         gc.collect()
    #         print("✅ GPU memory cleared before loading Judge model")
    # except Exception as e:
    #     print(f"⚠️  Could not clear GPU memory: {e}")
    
    # # 显式设置设备为 CUDA，避免自动检测失败
    # import os
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     print("⚠️  CUDA_VISIBLE_DEVICES not set, setting to '0'")
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print(f'🛠️ Requested max_model_len: {max_model_len}', flush=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enforce_eager=False  # 启用 CUDA Graph 优化
    )

    try:
        actual_len = None
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_config'):
            actual_len = llm.llm_engine.model_config.max_model_len
        elif hasattr(llm, 'model_config'):
            actual_len = getattr(llm.model_config, 'max_model_len', None)

        if actual_len is not None:
            print(f'✅ vLLM initialized with max_model_len={actual_len}', flush=True)
            if actual_len < max_model_len:
                print(f'⚠️  WARNING: effective max_model_len ({actual_len}) < requested ({max_model_len})', flush=True)
        else:
            print('⚠️  Could not verify vLLM max_model_len from engine config', flush=True)
    except Exception as verify_error:
        print(f'⚠️  Failed to verify max_model_len: {verify_error}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # 4. 批量评估
    print(f"📊 Evaluating {len(eval_data)} predictions with LLM Judge...")
    
    # 评估完成后，不进行显式清理
    # 原因：在多GPU环境下，vLLM 的资源清理（特别是 del llm）可能会卡住
    # 解决方案：让进程自然退出，系统会自动清理所有资源（GPU内存、CUDA上下文、IPC等）
    # 这是最安全可靠的方式，避免了多GPU同步问题
    eval_data = batch_evaluate(eval_data, llm, sampling_params, tokenizer)
    
    # 5. 统计结果
    total = len(eval_data)
    matched = sum(1 for item in eval_data if item.get('matched') == True)
    different = sum(1 for item in eval_data if item.get('matched') == False)
    unknown = sum(1 for item in eval_data if item.get('matched') is None)
    
    stats = {
        'total': total,
        'correct': matched,
        'wrong': different,
        'unknown': unknown,
        'accuracy': matched / total if total > 0 else 0.0,
    }
    
    print(f"✅ Evaluation complete: {matched}/{total} correct ({stats['accuracy']*100:.2f}%)")
    if unknown > 0:
        print(f"⚠️  {unknown} predictions could not be judged")
    
    return eval_data, stats


if __name__ == "__main__":
    main()