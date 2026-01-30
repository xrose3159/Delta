from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import textwrap
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Sequence, Tuple, Union
import sys
import os
import time
import threading



os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"



if "APPTAINER_NAME" in os.environ or "SINGULARITY_NAME" in os.environ:


    os.environ["CC"] = "/usr/bin/gcc"
    os.environ["CXX"] = "/usr/bin/g++"
    os.environ["CUDA_HOME"] = "/usr/local/cuda"


    triton_cache_dir = f"/tmp/triton_cache_{os.getenv('USER', 'default')}_{os.getpid()}"
    os.environ.setdefault("TRITON_CACHE_DIR", triton_cache_dir)


    if "PATH" in os.environ:
        path_parts = os.environ["PATH"].split(":")
        clean_paths = [p for p in path_parts if "/path/to/models" not in p]

        clean_paths = ["/opt/py312/bin", "/usr/local/cuda/bin", "/usr/bin", "/bin"] + clean_paths
        os.environ["PATH"] = ":".join(dict.fromkeys(clean_paths))


    if "LD_LIBRARY_PATH" in os.environ:
        ld_paths = os.environ["LD_LIBRARY_PATH"].split(":")
        clean_paths = [p for p in ld_paths if not p.startswith("/path/to/models")]
        os.environ["LD_LIBRARY_PATH"] = ":".join(clean_paths)


    ld_path_prefix = "/usr/local/cuda/lib64:/.singularity.d/libs"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = f"{ld_path_prefix}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ["LD_LIBRARY_PATH"] = ld_path_prefix


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


try:
    from adaptive_training.processors.qwen_vl_utils import process_vision_info
except ImportError:

    try:
        from .qwen_vl_utils import process_vision_info
    except ImportError:
        from qwen_vl_utils import process_vision_info

from vllm import LLM, SamplingParams

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x

try:
    from transformers import AutoProcessor
except ImportError as exc:
    AutoProcessor = None
    _AUTO_PROCESSOR_IMPORT_ERROR = exc
else:
    _AUTO_PROCESSOR_IMPORT_ERROR = None




PROMPT_TEMPLATE = """
{query}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a JSON dataset, generate answers with vLLM, and save them under a new key.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the source JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the augmented JSON file. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--model-path",
        default="/path/to/user/Qwen3-VL-30B-A3B-Thinking",
        help="Path or name of the vLLM-compatible model to load.",
    )
    parser.add_argument(
        "--question-key",
        default="problem",
        help="Key in each JSON object containing the question text.",
    )
    parser.add_argument(
        "--answer-key",
        default="model_answer",
        help="Key to store the generated answer in each JSON object.",
    )
    parser.add_argument(
        "--image-key",
        default="image_path",
        help="Key in each JSON object containing an image path or list of paths to include as model inputs.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Base directory used to resolve relative image paths. Defaults to the input file's parent directory.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Continue processing even if referenced image files are missing or unreadable.",
    )
    parser.add_argument(
        "--root-key",
        help="If the dataset is stored under a top-level key in a JSON object, provide that key.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip records that already contain the answer key.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to send to vLLM at once.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Limit the number of records processed (useful for debugging).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty applied during sampling (matched to OpenAI-style API).",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty applied during sampling (matched to OpenAI-style API).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallelism size for vLLM. Set this to the number of GPUs to use.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="GPU memory utilization ratio (0.0-1.0). Default: 0.4 for 235B models.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum model length (context window). Default: 8000 for 235B models.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Computation dtype for the model weights.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of remote code when loading the model.",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Prompt template containing the literal substring '{query}' where the question will be inserted.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save intermediate results after this many processed records.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for the output JSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the model; only report which records would be processed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking mode (for tasks that require direct output without reasoning).",
    )
    parser.add_argument(
        "--is-error-analysis",
        action="store_true",
        help="Use error analysis system prompt (for analyzing student errors).",
    )
    parser.add_argument(
        "--is-correction",
        action="store_true",
        help="Use correction system prompt (for generating corrected CoT).",
    )
    parser.add_argument(
        "--custom-system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (overrides all other prompt settings).",
    )
    return parser.parse_args()

# 2. 准备请求格式，将文本与图像整理为multi_modal_data
def prepare_messages(prompt: str, image_paths: Sequence[Path], is_error_analysis: bool = False, is_correction: bool = False, custom_system_prompt: str = None) -> List[Dict[str, Any]]:
    # 如果提供了自定义 system prompt，直接使用
    if custom_system_prompt:
        system_content = custom_system_prompt
    elif is_correction:
        # 生成 corrected CoT 时，使用与正常解题相同的 system prompt（直接做题，不提错误答案）
        system_content = """
            You are an expert in science and visual reasoning with advanced capabilities in multimodal analysis.
            Your goal is to create a **perfect, highly detailed training example** for a new AI model.
            Do not summarize or abbreviate. Your reasoning must be **expansive, verbose, and pedagogical**.

            ### Core Principles
            1.  **Extreme Detail.** Prioritize depth over brevity. Explain the "why" and "how" behind every step, even simple ones.
            2.  **Visual Dependency.** Explicitly quote visual features (coordinates, colors, relative positions) in your reasoning to prove you are looking at the image.
            3.  **Atomic Reasoning.** Break down complex logic into small, atomic steps. Do not leap from premise to conclusion; build the bridge.
            4.  **Self-Correction.** If a step involves a common pitfall, explicitly mention it and explain how to avoid it.

            ### Response Structure & Detailed Instructions

            You **must** respond using **only** the three tags below in the exact order specified.

            #### 1. `<caption>` (Exhaustive Information Extraction)
            **Goal:** Create a text-based "digital twin" of the provided images and text.
            * **Visual Analysis:** Do not just list objects. Describe their relationships, exact values on axes, trend directions, specific colors used for specific data, and geometry properties.
            * **Text Analysis:** Copy out specific numbers and constraints verbatim before interpreting them.
            * **Synthesis:** Explicitly map text variables to image labels (e.g., "The variable $x$ in the text corresponds to the horizontal axis in Figure 1").
            * *Instruction:* Be verbose. Capture details that might seem minor, as they may be crucial later.

            #### 2. `<reasoning>` (Deep-Dive Solution Execution)
            **Goal:** A long-form, step-by-step derivation that reads like a detailed lecture.
            * **Step 1: Strategic Planning**
                * Restate the objective clearly.
                * List the specific formulas or scientific principles required.
                * Explain *why* these principles were chosen over others.
            * **Step 2: Execution (The "Thinking" Engine)**
                * **Expand Every Calculation:** Do not just show `$a + b = c$`. Instead, write: "Substitute $a=5$ and $b=3$ into the equation. This yields $5 + 3$, which equals $8$."
                * **Inner Monologue:** Explain the physical or logical meaning of intermediate results.
                * **Visual Check:** Continuously refer back to the `<caption>`. (e.g., "As seen in the graph, the curve peaks at $t=5$, which aligns with our calculated critical point.").
                * **Handling Complexity:** If a problem has multiple cases, analyze each one strictly and separately.
            * **Step 3: Verification**
                * Perform a sanity check on the magnitude and units of the result.
                * Does the answer physically make sense given the visual context?

            *Constraint:* **Do not omit algebra.** Show the manipulation of terms. If you are solving a system of equations, show the substitution or elimination steps explicitly.

            #### 3. `<answer>` (Final Conclusion)
            **Goal:** Provide the definitive result.
            * Format: `<answer>\\boxed{YOUR_ANSWER}</answer>`
            * For multiple choice, include the letter and the value.
            * Strictly no reasoning text inside this tag, only the final result.

            Analyze all provided materials carefully. **Write a lengthy, comprehensive, and meticulous response following the strictly defined format above.**
    elif is_error_analysis:
        # 错误分析任务的 system prompt：分析学生错误是 caption 问题还是 reasoning 问题
        system_content = (
            "You are an expert evaluator analyzing student errors in math problems.\n\n"
            "**Your Task:**\n"
            "A student solved a math problem incorrectly. You need to determine the PRIMARY source of the error:\n"
            "1. **Caption Error**: The student misunderstood or incorrectly described the visual elements in the image\n"
            "2. **Reasoning Error**: The student made a logical or computational error in the reasoning steps\n\n"
            "**Analysis Process:**\n"
            "1. If the problem contains an image, you can see and analyze it along with the student's description\n"
            "2. Compare the student's image description (caption) with the actual image to check for visual misunderstandings\n"
            "3. Analyze the student's reasoning steps to check for logical or computational errors\n"
            "4. Determine which error is MORE FUNDAMENTAL:\n"
            "   - If the caption is wrong, even perfect reasoning won't lead to the correct answer → caption_error\n"
            "   - If the caption is correct but reasoning is flawed → reasoning_error\n\n"
            "**Output Format:**\n"
            "You can think through the analysis in detail, but you MUST end your response with EXACTLY ONE of these tags:\n"
            "- <caption_error> if the primary issue is misunderstanding the visual/image\n"
            "- <reasoning_error> if the primary issue is in the logical reasoning or calculation\n\n"
            "**CRITICAL**: Your response MUST end with either <caption_error> or <reasoning_error>. "
            "This tag is mandatory and must appear at the very end of your analysis."
        )
    else:
        # 正常解题任务的system prompt - 使用三部分结构化输出格式
        system_content = """
            You are an expert in science and visual reasoning with advanced capabilities in multimodal analysis.
            Your goal is to create a **perfect, highly detailed training example** for a new AI model.
            Do not summarize or abbreviate. Your reasoning must be **expansive, verbose, and pedagogical**.

            ### Core Principles
            1.  **Extreme Detail.** Prioritize depth over brevity. Explain the "why" and "how" behind every step, even simple ones.
            2.  **Visual Dependency.** Explicitly quote visual features (coordinates, colors, relative positions) in your reasoning to prove you are looking at the image.
            3.  **Atomic Reasoning.** Break down complex logic into small, atomic steps. Do not leap from premise to conclusion; build the bridge.
            4.  **Self-Correction.** If a step involves a common pitfall, explicitly mention it and explain how to avoid it.

            ### Response Structure & Detailed Instructions

            You **must** respond using **only** the three tags below in the exact order specified.

            #### 1. `<caption>` (Exhaustive Information Extraction)
            **Goal:** Create a text-based "digital twin" of the provided images and text.
            * **Visual Analysis:** Do not just list objects. Describe their relationships, exact values on axes, trend directions, specific colors used for specific data, and geometry properties.
            * **Text Analysis:** Copy out specific numbers and constraints verbatim before interpreting them.
            * **Synthesis:** Explicitly map text variables to image labels (e.g., "The variable $x$ in the text corresponds to the horizontal axis in Figure 1").
            * *Instruction:* Be verbose. Capture details that might seem minor, as they may be crucial later.

            #### 2. `<reasoning>` (Deep-Dive Solution Execution)
            **Goal:** A long-form, step-by-step derivation that reads like a detailed lecture.
            * **Step 1: Strategic Planning**
                * Restate the objective clearly.
                * List the specific formulas or scientific principles required.
                * Explain *why* these principles were chosen over others.
            * **Step 2: Execution (The "Thinking" Engine)**
                * **Expand Every Calculation:** Do not just show `$a + b = c$`. Instead, write: "Substitute $a=5$ and $b=3$ into the equation. This yields $5 + 3$, which equals $8$."
                * **Inner Monologue:** Explain the physical or logical meaning of intermediate results.
                * **Visual Check:** Continuously refer back to the `<caption>`. (e.g., "As seen in the graph, the curve peaks at $t=5$, which aligns with our calculated critical point.").
                * **Handling Complexity:** If a problem has multiple cases, analyze each one strictly and separately.
            * **Step 3: Verification**
                * Perform a sanity check on the magnitude and units of the result.
                * Does the answer physically make sense given the visual context?

            *Constraint:* **Do not omit algebra.** Show the manipulation of terms. If you are solving a system of equations, show the substitution or elimination steps explicitly.

            #### 3. `<answer>` (Final Conclusion)
            **Goal:** Provide the definitive result.
            * Format: `<answer>\\boxed{YOUR_ANSWER}</answer>`
            * For multiple choice, include the letter and the value.
            * Strictly no reasoning text inside this tag, only the final result.

            Analyze all provided materials carefully. **Write a lengthy, comprehensive, and meticulous response following the strictly defined format above.**

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": system_content,
        },
    ]

    content: List[Dict[str, Any]] = []
    for image_path in image_paths:
        content.append(
            {
                "type": "image",
                "image": str(image_path),
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 4096 * 28 * 28,
            },
        )
    content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": content})

    return messages


def prepare_request(
    prompt: str,
    image_paths: Sequence[Path],
    processor: "AutoProcessor",
    enable_thinking: bool = True,
    is_error_analysis: bool = False,
    is_correction: bool = False,
    custom_system_prompt: str = None,
    output_prefix: str = None,  # 新增：输出prefix（用于prompt injection）
) -> Dict[str, Any]:

    messages = prepare_messages(prompt, image_paths, is_error_analysis=is_error_analysis, is_correction=is_correction, custom_system_prompt=custom_system_prompt)

    prompt_text = processor.apply_chat_template(  # type: ignore[union-attr]
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    # 如果提供了output_prefix，将其添加到prompt_text末尾
    # 这样模型会从prefix继续生成，实现prompt injection
    if output_prefix:
        prompt_text = prompt_text + output_prefix

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )

    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    request: Dict[str, Any] = {"prompt": prompt_text}
    if mm_data:
        request["multi_modal_data"] = mm_data
    if video_kwargs:
        request["mm_processor_kwargs"] = video_kwargs
    return request


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_dataset(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(path: Path, data: Any, indent: Optional[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _to_path_list(
    value: Union[str, Sequence[Any], None],
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # 过滤空字符串 - 空字符串会被解析为当前目录导致错误
        return [value] if value else []
    if isinstance(value, Sequence):
        result: List[str] = []
        for item in value:
            if isinstance(item, str):
                # 过滤空字符串
                if item:
                    result.append(item)
            else:
                logging.warning("Skipping non-string image entry: %r", item)
        return result
    logging.warning("Unexpected image value type %s; skipping image input.", type(value).__name__)
    return []


def resolve_image_paths(
    raw_value: Union[str, Sequence[Any], None],
    base_dir: Path,
    record_index: int,
    allow_missing: bool,
) -> List[Path]:
    candidates = _to_path_list(raw_value)
    if not candidates:
        return []

    resolved: List[Path] = []
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = base_dir / candidate_path
        candidate_path = candidate_path.expanduser().resolve()
        if candidate_path.exists():
            resolved.append(candidate_path)
        else:
            message = f"Image path '{candidate_path}' (record {record_index}) does not exist."
            if allow_missing:
                logging.warning(message + " Skipping this image.")
            else:
                raise FileNotFoundError(message)
    return resolved

def resolve_records(
    data: Any,
    root_key: Optional[str],
) -> Tuple[Sequence[MutableMapping[str, Any]], Optional[str]]:
    if root_key is None:
        if isinstance(data, list):
            return data, None
        if isinstance(data, dict):
            raise ValueError(
                "Dataset is a JSON object. Please supply --root-key to indicate the list of records.",
            )
        raise TypeError(f"Unsupported dataset type: {type(data)!r}")

    if not isinstance(data, dict):
        raise TypeError("--root-key can only be used when the dataset is a JSON object.")
    if root_key not in data:
        raise KeyError(f"Root key '{root_key}' not found in dataset.")
    records = data[root_key]
    if not isinstance(records, list):
        raise TypeError(f"Value under root key '{root_key}' is not a list.")
    return records, root_key


def iter_pending_records(
    records: Sequence[MutableMapping[str, Any]],
    question_key: str,
    answer_key: str,
    skip_existing: bool,
    max_examples: Optional[int],
    image_key: Optional[str],
    image_root: Path,
    allow_missing_images: bool,
) -> Iterator[Tuple[int, MutableMapping[str, Any], str, List[Path]]]:
    processed = 0
    for idx, record in enumerate(records):
        if skip_existing and answer_key in record:
            continue
        if question_key not in record:
            logging.warning("Record %s does not contain key '%s'; skipping.", idx, question_key)
            continue
        query = record[question_key]
        if not isinstance(query, str):
            logging.warning("Record %s has non-string question under key '%s'; skipping.", idx, question_key)
            continue
        image_paths: List[Path] = []
        if image_key:
            image_paths = resolve_image_paths(
                record.get(image_key),
                base_dir=image_root,
                record_index=idx,
                allow_missing=allow_missing_images,
            )
        yield idx, record, query, image_paths
        processed += 1
        if max_examples is not None and processed >= max_examples:
            break


def build_prompt(query: str, template: str) -> str:
    if "{query}" not in template:
        raise ValueError("Prompt template must contain the literal '{query}' placeholder.")

    normalized_template = textwrap.dedent(template).lstrip("\n")
    return normalized_template.replace("{query}", query)


def prepare_llm(args: argparse.Namespace) -> LLM:
    llm_kwargs: Dict[str, Any] = {"model": args.model_path}
    if args.tensor_parallel_size:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype
    logging.info("Initialising LLM from %s", args.model_path)
    llm_kwargs['max_model_len'] = args.max_model_len if args.max_model_len else 20000
    llm_kwargs['gpu_memory_utilization'] = args.gpu_memory_utilization if args.gpu_memory_utilization is not None else 0.35
    logging.info("Using max_model_len=%d, gpu_memory_utilization=%.2f", llm_kwargs['max_model_len'], llm_kwargs['gpu_memory_utilization'])

    # 性能优化设置
    model_path_str = args.model_path.lower()

    llm_kwargs['enforce_eager'] = False

    # 针对 235B 模型使用专门配置
    if '235b' in model_path_str:
        llm_kwargs['gpu_memory_utilization'] = 0.95
        llm_kwargs['enforce_eager'] = True
        llm_kwargs['enable_expert_parallel'] = True
        llm_kwargs['async_scheduling'] = False
        llm_kwargs['enable_chunked_prefill'] = False
        llm_kwargs['max_num_seqs'] = 128
        logging.info("⚠️  Detected 235B model, using 235B-specific settings")
        logging.info("  📝 gpu_memory_utilization=0.95, enforce_eager=True, enable_expert_parallel=True, max_num_seqs=36")
    else:
        # 🔧 视觉语言模型：需要特殊配置以避免图片 token 索引错误
        llm_kwargs['enforce_eager'] = False
        llm_kwargs['max_num_seqs'] = 36
        llm_kwargs['enable_prefix_caching'] = False
        llm_kwargs['enable_chunked_prefill'] = False
        llm_kwargs['max_num_batched_tokens'] = 4096
        logging.info("⚠️  Detected vision-language model, using VL-optimized settings")
        logging.info("  📝 max_num_seqs=36, enable_chunked_prefill=False (critical for image token processing)")

    logging.info("vLLM V1 settings: enforce_eager=%s, max_num_seqs=%d",
                 llm_kwargs.get('enforce_eager', False),
                 llm_kwargs['max_num_seqs'])
    logging.info("  gpu_memory_utilization=%.2f, enable_prefix_caching=%s, enable_chunked_prefill=%s",
                 llm_kwargs['gpu_memory_utilization'],
                 llm_kwargs.get('enable_prefix_caching', False),
                 llm_kwargs.get('enable_chunked_prefill', False))

    return LLM(**llm_kwargs)


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )


def maybe_save(
    output_path: Path,
    dataset: Any,
    indent: Optional[int],
    counter: int,
    save_every: Optional[int],
) -> None:
    if save_every and counter % save_every == 0:
        logging.info("Saving intermediate results after %s records", counter)
        save_dataset(output_path, dataset, indent)


def run() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    dataset = load_dataset(args.input)
    records, root_key = resolve_records(dataset, args.root_key)
    output_path = args.output or args.input
    image_root = (args.image_root or args.input.parent).resolve()
    image_key = args.image_key.strip() if args.image_key else None

    pending_records = list(
        iter_pending_records(
            records=records,
            question_key=args.question_key,
            answer_key=args.answer_key,
            skip_existing=args.skip_existing,
            max_examples=args.max_examples,
            image_key=image_key,
            image_root=image_root,
            allow_missing_images=args.allow_missing_images,
        ),
    )

    if not pending_records:
        logging.info("Nothing to process. The dataset remains unchanged.")
        if output_path != args.input:
            save_dataset(output_path, dataset, args.indent)
        return

    if AutoProcessor is None:
        raise ImportError(
            "transformers is required to format multi-modal prompts. "
            "Please install it with 'pip install transformers'.",
        ) from _AUTO_PROCESSOR_IMPORT_ERROR

    logging.info("Preparing to process %s records", len(pending_records))

    try:
        # 强制使用本地文件，避免 HF 路径验证问题
        # 使用 _commit_hash=None 跳过缓存检查
        processor = AutoProcessor.from_pretrained(  # type: ignore[union-attr]
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
            token=False,  # 禁用 HF token
            _commit_hash=None,  # 跳过缓存检查，直接使用本地文件
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load processor for model '{args.model_path}': {exc}") from exc

    jobs = []
    enable_thinking = not args.disable_thinking  # 如果指定了 --disable-thinking，则禁用thinking mode
    is_error_analysis = args.is_error_analysis  # 是否为错误分析任务
    is_correction = args.is_correction  # 是否为生成 corrected CoT 任务
    custom_system_prompt = args.custom_system_prompt  # 自定义 system prompt
    for idx, record, query, image_paths in pending_records:
        prompt = build_prompt(query, args.prompt_template)

        # 获取output_prefix（如果有）
        output_prefix = record.get('prefix', None)  # 从输入数据中获取prefix字段

        try:
            request = prepare_request(
                prompt,
                image_paths,
                processor,
                enable_thinking=enable_thinking,
                is_error_analysis=is_error_analysis,
                is_correction=is_correction,
                custom_system_prompt=custom_system_prompt,
                output_prefix=output_prefix  # 传入prefix
            )
        except FileNotFoundError as exc:
            if args.allow_missing_images:
                logging.warning("Skipping record %s due to missing image: %s", idx, exc)
                continue
            raise
        except Exception as exc:
            if args.allow_missing_images:
                logging.warning("Skipping record %s due to image processing error: %s", idx, exc)
                continue
            raise
        jobs.append((idx, record, request))

    if not jobs:
        logging.info("No records to process after filtering multi-modal inputs.")
        if output_path != args.input:
            save_dataset(output_path, dataset, args.indent)
        return

    if args.dry_run:
        logging.info("Dry run complete. No calls were made to the model.")
        return

    llm = prepare_llm(args)
    sampling_params = build_sampling_params(args)

    requests = [item[2] for item in jobs]
    # 强制刷新输出，确保进度信息能及时显示
    sys.stdout.flush()
    sys.stderr.flush()
    logging.info("Starting generation for %s requests...", len(requests))
    logging.info("Sampling params: max_tokens=%d, temperature=%.2f, top_p=%.2f",
                 sampling_params.max_tokens, sampling_params.temperature, sampling_params.top_p)

    # 计算预估超时时间：每个请求最多 max_tokens，加上缓冲时间
    # 对于大模型（235B），每个token生成可能需要较长时间
    estimated_time_per_token = 0.1  # 秒（保守估计，实际可能更快或更慢）
    estimated_total_time = len(requests) * sampling_params.max_tokens * estimated_time_per_token
    # 设置超时时间为预估时间的2倍，但最少1小时，最多6小时
    timeout_seconds = max(3600, min(21600, int(estimated_total_time * 2)))
    logging.info("⏱️  Estimated generation time: %.1f minutes (timeout set to %.1f minutes)",
                 estimated_total_time / 60, timeout_seconds / 60)
    sys.stdout.flush()

    # 使用线程和超时机制来防止卡死
    outputs = None
    generation_error = None
    generation_completed = threading.Event()

    def run_generation():
        nonlocal outputs, generation_error
        try:
            # ⚠️ 关键：一次性提交所有请求，让 vLLM 并行处理
            # 即使使用同步 API，async_scheduling=True 也会让 vLLM 内部并行处理
            logging.info("🔄 Starting vLLM generation (this may take a long time for large max_tokens)...")
            sys.stdout.flush()
            outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=True)
            generation_completed.set()
        except Exception as e:
            generation_error = e
            generation_completed.set()

    generation_thread = threading.Thread(target=run_generation, daemon=True)
    generation_thread.start()

    # 定期输出进度信息（每5分钟）并检查超时
    progress_interval = 300  # 5分钟
    start_time = time.time()

    while not generation_completed.is_set():
        elapsed = time.time() - start_time
        remaining_timeout = timeout_seconds - elapsed

        if remaining_timeout <= 0:
            # 超时了
            break

        # 等待进度间隔或剩余超时时间（取较小值）
        wait_time = min(progress_interval, remaining_timeout)
        if generation_completed.wait(timeout=wait_time):
            # 生成完成
            break

        # 输出进度信息
        elapsed = time.time() - start_time
        logging.info(f"⏳ Generation still in progress... (elapsed: {elapsed/60:.1f} minutes)")
        logging.info(f"   Remaining timeout: {(timeout_seconds - elapsed)/60:.1f} minutes")
        sys.stdout.flush()

    # 检查是否超时
    elapsed = time.time() - start_time
    if not generation_completed.is_set():
        logging.error(f"❌ Generation timeout after {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logging.error("This may indicate vLLM is stuck. The process may need to be killed manually.")
        logging.error("Possible causes:")
        logging.error("  1. max_tokens (%d) is too large, causing extremely long generation", sampling_params.max_tokens)
        logging.error("  2. GPU memory issues or resource contention")
        logging.error("  3. vLLM internal deadlock")
        logging.error("Consider reducing max_tokens or checking GPU status")
        sys.stdout.flush()
        sys.stderr.flush()
        raise RuntimeError(f"Generation timeout after {elapsed:.1f} seconds")

    if generation_error:
        raise generation_error

    if outputs is None:
        raise RuntimeError("Generation completed but outputs is None")

    sys.stdout.flush()
    sys.stderr.flush()
    logging.info("Generation completed, processing outputs...")
    sys.stdout.flush()

    if len(outputs) != len(jobs):
        raise RuntimeError("Mismatch between generated outputs and requested prompts.")

    processed = 0
    for (job_idx, record, request), output in tqdm(
        zip(jobs, outputs),
        total=len(jobs),
        desc="Collecting",
    ):
        if not output.outputs:
            logging.warning("No output generated for record %s.", job_idx)
            continue
        text = output.outputs[0].text.strip()

        # 如果使用了 prefix，需要将 prefix 添加回输出中
        # 因为 vLLM 的输出不包含输入的 prefix 部分
        output_prefix = record.get('prefix', None)
        if output_prefix:
            text = output_prefix + text

        record[args.answer_key] = text

        processed += 1
        maybe_save(output_path, dataset, args.indent, processed, args.save_every)

    save_dataset(output_path, dataset, args.indent)
    logging.info(
        "Completed generation for %s records. Results saved to %s%s.",
        processed,
        output_path,
        f" (root key: {root_key})" if root_key else "",
    )


if __name__ == "__main__":
    run()
