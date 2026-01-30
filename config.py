import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_optional_float(key: str) -> Optional[float]:
    value = os.getenv(key)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        logger.warning("Invalid float for %s=%s, ignoring.", key, value)
        return None


@dataclass
class AdaptiveTrainingConfig:
    dataset_path: str = field(default_factory=lambda: _env(
        "DATASET_PATH",
        "./data/dataset.json",
    ))
    
    workspace_dir: Path = field(default_factory=lambda: Path(_env(
        "WORKSPACE_DIR",
        "./workspace",
    )))
    
    initial_dataset: Path = field(init=False) 
    test_dataset: Path = field(init=False)    
    output_dir: Path = field(init=False)     
    
    model_path: str = field(default_factory=lambda: _env(
        "MODEL_PATH", 
        "./models/Qwen3-VL-8B-Instruct"
    ))
    max_rounds: int = field(default_factory=lambda: int(_env("MAX_ROUNDS", "3")))
    num_rounds: int = field(init=False) 
    start_round: int = field(default_factory=lambda: int(_env("START_ROUND", "1")))
    round_total_samples: int = field(default_factory=lambda: int(_env("ROUND_TOTAL_SAMPLES", "-1")))
    random_seed: int = field(default_factory=lambda: int(_env("RANDOM_SEED", "42")))
    
    base_model_name: str = field(init=False)  
    model_base_path: Path = field(init=False) 
    
    enable_dpo: bool = field(default_factory=lambda: _env("ENABLE_DPO", "true").lower() in ("true", "1", "yes"))
    dpo_beta: float = field(default_factory=lambda: float(_env("DPO_BETA", "0.1")))
    dpo_epochs: int = field(default_factory=lambda: int(_env("DPO_EPOCHS", "1")))
    dpo_learning_rate: float = field(default_factory=lambda: float(_env("DPO_LEARNING_RATE", "5e-7")))
    dpo_batch_size: int = field(default_factory=lambda: int(_env("DPO_BATCH_SIZE", "1")))
    dpo_gradient_accumulation: int = field(default_factory=lambda: int(_env("DPO_GRADIENT_ACCUMULATION", "1")))
    dpo_max_length: int = field(default_factory=lambda: int(_env("DPO_MAX_LENGTH", "4096")))
    dpo_template: str = field(default_factory=lambda: _env("DPO_TEMPLATE", "qwen3_vl")) 
    dpo_num_gpus: int = field(default_factory=lambda: int(_env("DPO_NUM_GPUS", "4")))  
    dpo_memory_wait_time: int = field(default_factory=lambda: int(_env("DPO_MEMORY_WAIT_TIME", "10")))  
    
    dpo_freeze_vision_tower: bool = field(default_factory=lambda: _env("DPO_FREEZE_VISION_TOWER", "true").lower() in ("true", "1", "yes"))  
    dpo_freeze_projector: bool = field(default_factory=lambda: _env("DPO_FREEZE_PROJECTOR", "true").lower() in ("true", "1", "yes")) 
    dpo_freeze_llm: bool = field(default_factory=lambda: _env("DPO_FREEZE_LLM", "false").lower() in ("true", "1", "yes"))  
    
    enable_sft: bool = field(default_factory=lambda: _env("ENABLE_SFT", "true").lower() in ("true", "1", "yes"))  
    sft_num_train_epochs: float = field(default_factory=lambda: float(_env("SFT_NUM_TRAIN_EPOCHS", "3.0")))
    sft_per_device_train_batch_size: int = field(default_factory=lambda: int(_env("SFT_PER_DEVICE_TRAIN_BATCH_SIZE", "1")))
    sft_gradient_accumulation_steps: int = field(default_factory=lambda: int(_env("SFT_GRADIENT_ACCUMULATION_STEPS", "2")))
    sft_learning_rate: float = field(default_factory=lambda: float(_env("SFT_LEARNING_RATE", "5e-6")))
    sft_learning_rate_first_round: Optional[float] = field(
        default_factory=lambda: _env_optional_float("SFT_LR_FIRST_ROUND")
    )
    sft_learning_rate_other_rounds: Optional[float] = field(
        default_factory=lambda: _env_optional_float("SFT_LR_OTHER_ROUNDS")
    )
    sft_warmup_ratio: float = field(default_factory=lambda: float(_env("SFT_WARMUP_RATIO", "0.1")))
    sft_lr_scheduler_type: str = field(default_factory=lambda: _env("SFT_LR_SCHEDULER_TYPE", "cosine"))
    sft_logging_steps: int = field(default_factory=lambda: int(_env("SFT_LOGGING_STEPS", "1000")))
    sft_save_steps: int = field(default_factory=lambda: int(_env("SFT_SAVE_STEPS", "100")))
    sft_cutoff_len: int = field(default_factory=lambda: int(_env("SFT_CUTOFF_LEN", "10240")))
    sft_use_lora: bool = field(default_factory=lambda: _env("SFT_USE_LORA", "false").lower() in ("true", "1", "yes"))
    sft_lora_rank: int = field(default_factory=lambda: int(_env("SFT_LORA_RANK", "8")))
    sft_lora_alpha: int = field(default_factory=lambda: int(_env("SFT_LORA_ALPHA", "16")))
    sft_lora_dropout: float = field(default_factory=lambda: float(_env("SFT_LORA_DROPOUT", "0.05")))
    sft_freeze_vision_tower: bool = field(default_factory=lambda: _env("SFT_FREEZE_VISION_TOWER", "true").lower() in ("true", "1", "yes"))  
    sft_freeze_projector: bool = field(default_factory=lambda: _env("SFT_FREEZE_PROJECTOR", "false").lower() in ("true", "1", "yes"))  
    sft_freeze_llm: bool = field(default_factory=lambda: _env("SFT_FREEZE_LLM", "false").lower() in ("true", "1", "yes"))  
    sft_enable_packing: bool = field(default_factory=lambda: _env("SFT_ENABLE_PACKING", "false").lower() in ("true", "1", "yes"))
    resume_from_step: int = field(default_factory=lambda: int(_env("RESUME_FROM_STEP", "1")))
    
    manual_skip_to_upgrade: bool = field(default_factory=lambda: _env("MANUAL_SKIP_TO_UPGRADE", "false").lower() in ("true", "1", "yes"))
    manual_skip_round: int = field(default_factory=lambda: int(_env("MANUAL_SKIP_ROUND", "1")))
    manual_skip_model_path: str = field(default_factory=lambda: _env("MANUAL_SKIP_MODEL_PATH", ""))

    resume_model_path: str = field(default_factory=lambda: _env("RESUME_MODEL_PATH", ""))
    
    upgrade_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    upgrade_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash-exp"))
    upgrade_base_url: str = field(default_factory=lambda: _env("GEMINI_BASE_URL", ""))
    upgrade_max_retries: int = field(default_factory=lambda: int(_env("GEMINI_MAX_RETRIES", "3")))
    gemini_max_tokens: int = field(default_factory=lambda: int(_env("GEMINI_MAX_TOKENS", "65536")))
    gemini_max_output_tokens: int = field(default_factory=lambda: int(_env("GEMINI_MAX_OUTPUT_TOKENS", "65536")))
    

    corrected_cot_api_key: str = field(default_factory=lambda: _env("CORRECTED_COT_API_KEY", ""))
    corrected_cot_model: str = field(default_factory=lambda: _env("CORRECTED_COT_MODEL", "gemini-2.5-flash-thinking-24576"))
    corrected_cot_base_url: str = field(default_factory=lambda: _env("CORRECTED_COT_BASE_URL", ""))
    corrected_cot_max_workers: int = field(default_factory=lambda: int(_env("CORRECTED_COT_MAX_WORKERS", "50")))  
    
    enable_hes_filtering: bool = field(default_factory=lambda: _env("ENABLE_HES_FILTERING", "true").lower() in ("true", "1", "yes"))
    hes_percentile_cutoff: float = field(default_factory=lambda: float(_env("HES_PERCENTILE_CUTOFF", "0.005")))  
    hes_top_ratio: float = field(default_factory=lambda: float(_env("HES_TOP_RATIO", "0.2")))  
    hes_batch_size: int = field(default_factory=lambda: int(_env("HES_BATCH_SIZE", "8")))
    hes_tensor_parallel_size: int = field(default_factory=lambda: int(_env("HES_TENSOR_PARALLEL_SIZE", "8")))
    
    quality_check_enabled: bool = field(default_factory=lambda: _env("ENABLE_QUALITY_CHECK", "true").lower() in ("true", "1", "yes"))
    enable_quality_check: bool = field(init=False)  
    quality_check_attempts: int = field(default_factory=lambda: int(_env("QUALITY_CHECK_ATTEMPTS", "5")))
    quality_check_tensor_parallel_size: int = field(default_factory=lambda: int(_env("QUALITY_CHECK_TENSOR_PARALLEL_SIZE", "4")))
    
    enable_error_analysis: bool = field(default_factory=lambda: _env("ENABLE_ERROR_ANALYSIS", "true").lower() in ("true", "1", "yes"))
    error_analyzer_model_path: str = field(default_factory=lambda: _env(
        "ERROR_ANALYZER_MODEL_PATH",
        "./models/Qwen3-VL-30B-A3B-Thinking"
    ))
    
    use_llm_generator: bool = field(default_factory=lambda: _env("USE_LLM_GENERATOR", "true").lower() in ("true", "1", "yes"))
    llm_generator_model_path: str = field(default_factory=lambda: _env(
        "LLM_GENERATOR_MODEL_PATH",
        "./models/Qwen3-VL-30B-A3B-Thinking"
    ))
    llm_generator_tensor_parallel_size: int = field(default_factory=lambda: int(_env("LLM_GENERATOR_TENSOR_PARALLEL_SIZE", "8")))
    llm_generator_gpu_memory_utilization: float = field(default_factory=lambda: float(_env("LLM_GENERATOR_GPU_MEMORY_UTILIZATION", "0.85"))) 
    llm_generator_max_model_len: int = field(default_factory=lambda: int(_env("LLM_GENERATOR_MAX_MODEL_LEN", "98304")))  
    cot_model: str = field(init=False)  # 从 llm_generator_model_path 提取
    cot_temperature: float = field(default_factory=lambda: float(_env("COT_TEMPERATURE", "0.7")))
    cot_max_tokens: int = field(default_factory=lambda: int(_env("COT_MAX_TOKENS", "4096")))
    corrected_cot_max_tokens: int = field(default_factory=lambda: int(_env("CORRECTED_COT_MAX_TOKENS", "65536"))) 

    error_analyzer_tensor_parallel_size: int = field(default_factory=lambda: int(_env("ERROR_ANALYZER_TENSOR_PARALLEL_SIZE", "8")))
    error_analyzer_gpu_memory_utilization: float = field(default_factory=lambda: float(_env("ERROR_ANALYZER_GPU_MEMORY_UTILIZATION", "0.75")))
    error_analyzer_max_model_len: int = field(default_factory=lambda: int(_env("ERROR_ANALYZER_MAX_MODEL_LEN", "40000")))
    error_analyzer_max_tokens: int = field(default_factory=lambda: int(_env("ERROR_ANALYZER_MAX_TOKENS", "8192")))
    
    use_llm_judge: bool = field(default_factory=lambda: _env("USE_LLM_JUDGE", "true").lower() in ("true", "1", "yes"))
    judge_model_path: str = field(default_factory=lambda: _env(
        "JUDGE_MODEL_PATH",
        "./models/Qwen3-30B-A3B-Instruct-2507"
    ))
    judge_tensor_parallel_size: int = field(default_factory=lambda: int(_env("JUDGE_TENSOR_PARALLEL_SIZE", "4")))
    judge_gpu_memory_utilization: float = field(default_factory=lambda: float(_env("JUDGE_GPU_MEMORY_UTILIZATION", "0.8")))
    judge_temperature: float = field(default_factory=lambda: float(_env("JUDGE_TEMPERATURE", "0.0")))
    judge_max_tokens: int = field(default_factory=lambda: int(_env("JUDGE_MAX_TOKENS", "8192")))
    judge_max_model_len: int = field(default_factory=lambda: int(_env("JUDGE_MAX_MODEL_LEN", "70000"))) 
    
    eval_temperature: float = field(default_factory=lambda: float(_env("EVAL_TEMPERATURE", "0.2")))
    eval_top_p: float = field(default_factory=lambda: float(_env("EVAL_TOP_P", "1.0")))
    eval_frequency_penalty: float = field(default_factory=lambda: float(_env("EVAL_FREQUENCY_PENALTY", "0")))
    eval_max_tokens: int = field(default_factory=lambda: int(_env("EVAL_MAX_TOKENS", "32768")))
    eval_tensor_parallel_size: int = field(default_factory=lambda: int(_env("EVAL_TENSOR_PARALLEL_SIZE", "4")))
    
    use_apptainer: bool = field(default_factory=lambda: _env("USE_APPTAINER_FOR_VLLM", "false").lower() in ("true", "1", "yes"))
    apptainer_image: str = field(default_factory=lambda: _env("APPTAINER_IMAGE", ""))
    

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    save_intermediate_results: bool = True
    
    categories: Dict[int, str] = field(default_factory=lambda: {
        1: "Arithmetic and Number Theory",
        2: "Elementary Algebra",
        3: "Functions and Trigonometry",
        4: "Geometry",
        5: "Combinatorics and Discrete Mathematics",
        6: "Probability and Statistics",
        7: "Linear Algebra",
        8: "Calculus",
        9: "Logic and Proof",
    })
    
    category_descriptions: Dict[int, str] = field(default_factory=lambda: {
        1: "Arithmetic and Number Theory: integers, fractions, decimals; divisibility, primes, modular arithmetic, etc.",
        2: "Elementary Algebra: algebraic expressions, equations (linear/quadratic), inequalities, factoring, basic exponents and logarithms.",
        3: "Functions and Trigonometry: function concepts, graphs, transformations; trigonometric functions, inverse trig functions, trigonometric identities.",
        4: "Geometry: plane geometry (triangles, circles, etc.), solid geometry (volume, surface area), basic geometric proofs.",
        5: "Combinatorics and Discrete Mathematics: counting principles, permutations, combinations, inclusion-exclusion; graph theory, recurrence relations, Boolean logic, basic algorithmic thinking.",
        6: "Probability and Statistics: descriptive statistics, probability theory, random variables, distributions, expectation, variance, etc.",
        7: "Linear Algebra: vectors, matrices, matrix operations, determinants, eigenvalues, eigenvectors, systems of linear equations.",
        8: "Calculus: limits, derivatives, integrals, sequences and series, multivariable calculus (partial derivatives, multiple integrals, etc.).",
        9: "Logic and Proof: propositional logic, predicate logic, proofs (direct, contradiction, induction), sets, relations, functions."
    })
    
    def __post_init__(self):
        self.initial_dataset = Path(self.dataset_path)
        self.test_dataset = self.initial_dataset
        self.output_dir = self.workspace_dir
        model_path_obj = Path(self.model_path)
        self.base_model_name = model_path_obj.name
        self.model_base_path = model_path_obj.parent
        self.cot_model = Path(self.llm_generator_model_path).name
        self.enable_quality_check = self.quality_check_enabled
        self.num_rounds = self.max_rounds
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("logs",):
            (self.workspace_dir / subdir).mkdir(exist_ok=True)
        if not self.initial_dataset.exists():
            logger.warning(f"Dataset not found: {self.initial_dataset}")
        if not self.model_path:
            logger.warning("MODEL_PATH not set")
        if not self.upgrade_api_key:
            logger.warning("GEMINI_API_KEY not set – difficulty upgrade will be skipped")
        if self.resume_from_step < 1:
            self.resume_from_step = 1
    
    def get_round_dir(self, round_num: int) -> Path:
        round_dir = self.workspace_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
        return round_dir


def get_default_config() -> AdaptiveTrainingConfig:
    return AdaptiveTrainingConfig()
