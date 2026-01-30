

import gc
import json
import logging
import os
import re
import subprocess
import time
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from adaptive_training.processors.gemini_generator import REASONING_DIFFICULTY_LEVELS as DEFAULT_REASONING_LEVELS

from .sft_trainer import SFTTrainer
from .problem_validator import ProblemValidator
from .error_analyzer import ErrorAnalyzer
from .hes_scorer import HESScorer
from ..processors.sft_data_builder import SFTDataBuilder


logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


class MultiRoundPipelineDPO:
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "math_generation"))
        
        from adaptive_training.processors.gemini_generator import GeminiHardProblemGenerator


        self.sft_builder = SFTDataBuilder(config)
        self.sft_trainer = SFTTrainer(config)
        self.problem_validator = None
        self.error_analyzer = None
        self.llm_generator = None
        self.llm_judge = None
        self._init_llm_components()
        
        self.initial_model_path = config.model_path 
        self.current_model_path = config.model_path  
        self.round_history = []
        self.sft_data_files = []  
        
        self.min_reasoning_level = float(getattr(config, "min_reasoning_level", 1.0))
        self.max_reasoning_level = float(getattr(config, "max_reasoning_level", 10.0))
        self.reasoning_level_step = float(getattr(config, "reasoning_level_step", 0.5)) or 0.5
        self.reasoning_level_increment = float(getattr(config, "reasoning_level_increment", 0.5))
        reasoning_level_values = getattr(config, "reasoning_level_values", None)
        self.reasoning_scale = self._build_reasoning_scale(reasoning_level_values)
        self.default_reasoning_level = self._clamp_reasoning_level(
            float(getattr(config, "default_reasoning_level", 1.5))
        )
        
        self.min_visual_level = int(getattr(config, "min_visual_level", 1))
        self.max_visual_level = int(getattr(config, "max_visual_level", 7))
        self.visual_level_step = int(getattr(config, "visual_level_step", 1)) or 1
        self.visual_level_increment = int(getattr(config, "visual_level_increment", 1))
        self.default_visual_level = self._clamp_visual_level(
            int(getattr(config, "default_visual_level", 1))
        )

        self.next_generated_id: int = self._get_max_id_from_initial_dataset() + 1
        self.logger.info(f"📝 Next generated problem ID will start from: {self.next_generated_id}")
        self.gemini_generator = GeminiHardProblemGenerator(
            api_key=config.upgrade_api_key,
            model=config.upgrade_model,
            base_url=config.upgrade_base_url,
            max_tokens=config.gemini_max_tokens,
            max_output_tokens=config.gemini_max_output_tokens,
        )
        self.logger.info(f"✅ Gemini Generator initialized: {config.upgrade_model}")
        self.base_sft_learning_rate = self.config.sft_learning_rate
    
    def _init_llm_components(self):
        """初始化 LLM Generator 和 Judge（延迟导入）"""
        try:
            from adaptive_training.processors.llm_wrapper import LLMasGenerator, LLMasJudge
            
            self.logger.info("Initializing LLM Generator and Judge...")
            
    
            if self.config.use_llm_generator:
                self.llm_generator = LLMasGenerator(self.config)
                self.logger.info(f"✅ LLM Generator initialized: {self.config.llm_generator_model_path}")
            else:
                self.logger.warning("LLM Generator disabled")
            
    
            if self.config.use_llm_judge:
                self.llm_judge = LLMasJudge(self.config)
                self.logger.info(f"✅ LLM Judge initialized: {self.config.judge_model_path}")
            else:
                self.logger.warning("LLM Judge disabled")
    
            self.problem_validator = ProblemValidator(
                config=self.config,
                llm_generator=self.llm_generator,
                llm_judge=self.llm_judge
            )
            self.logger.info("✅ Problem Validator initialized")
            
            if self.llm_generator:
                self.error_analyzer = ErrorAnalyzer(
                    config=self.config,
                    llm_generator=self.llm_generator
                )
                self.logger.info("✅ Error Analyzer initialized")
            else:
                self.logger.warning("Error Analyzer disabled (LLM Generator not available)")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM components: {e}")
            raise
    
    def _get_max_id_from_initial_dataset(self) -> int:

        max_id = 0
        
        try:
        
            initial_dataset_path = Path(self.config.initial_dataset)
            if not initial_dataset_path.exists():
                self.logger.warning(f"Initial dataset not found: {initial_dataset_path}")
                return max_id
            
            with open(initial_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for problem in data:
         
                problem_id = problem.get('id', 0)
                
                try:
                    if isinstance(problem_id, int):
                        id_num = problem_id
                    elif isinstance(problem_id, str):
                        id_str = problem_id.split('_')[0]
                        id_num = int(id_str)
                    else:
                        continue
                    
                    if id_num > max_id:
                        max_id = id_num
                except (ValueError, AttributeError):
                    continue
            
            self.logger.info(f"📊 Found max ID in initial dataset: {max_id} (from {len(data)} problems)")
        except Exception as e:
            self.logger.warning(f"Failed to read max ID from initial dataset: {e}")
        
        return max_id

    def _build_reasoning_scale(self, custom_values: Optional[Any]) -> List[float]:
        values: List[float] = []
        if isinstance(custom_values, (list, tuple)):
            candidates = custom_values
        elif custom_values is None:
            candidates = None
        else:
            candidates = [custom_values]
        if candidates:
            for raw in candidates:
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                values.append(val)
        if not values:
            values = list(DEFAULT_REASONING_LEVELS)
        filtered = sorted(
            {val for val in values if self.min_reasoning_level <= val <= self.max_reasoning_level}
        )
        if filtered:
            return filtered
        step = self.reasoning_level_step if self.reasoning_level_step > 0 else 0.5
        generated: List[float] = []
        current = self.min_reasoning_level
        while current <= self.max_reasoning_level + 1e-9:
            generated.append(round(current, 2))
            current += step
        return generated or [self.min_reasoning_level, self.max_reasoning_level]

    def _align_reasoning_level(self, value: float, direction: str = "nearest") -> float:
        if not self.reasoning_scale:
            step = self.reasoning_level_step if self.reasoning_level_step > 0 else 0.5
            snapped = round(value / step) * step
            return float(min(self.max_reasoning_level, max(self.min_reasoning_level, snapped)))
        bounded = float(min(self.reasoning_scale[-1], max(self.reasoning_scale[0], value)))
        if direction == "up":
            for level in self.reasoning_scale:
                if level >= bounded - 1e-9:
                    return level
            return self.reasoning_scale[-1]
        if direction == "down":
            for level in reversed(self.reasoning_scale):
                if level <= bounded + 1e-9:
                    return level
            return self.reasoning_scale[0]
        return min(self.reasoning_scale, key=lambda lvl: (abs(lvl - bounded), lvl))

    def _clamp_reasoning_level(self, value: float) -> float:
        return self._align_reasoning_level(value, direction="nearest")
    
    def _clamp_visual_level(self, value: int) -> int:
        step = self.visual_level_step if self.visual_level_step > 0 else 1
        snapped = round(value / step) * step
        return int(min(self.max_visual_level, max(self.min_visual_level, snapped)))

    def _get_sft_learning_rate_for_round(self, round_num: int) -> float:
        if round_num == 1 and self.config.sft_learning_rate_first_round is not None:
            return self.config.sft_learning_rate_first_round
        if round_num > 1 and self.config.sft_learning_rate_other_rounds is not None:
            return self.config.sft_learning_rate_other_rounds
        return self.base_sft_learning_rate
    
    def _normalize_reasoning_level(self, raw_value: Optional[Any]) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = self.default_reasoning_level
        return self._clamp_reasoning_level(value)
    
    def _normalize_visual_level(self, raw_value: Optional[Any]) -> int:
        try:
            value = int(round(float(raw_value)))
        except (TypeError, ValueError):
            value = self.default_visual_level
        return self._clamp_visual_level(value)
    
    def _compute_target_levels(
        self,
        difficulty_aspect: str,
        current_reasoning: float,
        current_visual: int
    ) -> Tuple[float, int]:
        target_reasoning = current_reasoning
        target_visual = current_visual
        
        if difficulty_aspect == "reasoning":
            if self.reasoning_level_increment > 0:
                delta = self.reasoning_level_increment
            elif len(self.reasoning_scale) >= 2:
                delta = self.reasoning_scale[1] - self.reasoning_scale[0]
            else:
                delta = self.reasoning_level_step if self.reasoning_level_step > 0 else 0.5
            candidate = current_reasoning + delta
            target_reasoning = self._align_reasoning_level(candidate, direction="up")
            if target_reasoning <= current_reasoning and len(self.reasoning_scale) > 1:
                target_reasoning = self._align_reasoning_level(current_reasoning + 1e-9, direction="up")
        elif difficulty_aspect == "visual":
            delta = self.visual_level_increment if self.visual_level_increment > 0 else 1
            target_visual = self._clamp_visual_level(current_visual + delta)
        return target_reasoning, target_visual
    
    def run(self) -> None:
        pipeline_start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("Starting Multi-Round Pipeline with SFT")
        self.logger.info("=" * 80)
        self.logger.info(f"Number of rounds: {self.config.num_rounds}")
        self.logger.info(f"SFT enabled: {self.config.enable_sft}")
        self.logger.info(f"Error analysis enabled: {self.config.enable_error_analysis}")
        self.logger.info("=" * 80)
        
        start_round = max(1, getattr(self.config, "start_round", 1))
        if start_round > self.config.num_rounds:
            return
        
        if start_round == 1:
            test_data = self._load_initial_data()
        else:
            previous_round = start_round - 1
            test_data = self._load_next_round_dataset(previous_round)
            self._use_model_from_round(previous_round)
            self._recover_sft_history(start_round)
        
        for round_num in range(start_round, self.config.num_rounds + 1):
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"Round {round_num} / {self.config.num_rounds}")
            self.logger.info("=" * 80)
            
            try:
       
                test_data = self._run_single_round(round_num, test_data)
                
              
                self._save_round_summary(round_num)
                
            except Exception as e:
                self.logger.error(f"Round {round_num} failed: {e}", exc_info=True)
                break
        
        pipeline_elapsed = time.time() - pipeline_start_time
        self.logger.info("=" * 80)
        self.logger.info("✅ Multi-round training completed!")
        self.logger.info(f"⏱️  Total pipeline time: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes / {pipeline_elapsed/3600:.2f} hours)")
        self.logger.info("=" * 80)
    
    def _load_initial_data(self) -> List[Dict[str, Any]]:
        self.logger.info(f"Loading initial test data from: {self.config.test_dataset}")
        
        with open(self.config.test_dataset, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_count = len(data)
        self.logger.info(f"Loaded {total_count} test problems")

        if self.config.round_total_samples > 0 and total_count > self.config.round_total_samples:
            import random
            random.seed(self.config.random_seed)
            data = random.sample(data, self.config.round_total_samples)
            self.logger.info(f"Sampled {len(data)} problems for this round (from {total_count} total)")
        elif self.config.round_total_samples > 0 and total_count <= self.config.round_total_samples:
            self.logger.info(f"Using all {total_count} problems (less than or equal to round_total_samples={self.config.round_total_samples})")
        else:
            self.logger.info(f"Using all {total_count} problems (round_total_samples={self.config.round_total_samples} <= 0, no sampling)")
        
        return data
    
    def _recover_sft_history(self, start_round: int) -> None:
        recovered = 0
        for past_round in range(1, start_round):
            sft_dir = self.config.output_dir / f"round_{past_round}" / "sft"
            if not sft_dir.exists():
                continue
           
            candidate = sft_dir / "sft_data.json"
            if not candidate.exists():
                candidate = sft_dir / "sft_data_merged.json"
            
            if candidate.exists():
                self.sft_data_files.append(candidate)
                recovered += 1
    
    def _load_next_round_dataset(self, previous_round: int) -> List[Dict[str, Any]]:

        round_dir = self.config.output_dir / f"round_{previous_round}"
        next_round_file = round_dir / "next_round_dataset.json"
        
        
        self.logger.info("Loading next round dataset from: %s", next_round_file)
        with open(next_round_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(
            "Loaded %d problems from round %d output as the starting test set.",
            len(data),
            previous_round,
        )
        return data
    
    def _use_model_from_round(self, previous_round: int) -> None:
        resume_model_path = getattr(self.config, "resume_model_path", "")
        if resume_model_path and Path(resume_model_path).exists():
            self.current_model_path = resume_model_path
            self.logger.info(
                "Using manually specified RESUME_MODEL_PATH: %s",
                self.current_model_path,
            )
            return
        round_dir = self.config.output_dir / f"round_{previous_round}"
        candidate = round_dir / "sft" / f"sft_model_round{previous_round}"
        
        if candidate.exists():
            self.current_model_path = str(candidate)
            self.logger.info(
                "Resuming with model from round %d: %s",
                previous_round,
                self.current_model_path,
            )
    
    def _run_single_round(
        self,
        round_num: int,
        test_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        round_start_time = time.time()
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting Round {round_num}")
        self.logger.info(f"{'='*80}")
        
        round_dir = self.config.output_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        resume_from_step = max(1, getattr(self.config, "resume_from_step", 1))
        if resume_from_step > 1:
            self.logger.warning(
                "Resume mode enabled for Round %d: starting from Step %d",
                round_num,
                resume_from_step,
            )
        eval_results: List[Dict[str, Any]] = []
        if resume_from_step <= 1:
            self.logger.info(f"\n[Step 1/{10 if self.config.enable_sft else 8}] Testing model...")
            step_start_time = time.time()
            eval_results = self._test_model(round_num, test_data, round_dir)
            step_elapsed = time.time() - step_start_time
            self.logger.info(f"⏱️  Step 1 completed in {step_elapsed:.2f} seconds ({step_elapsed/60:.2f} minutes)")
        else:
            self.logger.info(
                f"\n[Step 1/{10 if self.config.enable_sft else 8}] Testing model... (skipped, resume_from_step={resume_from_step})"
            )
            eval_results = self._load_eval_results(round_dir)
            if not eval_results:
                raise RuntimeError("Failed to load cached eval_results for resume mode.")
            self.logger.info("Loaded %d cached evaluation results.", len(eval_results))

        if resume_from_step <= 2:
            self.logger.info(f"\n[Step 2/{10 if self.config.enable_sft else 8}] Separating correct/incorrect problems...")
            step_start_time = time.time()
            wrong_problems, correct_problems = self._separate_problems(eval_results, test_data, round_dir)
            step_elapsed = time.time() - step_start_time
            self.logger.info(f"⏱️  Step 2 completed in {step_elapsed:.2f} seconds")
            
            self.logger.info(f"  - Wrong problems: {len(wrong_problems)}")
            self.logger.info(f"  - Correct problems: {len(correct_problems)}")
            self._save_separated_problems(round_dir, wrong_problems, correct_problems)
        else:
            self.logger.info(
                f"\n[Step 2/{10 if self.config.enable_sft else 8}] Separating correct/incorrect problems... "
                f"(skipped, resume_from_step={resume_from_step})"
            )
            wrong_problems, correct_problems = self._load_saved_separated_problems(round_dir)
            if not wrong_problems:
                raise RuntimeError("Failed to load cached wrong_problems for resume mode.")
            self.logger.info(
                "Loaded cached problems: wrong=%d, correct=%d",
                len(wrong_problems),
                len(correct_problems),
            )
        
        step_num = 3
        manual_skip_to_upgrade = self._should_skip_to_upgrade(round_num)
        high_entropy_correct = []
        if getattr(self.config, "enable_hes_filtering", False) and correct_problems and not manual_skip_to_upgrade:
            if resume_from_step <= 2: 
                self.logger.info(f"\n[Step 2.5] Computing HES scores for {len(correct_problems)} correct problems...")
                step_start_time = time.time()
                high_entropy_correct = self._filter_high_entropy_correct_problems(correct_problems, round_dir)
                step_elapsed = time.time() - step_start_time
                self.logger.info(f"⏱️  Step 2.5 (HES Filtering) completed in {step_elapsed:.2f} seconds ({step_elapsed/60:.2f} minutes)")
                self.logger.info(f"✅ Selected {len(high_entropy_correct)} high-entropy correct problems (top {self.config.hes_top_ratio*100:.0f}%)")

                self._save_high_entropy_correct(round_dir, high_entropy_correct)
            else:     
                high_entropy_correct = self._load_high_entropy_correct(round_dir)
                if high_entropy_correct:
                    self.logger.info(f"Loaded {len(high_entropy_correct)} cached high-entropy correct problems")

        problems_for_cot = wrong_problems.copy()
        if high_entropy_correct:
            for p in high_entropy_correct:
                p["is_high_entropy_correct"] = True 
            problems_for_cot.extend(high_entropy_correct)
            self.logger.info(f"📊 Total problems for CoT generation: {len(problems_for_cot)} (wrong={len(wrong_problems)}, high_entropy_correct={len(high_entropy_correct)})")
        
        corrected_cots = {}
        if self.config.enable_sft and problems_for_cot and not manual_skip_to_upgrade:
            if resume_from_step <= 3:
                self.logger.info(f"\n[Step {step_num}/{10 if self.config.enable_sft else 8}] Generating corrected CoT for {len(problems_for_cot)} problems...")
                step_start_time = time.time()
                corrected_cots = self._generate_corrected_cots_for_wrong_problems(problems_for_cot, round_dir)
                step_elapsed = time.time() - step_start_time
                self.logger.info(f"⏱️  Step {step_num} (Corrected CoT Generation) completed in {step_elapsed:.2f} seconds ({step_elapsed/60:.2f} minutes)")
                self.logger.info(f"✅ Generated corrected CoT for {len(corrected_cots)}/{len(problems_for_cot)} problems")
                self._save_corrected_cots(round_dir, corrected_cots)
            else:
                self.logger.info(
                    f"\n[Step {step_num}/{10 if self.config.enable_sft else 8}] Loading cached corrected CoT... (resume_from_step={resume_from_step})"
                )
                corrected_cots = self._load_corrected_cots(round_dir)
                if corrected_cots:
                    self.logger.info(f"✅ Loaded {len(corrected_cots)} cached corrected CoTs")
                else:
                    self.logger.warning("No cached corrected CoTs found. Error analysis will be skipped.")
            step_num += 1
        elif manual_skip_to_upgrade:
            self.logger.info("Manual skip enabled - bypassing corrected CoT generation.")
            step_num += 1
        
        if wrong_problems:
            if manual_skip_to_upgrade:
                self.logger.info("Manual skip enabled - applying cached error analysis summary.")
                self._apply_saved_error_analysis(round_dir, wrong_problems)
                step_num += 1
            elif corrected_cots:
                if resume_from_step <= 4:
                    self.logger.info(f"\n[Step {step_num}/{10 if self.config.enable_sft else 8}] Analyzing error types using corrected CoT...")
                    step_start_time = time.time()
                    self._analyze_error_types_with_corrected_cot(wrong_problems, corrected_cots, round_dir)
                    step_elapsed = time.time() - step_start_time
                    self.logger.info(f"⏱️  Step {step_num} (Error Analysis) completed in {step_elapsed:.2f} seconds")
                    self._save_separated_problems(round_dir, wrong_problems, correct_problems)
                else:
                    self.logger.info(
                        f"\n[Step {step_num}/{10 if self.config.enable_sft else 8}] Analyzing error types using corrected CoT... "
                        f"(skipped, resume_from_step={resume_from_step})"
                    )
                    self._apply_saved_error_analysis(round_dir, wrong_problems)
                step_num += 1
            else:
                self.logger.warning("No corrected CoT available for error analysis.")
                step_num += 1
        

        step_num = 6
        if self.config.enable_sft and problems_for_cot and not manual_skip_to_upgrade:
            self.logger.info(f"\n[Step {step_num}/{10 if self.config.enable_sft else 8}] SFT Training on {len(problems_for_cot)} problems...")
            step_start_time = time.time()
            sft_success = self._run_sft_training(round_num, problems_for_cot, round_dir, corrected_cots)
            step_elapsed = time.time() - step_start_time
            self.logger.info(f"⏱️  Step {step_num} (SFT Training) completed in {step_elapsed:.2f} seconds ({step_elapsed/60:.2f} minutes)")
            
            if sft_success:
                self.current_model_path = str(round_dir / "sft" / f"sft_model_round{round_num}")
                self.logger.info(f"✅ SFT training completed. Updated model path: {self.current_model_path}")
            else:
                self.logger.warning("⚠️ SFT training failed, continuing with current model")
                raise RuntimeError("SFT training failed")
            
            step_num += 1
        elif manual_skip_to_upgrade:
            manual_model_path = getattr(self.config, "manual_skip_model_path", "")
            if not manual_model_path:
                raise RuntimeError("Manual skip to upgrade requires MANUAL_SKIP_MODEL_PATH to be set.")
            self.current_model_path = manual_model_path
            self.logger.info("Manual skip enabled - using pre-trained model: %s", manual_model_path)

            existing_sft_data = round_dir / "sft" / "sft_data.json"
            if existing_sft_data.exists():
                self.sft_data_files.append(existing_sft_data)
                self.logger.info("📊 Added existing sft_data.json to accumulation list: %s", existing_sft_data)
            else:
                self.logger.warning("⚠️ No existing sft_data.json found at %s for accumulation", existing_sft_data)
            
            step_num += 1
        else:
            if not wrong_problems:
                self.logger.info("ℹ️ No wrong problems, skipping SFT training")

        self.logger.info(f"\n[Step {step_num}/{10 if self.config.enable_sft else 8}] Preparing problems for difficulty upgrade...")
        step_start_time = time.time()
        
        problems_to_upgrade_wrong = [
            p for p in wrong_problems 
            if p.get('error_type') and p.get('error_type') != 'skipped'
        ]
        skipped_problems_count = len(wrong_problems) - len(problems_to_upgrade_wrong)
        
        problems_to_upgrade_hes = [
            p for p in high_entropy_correct 
            if str(p.get('id', '')) in corrected_cots
        ] if high_entropy_correct else []
        if problems_to_upgrade_wrong:
            caption_errors = sum(1 for p in problems_to_upgrade_wrong if p.get('error_type') == 'caption')
            reasoning_errors = sum(1 for p in problems_to_upgrade_wrong if p.get('error_type') == 'reasoning')
            unknown_errors = sum(1 for p in problems_to_upgrade_wrong if p.get('error_type') not in ['caption', 'reasoning', 'skipped'])
        step_num += 1
 
        upgraded_problems = []
        if problems_to_upgrade_wrong:
            upgraded_from_wrong = self._upgrade_difficulty(
                round_num, 
                problems_to_upgrade_wrong, 
                round_dir, 
                upgrade_mode="error_based"
            )
            upgraded_problems.extend(upgraded_from_wrong)

      
        if problems_to_upgrade_hes:
            for p in problems_to_upgrade_hes:
                p["error_type"] = "high_entropy"
            upgraded_from_hes = self._upgrade_difficulty(
                round_num,
                problems_to_upgrade_hes,
                round_dir,
                upgrade_mode="both"  
            )
            upgraded_problems.extend(upgraded_from_hes)
        
        upgraded_with_images = self._materialise_problem_images(round_num, upgraded_problems, round_dir)
        
        if len(upgraded_with_images) < len(upgraded_problems):
            discarded = len(upgraded_problems) - len(upgraded_with_images)
            self.logger.warning(f"⚠️  {discarded} problems discarded due to image generation failure")
        
        validated_problems = upgraded_with_images
        qualified_problems = validated_problems
        next_round_data = qualified_problems
        
        for problem in next_round_data:
            if 'image_path' in problem and isinstance(problem['image_path'], list):
                if problem['image_path']:
                    problem['image_path'] = problem['image_path'][0]
                else:
                    problem['image_path'] = ""
        
        
        next_round_file = round_dir / "next_round_dataset.json"
        with open(next_round_file, 'w', encoding='utf-8') as f:
            json.dump(next_round_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Next round dataset saved to: {next_round_file}")
        
        round_elapsed = time.time() - round_start_time
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"⏱️  Round {round_num} completed in {round_elapsed:.2f} seconds ({round_elapsed/60:.2f} minutes / {round_elapsed/3600:.2f} hours)")
        self.logger.info(f"{'='*80}\n")
 
        self._cleanup_gpu_processes()
        
        self.config.resume_from_step = 1
        
        return next_round_data
    
    def _should_skip_to_upgrade(self, round_num: int) -> bool:
        """Whether to skip directly to upgrade stage for a given round."""
        if not getattr(self.config, "manual_skip_to_upgrade", False):
            return False
        target_round = int(getattr(self.config, "manual_skip_round", 1))
        return round_num == target_round
    
    def _filter_invalid_sft_samples(
        self,
        problems: List[Dict[str, Any]],
        round_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        valid_samples: List[Dict[str, Any]] = []
        removed_ids: List[str] = []
        invalid_records: List[Dict[str, Any]] = []
        
        def _normalize_image_list(image_field: Any) -> List[str]:
            if isinstance(image_field, list):
                return [str(p).strip() for p in image_field if str(p).strip()]
            if isinstance(image_field, str):
                image_field = image_field.strip()
                return [image_field] if image_field else []
            return []
        
        for sample in problems:
            problem_id = str(sample.get("id", ""))
            question = sample.get("problem") or ""
            image_token_count = question.count("<image>")
            images = _normalize_image_list(sample.get("image_path", []))
            image_count = len(images)
            
            is_valid = False
            if image_token_count == 0 and image_count == 0:
                is_valid = True
            elif image_token_count > 0 and image_token_count == image_count:
                is_valid = True
            
            if is_valid:
                valid_samples.append(sample)
            else:
                removed_ids.append(problem_id)
                invalid_records.append(
                    {
                        "id": problem_id,
                        "image_token_count": image_token_count,
                        "image_path_count": image_count,
                        "image_paths": images,
                    }
                )
        
        if invalid_records:
            invalid_file = round_dir / "invalid_sft_samples.json"
            try:
                with invalid_file.open("w", encoding="utf-8") as f:
                    json.dump(invalid_records, f, ensure_ascii=False, indent=2)
                self.logger.info("Invalid SFT samples logged to %s", invalid_file)
            except Exception as exc:  
                self.logger.warning("Failed to write invalid SFT sample log: %s", exc)
        
        return valid_samples, removed_ids
    
    def _load_json_file(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to load JSON file %s: %s", path, exc)
            return None
    
    def _load_eval_results(self, round_dir: Path) -> List[Dict[str, Any]]:
        eval_file = round_dir / "eval" / "eval_results.json"
        data = self._load_json_file(eval_file)
        return data or []
    
    def _load_saved_separated_problems(
        self,
        round_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        eval_dir = round_dir / "eval"
        wrong = self._load_json_file(eval_dir / "wrong_problems.json") or []
        correct = self._load_json_file(eval_dir / "correct_problems.json") or []
        return wrong, correct
    
    def _save_corrected_cots(self, round_dir: Path, corrected_cots: Dict[str, str]) -> None:
        if not corrected_cots:
            return
        sft_dir = round_dir / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        file_path = sft_dir / "corrected_cots.json"
        serializable = {str(k): v for k, v in corrected_cots.items()}
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        self.logger.info("Corrected CoTs saved to %s", file_path)
    
    def _load_corrected_cots(self, round_dir: Path) -> Dict[str, str]:
        file_path = round_dir / "sft" / "corrected_cots.json"
        data = self._load_json_file(file_path) or {}
        return {str(k): v for k, v in data.items()}
    
    def _apply_saved_error_analysis(self, round_dir: Path, wrong_problems: List[Dict[str, Any]]) -> None:
        summary_file = round_dir / "error_analysis" / "error_analysis_summary.json"
        summary = self._load_json_file(summary_file)
        if not summary:
            self.logger.warning(
                "No cached error analysis summary found at %s. Proceeding without error types.",
                summary_file,
            )
            return
        mapping = {str(item.get("id")): item for item in summary if item.get("id") is not None}
        for problem in wrong_problems:
            pid = str(problem.get("id", ""))
            if pid in mapping:
                problem["error_type"] = mapping[pid].get("error_type")
                problem["error_reason"] = mapping[pid].get("error_reason", "")
    
    def _cleanup_gpu_processes(self):
  
        self.logger.info("🧹 Cleaning up residual GPU processes...")
        
        try:
            import subprocess
            current_pid = os.getpid()
            current_user = os.getenv("USER", "")
            
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.logger.warning("Failed to query nvidia-smi")
                return
            
            gpu_pids = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[0].strip())
                            memory = int(parts[1].strip())
                            gpu_pids.append((pid, memory))
                        except ValueError:
                            continue
            
            if not gpu_pids:
                self.logger.info("No GPU processes found")
                return
            
            self.logger.info(f"Found {len(gpu_pids)} GPU processes")
            
            killed_count = 0
            for pid, memory in gpu_pids:
                if pid == current_pid:
                    self.logger.debug(f"Skipping current process {pid}")
                    continue
                
                try:
                    proc_result = subprocess.run(
                        ["ps", "-o", "user=", "-p", str(pid)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    proc_user = proc_result.stdout.strip()
                    
                    if proc_user != current_user:
                        self.logger.debug(f"Skipping process {pid} (owned by {proc_user})")
                        continue
                    
                    cmd_result = subprocess.run(
                        ["ps", "-o", "cmd=", "-p", str(pid)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    cmd = cmd_result.stdout.strip()
                    
                    training_keywords = ["deepspeed", "llamafactory", "torchrun", "accelerate", "sft"]
                    is_training_process = any(kw in cmd.lower() for kw in training_keywords)
                    
                    if is_training_process:
                        self.logger.info(f"Killing residual training process {pid} (using {memory} MiB): {cmd[:80]}...")
                        os.kill(pid, signal.SIGKILL)
                        killed_count += 1
                    else:
                        self.logger.debug(f"Skipping non-training process {pid}: {cmd[:50]}...")
                        
                except ProcessLookupError:
                    pass  
                except Exception as e:
                    self.logger.debug(f"Error checking process {pid}: {e}")
            
            if killed_count > 0:
                self.logger.info(f"✅ Killed {killed_count} residual training processes")
                time.sleep(5)  
            else:
                self.logger.info("No residual training processes found")
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()
                    self.logger.info("CUDA cache cleared")
            except Exception as e:
                self.logger.debug(f"Failed to clear CUDA cache: {e}")
            
            time.sleep(10)
            self.logger.info("✅ GPU cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during GPU cleanup: {e}")
    
    def _test_model(
        self,
        round_num: int,
        test_data: List[Dict[str, Any]],
        round_dir: Path
    ) -> List[Dict[str, Any]]:
    
        eval_dir = round_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import torch
            torch.cuda.empty_cache()
            self.logger.info("Cleared CUDA cache")
        except ImportError:
            self.logger.warning("torch not available, skipping CUDA cache clear")
        
        test_file = eval_dir / "test_data.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Testing model: {self.current_model_path}")
        self.logger.info(f"Test data: {len(test_data)} problems")
        
        predictions_file = eval_dir / "predictions.json"
        self.logger.info("Generating predictions with vLLM...")
        
        success = self.llm_generator.generate_predictions(
            test_file=test_file,
            output_file=predictions_file,
            model_path=self.current_model_path
        )
        
        if not success:
            self.logger.error("Failed to generate predictions")
            raise RuntimeError("Prediction generation failed")
        
        if not predictions_file.exists():
            self.logger.error(f"Prediction file not found: {predictions_file}")
            raise RuntimeError("Prediction file not generated")
        
        self.logger.info(f"Predictions saved to: {predictions_file}")
        
        self.logger.info("Evaluation complete.")
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            self.logger.info("GPU memory released. Waiting 10 seconds for GPU to fully release memory...")
            time.sleep(10)  
            torch.cuda.synchronize()  
            torch.cuda.empty_cache()  
            
            if torch.distributed.is_initialized():
                self.logger.info("Cleaning up distributed process group...")
                torch.distributed.destroy_process_group()
                time.sleep(5)  
            
            self.logger.info("Ready to start LLM Judge.")
        except ImportError:
            self.logger.warning("torch not available, skipping GPU memory management")
            self.logger.info("Waiting 30 seconds for GPU to fully release memory...")
            time.sleep(30)
        except Exception as e:
            self.logger.warning(f"Error during GPU cleanup: {e}")
            self.logger.info("Waiting 15 seconds for GPU to fully release memory...")
            time.sleep(15)
        
        self.logger.info("Evaluating predictions with LLM Judge...")
        
        eval_records_file = eval_dir / "eval_records.json"
        eval_records = []
        for item in test_data:
            ground_truth_answer = item.get("answer", "") or item.get("model_answer", "")
            eval_records.append({
                "id": item.get("id"),
                "question": item.get("problem"),
                "answer": ground_truth_answer,  
                "category": item.get("category"),
                "category_name": item.get("category_name", "")
            })
        
        with open(eval_records_file, 'w', encoding='utf-8') as f:
            json.dump(eval_records, f, ensure_ascii=False, indent=2)
        
        judge_results_file = eval_dir / "judge_results.json"
        success = self.llm_judge.evaluate(
            predictions_file=predictions_file,
            eval_records_file=eval_records_file,
            output_file=judge_results_file
        )
        
        if not success:
            self.logger.error("Failed to evaluate predictions")
            raise RuntimeError("Judge evaluation failed")
        
        self.logger.info(f"Judge results saved to: {judge_results_file}")
        
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        with open(judge_results_file, 'r', encoding='utf-8') as f:
            judge_results_data = json.load(f)
        
        if isinstance(judge_results_data, dict) and "eval_data" in judge_results_data:
            judge_results = judge_results_data["eval_data"]
            judge_stats = judge_results_data.get("judge_stats", {})
            
            self.logger.info("=" * 70)
            self.logger.info("LLM Judge Statistics:")
            self.logger.info("  Total: %d", judge_stats.get('total', 0))
            self.logger.info("  Correct: %d", judge_stats.get('correct', 0))
            self.logger.info("  Wrong: %d", judge_stats.get('wrong', 0))
            self.logger.info("  Unknown: %d", judge_stats.get('unknown', 0))
            self.logger.info("  Accuracy: %.2f%%", judge_stats.get('accuracy', 0.0) * 100)
            self.logger.info("=" * 70)
        else:
            judge_results = judge_results_data
        
        pid_to_judge = {}
        for judge_item in judge_results:
            pid = judge_item.get("id")
            if pid:
                pid_to_judge[str(pid)] = judge_item  
        
        eval_results = []
        for i, item in enumerate(test_data):
            pid = str(item.get("id"))
            pred_item = predictions[i] if i < len(predictions) else {}
            judge_item = pid_to_judge.get(pid, {})
            
            eval_results.append({
                **item,
                "model_prediction": pred_item.get("predict", ""),  
                "matched": judge_item.get("matched", False),
                "judge_analysis": judge_item.get("match_analysis", ""),
                "reference_answer": item.get("answer", "") or item.get("model_answer", "")
            })
        
        results_file = eval_dir / "eval_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        correct = sum(1 for r in eval_results if r.get("matched", False))
        total = len(eval_results)
        self.logger.info(f"Evaluation complete: {correct}/{total} correct ({correct/total*100:.1f}%)")
        
        return eval_results
    
    def _separate_problems(
        self,
        eval_results: List[Dict[str, Any]],
        original_data: List[Dict[str, Any]],
        round_dir: Path
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        wrong_problems = []
        correct_problems = []
        
        for result in eval_results:
            reference_answer = result.get("answer", "") or result.get("reference_answer", "") or result.get("model_answer", "")
            
            problem_data = {
                "problem": result.get("problem", ""),
                "reference_answer": reference_answer,  
                "model_prediction": result.get("model_prediction", ""),
                "image_path": result.get("image_path", ""),  
                "category": result.get("category", 0),
                "category_name": result.get("category_name", ""),
                "id": result.get("id", 0),
                "answer": result.get("answer", ""),  
                "model_answer": result.get("model_answer", ""),  
                "reasoning_level": self._normalize_reasoning_level(result.get("reasoning_level")),
                "visual_level": self._normalize_visual_level(result.get("visual_level"))
            }
            
            if result.get("matched", False):
                correct_problems.append(problem_data)
            else:
                wrong_problems.append(problem_data)
        
        return wrong_problems, correct_problems
    
    def _save_separated_problems(
        self,
        round_dir: Path,
        wrong_problems: List[Dict[str, Any]],
        correct_problems: List[Dict[str, Any]]
    ) -> None:
        eval_dir = round_dir / "eval"
        
        with open(eval_dir / "wrong_problems.json", 'w', encoding='utf-8') as f:
            json.dump(wrong_problems, f, ensure_ascii=False, indent=2)
        
        with open(eval_dir / "correct_problems.json", 'w', encoding='utf-8') as f:
            json.dump(correct_problems, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved separated problems to {eval_dir}")
    
    def _filter_high_entropy_correct_problems(
        self,
        correct_problems: List[Dict[str, Any]],
        round_dir: Path
    ) -> List[Dict[str, Any]]:      
        if not correct_problems:
            return []
        
        hes_dir = round_dir / "hes"
        hes_dir.mkdir(parents=True, exist_ok=True)
        
        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "false").lower() in ("true", "1", "yes")
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")
        
        all_hes_scores = []
        
        if use_apptainer and apptainer_image:
            self.logger.info(f"Using Apptainer for HES calculation: {apptainer_image}")
            
            input_jsonl = hes_dir / "hes_input.jsonl"
            output_jsonl = hes_dir / "hes_output.jsonl"
            
            with open(input_jsonl, 'w', encoding='utf-8') as f:
                for p in correct_problems:
                    prompt = p.get("problem", "")
                    completion = p.get("model_prediction", "") or p.get("model_answer", "")
                    record = {
                        "id": p.get("id", ""),
                        "prompt": prompt,
                        "completion": completion,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            self.logger.info(f"Prepared {len(correct_problems)} samples for HES calculation")
            
            script_path = Path(__file__).parent / "hes_scorer.py"
            project_root = Path(__file__).parent.parent.parent
            triton_cache_dir = f"/tmp/triton_cache_{os.getenv('USER', 'default')}_{os.getpid()}"
            
            tensor_parallel_size = getattr(self.config, "hes_tensor_parallel_size", 8)
            percentile_cutoff = getattr(self.config, "hes_percentile_cutoff", 0.005)
            batch_size = getattr(self.config, "hes_batch_size", 8)
            
            cmd = [
                "apptainer", "exec", "--nv",
                "--cleanenv",
                "--bind", "/share:/share,/mnt:/mnt",
                "--env", "HF_HUB_OFFLINE=1",
                "--env", "TRANSFORMERS_OFFLINE=1",
                "--env", "HF_DATASETS_OFFLINE=1",
                "--env", "PYTHONUNBUFFERED=1",
                "--env", f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
                "--env", "VLLM_LOGGING_LEVEL=WARNING",
                "--env", f"TRITON_CACHE_DIR={triton_cache_dir}",
                "--env", "PATH=/opt/py312/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                "--env", "CC=/usr/bin/gcc",
                "--env", "CXX=/usr/bin/g++",
                "--env", "CUDAHOSTCXX=/usr/bin/g++",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--env", f"PYTHONPATH={project_root}",
                apptainer_image,
                "python", "-u", str(script_path),
                "--model_path", self.current_model_path,
                "--dataset_path", str(input_jsonl),
                "--output_path", str(output_jsonl),
                "--prompt_key", "prompt",
                "--completion_key", "completion",
                "--percentile_cutoff", str(percentile_cutoff),
                "--tensor_parallel_size", str(tensor_parallel_size),
                "--batch_size", str(batch_size),
                "--gpu_memory_utilization", "0.7",
                "--max_model_len", "16384",
            ]
            
            self.logger.info(f"Running HES scorer in Apptainer container...")
            self.logger.debug(f"Command: {' '.join(cmd)}")
            
            try:
                import subprocess
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,  
                )
                
                if result.returncode != 0:
                    self.logger.error(f"HES scorer failed with exit code {result.returncode}")
                    self.logger.error(f"stderr: {result.stderr[-2000:]}")  
                    return []
                
                self.logger.info("HES scorer completed successfully")
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')[-10:]
                    for line in stdout_lines:
                        self.logger.info(f"[HES] {line}")
                
            except subprocess.TimeoutExpired:
                self.logger.error("HES scorer timed out after 1 hour")
                return []
            except Exception as e:
                self.logger.error(f"Failed to run HES scorer: {e}")
                return []
            
            if not output_jsonl.exists():
                self.logger.error(f"HES output file not found: {output_jsonl}")
                return []
            
            hes_results = {}
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        hes_results[obj.get("id", "")] = obj.get("hes_score", 0.0)
            
            for p in correct_problems:
                p["hes_score"] = hes_results.get(p.get("id", ""), 0.0)
                all_hes_scores.append(p["hes_score"])
            
        else:
            self.logger.info("Using local HESScorer (Apptainer disabled)")
            
            try:
                scorer = HESScorer(
                    model_path=self.current_model_path,
                    tensor_parallel_size=getattr(self.config, "hes_tensor_parallel_size", 8),
                    percentile_cutoff=getattr(self.config, "hes_percentile_cutoff", 0.005),
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize HESScorer: {e}")
                self.logger.warning("Falling back to skipping HES filtering")
                return []
            
            prompts = []
            completions = []
            for p in correct_problems:
                prompt = p.get("problem", "")
                completion = p.get("model_prediction", "") or p.get("model_answer", "")
                prompts.append(prompt)
                completions.append(completion)
            
            batch_size = getattr(self.config, "hes_batch_size", 8)
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_completions = completions[i:i+batch_size]
                try:
                    scores = scorer.calculate_hes_for_batch(batch_prompts, batch_completions)
                    all_hes_scores.extend(scores)
                except Exception as e:
                    self.logger.warning(f"HES calculation failed for batch {i//batch_size}: {e}")
                    all_hes_scores.extend([0.0] * len(batch_prompts))
            
            del scorer
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            for idx, p in enumerate(correct_problems):
                p["hes_score"] = all_hes_scores[idx] if idx < len(all_hes_scores) else 0.0
        
        if not all_hes_scores:
            self.logger.warning("No HES scores calculated, returning empty list")
            return []
        
        sorted_problems = sorted(correct_problems, key=lambda x: x.get("hes_score", 0.0), reverse=True)
        
        top_ratio = getattr(self.config, "hes_top_ratio", 0.2)
        top_k = max(1, int(len(sorted_problems) * top_ratio))
        high_entropy_correct = sorted_problems[:top_k]
        
        with open(hes_dir / "correct_problems_with_hes.json", 'w', encoding='utf-8') as f:
            json.dump(sorted_problems, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"HES scores: min={min(all_hes_scores):.4f}, max={max(all_hes_scores):.4f}, mean={sum(all_hes_scores)/len(all_hes_scores):.4f}")
        self.logger.info(f"Selected top {top_k} high-entropy correct problems (threshold HES >= {high_entropy_correct[-1].get('hes_score', 0):.4f})")
        
        return high_entropy_correct
    
    def _save_high_entropy_correct(self, round_dir: Path, high_entropy_correct: List[Dict[str, Any]]) -> None:
        hes_dir = round_dir / "hes"
        hes_dir.mkdir(parents=True, exist_ok=True)
        with open(hes_dir / "high_entropy_correct.json", 'w', encoding='utf-8') as f:
            json.dump(high_entropy_correct, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved {len(high_entropy_correct)} high-entropy correct problems to {hes_dir}")
    
    def _load_high_entropy_correct(self, round_dir: Path) -> List[Dict[str, Any]]:
        file_path = round_dir / "hes" / "high_entropy_correct.json"
        return self._load_json_file(file_path) or []
    
    def _remove_think_content(self, text: str, problem_id: str) -> str:

        if not text:
            return text
        
        think_end_pattern = r'</think>\s*'
        match = re.search(think_end_pattern, text, re.IGNORECASE)
        
        if match:
            text = text[match.end():].strip()
            cleaned_text = text[match.end():].strip()
    
            return cleaned_text
        else:
            caption_match = re.search(r'<caption>', text, re.IGNORECASE)
            if caption_match and caption_match.start() > 500:
        
            
            return text
    
    def _extract_boxed_answer(self, text: str, problem_id: Optional[str] = None) -> Optional[str]:

        if not text:
            return None
        
        def extract_all_boxed(content: str) -> List[str]:
            results = []
            i = 0
            while i < len(content):
                start = content.find('\\boxed{', i)
                if start == -1:
                    break
                
                brace_start = start + 7  
                brace_count = 1
                j = brace_start
                
                while j < len(content) and brace_count > 0:
                    if content[j] == '{':
                        brace_count += 1
                    elif content[j] == '}':
                        brace_count -= 1
                    j += 1
                
                if brace_count == 0:
                    answer = content[brace_start:j-1].strip()
                    if answer:  
                        results.append(answer)
                    i = j
                else:
                    i = brace_start
            return results
        
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            boxed_in_answer = extract_all_boxed(answer_content)
            if boxed_in_answer:
                if len(boxed_in_answer) > 1:
                    logging.warning(f"Found {len(boxed_in_answer)} \\boxed{{}} in <answer> section, using the last one")
                return boxed_in_answer[-1]
            
            if '\\boxed{' in answer_content:
                problem_prefix = f"Problem {problem_id}: " if problem_id else ""
                self.logger.warning(f"{problem_prefix}<answer> contains incomplete \\boxed{{}} (missing closing brace), skipping")
            elif answer_content and answer_content not in ['$$', '$', '{}', '()', '[]', '']:
                cleaned_content = answer_content.strip('$').strip()
                if cleaned_content and len(cleaned_content) >= 1 and cleaned_content not in ['$$', '{}', '()', '$']:
                    problem_prefix = f"Problem {problem_id}: " if problem_id else ""
                    self.logger.warning(f"{problem_prefix}<answer> section missing \\boxed{{}}, auto-wrapping content: {cleaned_content[:50]}...")
                    return cleaned_content
        
        all_boxed = extract_all_boxed(text)
        if not all_boxed:
            return None
        
        unique_answers = set(all_boxed)
        if len(unique_answers) > 1:
            logging.warning(f"Found {len(all_boxed)} \\boxed{{}} with {len(unique_answers)} different answers in full text")
        
        return all_boxed[-1]
    
    def _normalize_image_paths(self, image_path_raw: Union[str, List[str]]) -> List[str]:
        if isinstance(image_path_raw, list):
            return image_path_raw
        elif isinstance(image_path_raw, str) and image_path_raw:
            return [image_path_raw]
        else:
            return []
    
    def _get_problem_id(self, problem: Dict[str, Any]) -> str:
        problem_id_val = problem.get('id')
        if problem_id_val is None or problem_id_val == 0:
            return str(hash(problem.get('problem', '')) % 100000)
        else:
            return str(problem_id_val)
    
    def _validate_corrected_cot_format(
        self,
        corrected_cot: str,
        problem_id: str,
        wrong_answer: str
    ) -> Tuple[bool, Optional[str]]:
        corrected_box = self._extract_boxed_answer(corrected_cot, problem_id)
        if not corrected_box:
            return False, f"Cannot extract answer from <answer> section"
        
        sentences = corrected_cot.split('.')
        if len(sentences) > 10:
            for j in range(len(sentences) - 3):
                if sentences[j] == sentences[j+1] == sentences[j+2] == sentences[j+3]:
                    return False, "Contains repeated sentences"
        
        return True, None
    
    def _validate_corrected_cots_with_judge(
        self,
        temp_corrected_cots: Dict[str, str],
        wrong_problems: List[Dict[str, Any]],
        problem_ids: List[str],
        valid_inputs: List[Tuple[int, Any]],
        round_dir: Path = None
    ) -> Dict[str, str]:
        predictions = []
        eval_records = []
        id_to_cot = {}  
        
        for problem_id, corrected_cot in temp_corrected_cots.items():
            problem = None
            for wp in wrong_problems:
                if str(wp.get('id')) == str(problem_id):
                    problem = wp
                    break
            
            if not problem:
                self.logger.warning(f"Problem {problem_id} not found in wrong_problems")
                continue
            
            pred_item = {
                'id': problem_id,
                'model_prediction': corrected_cot  
            }
            predictions.append(pred_item)
            
            eval_item = {
                'pid': problem_id,
                'question': problem.get('problem', ''),
                'answer': problem.get('answer', ''),
                'image_path': problem.get('image_path', '')
            }
            eval_records.append(eval_item)
            
            id_to_cot[problem_id] = corrected_cot
        
        if not predictions:
            self.logger.warning("No predictions to validate with judge")
            return {}

        if round_dir:
            judge_dir = round_dir / "sft" / "judge_validation"
            judge_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Saving Judge validation files to: {judge_dir}")
        else:
            judge_dir = Path(self.config.output_dir) / "temp_judge_validation"
            judge_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"⚠️  round_dir not provided, using global temp dir: {judge_dir}")
        
        predictions_file = judge_dir / "corrected_predictions.json"
        eval_records_file = judge_dir / "corrected_eval_records.json"
        judge_results_file = judge_dir / "corrected_judge_results.json"
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        with open(eval_records_file, 'w', encoding='utf-8') as f:
            json.dump(eval_records, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📝 Judge input files saved:")
        self.logger.info(f"   - Predictions: {predictions_file}")
        self.logger.info(f"   - Eval records: {eval_records_file}")
        
        self.logger.info(f"Validating {len(predictions)} corrected CoTs with LLM-as-Judge...")
        success = self.llm_judge.evaluate(
            predictions_file=predictions_file,
            eval_records_file=eval_records_file,
            output_file=judge_results_file
        )
        
        if not success or not judge_results_file.exists():
            self.logger.error("LLM-as-Judge validation failed")
            return temp_corrected_cots  
        
        self.logger.info(f"✅ Judge results saved to: {judge_results_file}")
        
        with open(judge_results_file, 'r', encoding='utf-8') as f:
            judge_data = json.load(f)
        
        if isinstance(judge_data, dict) and 'eval_data' in judge_data:
            judge_results = judge_data['eval_data']
        elif isinstance(judge_data, list):
            judge_results = judge_data
        else:
            self.logger.error(f"Unexpected judge result format: {type(judge_data)}")
            return temp_corrected_cots
        
        validated_cots = {}
        failed_details = []  
        
        for result in judge_results:
            if not isinstance(result, dict):
                self.logger.warning(f"Skipping non-dict result: {result}")
                continue
                
            problem_id = str(result.get('id'))
            matched = result.get('matched', False)
            
            if matched and problem_id in id_to_cot:
                validated_cots[problem_id] = id_to_cot[problem_id]
                self.logger.debug(f"Problem {problem_id}: corrected CoT validated by judge")
            elif problem_id in id_to_cot:
                self.logger.warning(f"Problem {problem_id}: corrected CoT answer is wrong according to judge")
                failed_details.append({
                    'problem_id': problem_id,
                    'matched': matched,
                    'match_analysis': result.get('match_analysis', 'N/A')
                })
        
        if failed_details and round_dir:
            failed_report_file = judge_dir / "failed_validation_report.json"
            with open(failed_report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_validated': len(judge_results),
                    'passed': len(validated_cots),
                    'failed': len(failed_details),
                    'failed_problems': failed_details
                }, f, ensure_ascii=False, indent=2)
            self.logger.info(f"📋 Failed validation report saved to: {failed_report_file}")
        
        return validated_cots
    
    def _prepare_batch_inputs(
        self,
        wrong_problems: List[Dict[str, Any]]
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[str]]:
        batch_inputs = []
        problem_ids = []
        
        for problem in wrong_problems:
            problem_id = self._get_problem_id(problem)
            problem_ids.append(problem_id)
            
            problem_text = problem.get('problem', '')
            wrong_answer = problem.get('model_prediction', '')
            image_paths = self._normalize_image_paths(problem.get('image_path', ''))
            
            if not problem_text or not wrong_answer:
                self.logger.warning(f"Skipping problem {problem_id}: missing problem or wrong_answer")
                batch_inputs.append(None)
                continue
            
            if '<image>' not in problem_text and image_paths:
                problem_text = '<image> ' + problem_text
            
            batch_inputs.append({
                "problem": problem_text,
                "image_path": image_paths
            })
        
        return batch_inputs, problem_ids
    
    def _analyze_error_types_with_corrected_cot(
        self,
        wrong_problems: List[Dict[str, Any]],
        corrected_cots: Dict[str, str],
        round_dir: Path
    ) -> None:
        if not wrong_problems:
            return
        
        error_analysis_dir = round_dir / "error_analysis"
        error_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        enable_error_analysis = getattr(self.config, 'enable_error_analysis', True)
        
        if not enable_error_analysis or not self.error_analyzer:
            self.logger.warning("Error analysis not enabled or ErrorAnalyzer not available, all wrong problems will be marked as reasoning")
            for problem_data in wrong_problems:
                problem_data["error_type"] = "reasoning"
                problem_data["error_reason"] = "Error analysis disabled or ErrorAnalyzer not available"
            return
        
        problems_to_analyze = []
        skipped_count = 0
        
        for problem in wrong_problems:
            problem_id = str(problem.get("id", ""))
            if problem_id in corrected_cots:
                problem["reference_answer"] = corrected_cots[problem_id]
                problems_to_analyze.append(problem)
                self.logger.debug(f"Problem {problem_id}: using corrected CoT as reference_answer")
            else:
                problem["error_type"] = "skipped"
                problem["error_reason"] = "No valid corrected CoT available"
                skipped_count += 1
                self.logger.warning(f"Problem {problem_id}: no corrected CoT, skipping error analysis and new problem generation")
        
        if skipped_count > 0:
            self.logger.warning(f"⚠️  {skipped_count} problems skipped (no corrected CoT): will not be analyzed or used for generating new problems")
        
        if not problems_to_analyze:
            self.logger.warning("No problems to analyze (all problems have no corrected CoT)")
            return
        
        try:
            self.logger.info(f"Starting batch analysis of {len(problems_to_analyze)} wrong problems (using corrected CoT as reference)...")
            
            error_results = self.error_analyzer.analyze_errors_batch(
                wrong_problems=problems_to_analyze,
                temp_dir=error_analysis_dir
            )
            
            for i, (error_type, error_reason) in enumerate(error_results):
                if i < len(wrong_problems):
                    wrong_problems[i]["error_type"] = error_type
                    wrong_problems[i]["error_reason"] = error_reason
            
            self.logger.info(f"✅ Batch error analysis completed")
            
        except Exception as e:
            self.logger.error(f"Batch error analysis failed: {e}, all wrong problems will be marked as reasoning")
            for problem_data in problems_to_analyze:
                problem_data["error_type"] = "reasoning"
                problem_data["error_reason"] = f"Batch analysis failed: {str(e)}"
        
        if wrong_problems:
            analysis_summary_file = error_analysis_dir / "error_analysis_summary.json"
            error_summary = []
            for wp in wrong_problems:
                error_summary.append({
                    "id": wp.get("id"),
                    "error_type": wp.get("error_type"),
                    "error_reason": wp.get("error_reason", "")[:200],  
                    "problem_preview": wp.get("problem", "")[:100]
                })
            with open(analysis_summary_file, 'w', encoding='utf-8') as f:
                json.dump(error_summary, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Error analysis summary saved to: {analysis_summary_file}")
    
    def _generate_corrected_cots_with_api(
        self,
        batch_input_file: Path,
        batch_output_file: Path,
        max_tokens: int
    ) -> bool:
        """
        Use API to generate corrected CoT (instead of local vLLM deployment), use concurrent acceleration
        """
        import openai
        import base64
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with open(batch_input_file, 'r', encoding='utf-8') as f:
            inputs = json.load(f)
        
        api_base = self.config.corrected_cot_base_url
        api_key = self.config.corrected_cot_api_key
        api_model = self.config.corrected_cot_model
        max_workers = min(self.config.corrected_cot_max_workers, len(inputs))
        
        self.logger.info(f"Corrected CoT API configuration: model={api_model}, base={api_base}")
        self.logger.info(f"Using concurrent generation with max_workers={max_workers}")
        
        client = openai.OpenAI(base_url=api_base, api_key=api_key)
        
        system_prompt = """
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
        """





        def generate_single_cot(idx_and_item):
            idx, item = idx_and_item
            problem = item.get('problem', '')
            image_paths = item.get('image_path', [])
            
            if not isinstance(image_paths, list):
                image_paths = [image_paths] if image_paths else []
            
            def detect_repetition(text, window=200, threshold=3):
                if len(text) < window * 2:
                    return False
                last_window = text[-window:]
                search_region = text[-window*5:-window]
                count = search_region.count(last_window[:50]) if len(last_window) >= 50 else 0
                return count >= threshold
            
            def is_valid_cot(text):
                if not text or len(text) == 0:
                    return False, "Empty"
                
                if detect_repetition(text):
                    return False, "Repetition detected"
                
                text_lower = text.lower()
                if '<caption>' not in text_lower:
                    return False, "Missing <caption> tag"
                if '<reasoning>' not in text_lower:
                    return False, "Missing <reasoning> tag"
                if '<answer>' not in text_lower:
                    return False, "Missing <answer> tag"
                
                return True, "Valid"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": problem}]}
            ]
            
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        with open(img_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        img_ext = os.path.splitext(img_path)[1].lower()
                        mime_type = {
                            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                            '.png': 'image/png', '.gif': 'image/gif',
                            '.webp': 'image/webp'
                        }.get(img_ext, 'image/jpeg')
                        
                        messages[1]["content"].insert(0, {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_data}"}
                        })
                    except Exception as e:
                        self.logger.warning(f"[Thread-{idx+1}] cannot read image {img_path}: {e}")
            
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    temperature = 0.7 + (attempt * 0.1)  
                    frequency_penalty = 0.3 + (attempt * 0.2)  
                    
                    response = client.chat.completions.create(
                        model=api_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=min(temperature, 1.0),
                        frequency_penalty=min(frequency_penalty, 1.0),
                        presence_penalty=0.3
                    )
                    
                    message = response.choices[0].message
                    full_content = (getattr(message, "content", None) or "").strip()
                    reasoning_content = (getattr(message, "reasoning_content", None) or "").strip()

                    if reasoning_content:
                        import re

                        def _replace_reasoning(match):
                            return f"{match.group(1)}{reasoning_content}{match.group(3)}"

                        updated_content, replaced = re.subn(
                            r"(<reasoning>)(.*?)(</reasoning>)",
                            _replace_reasoning,
                            full_content,
                            count=1,
                            flags=re.IGNORECASE | re.DOTALL
                        )

                        if replaced > 0:
                            prediction = updated_content
                        else:
                            prediction = full_content
                            self.logger.warning("Reasoning content returned but <reasoning> tag not found; using original content.")
                    else:
                        prediction = full_content

                    is_valid, reason = is_valid_cot(prediction)
                    
                    if is_valid:
                        if attempt > 0:
                            self.logger.info(f"✅ [Thread-{idx+1}] Retry successful (attempt {attempt+1}/{len(inputs)})")
                        else:
                            self.logger.info(f"✅ [Thread-{idx+1}] Completed {idx+1}/{len(inputs)}")
                        return idx, {'predict': prediction}
                    else:
                        self.logger.warning(f"⚠️  [Thread-{idx+1}] Validation failed ({reason}), trying again {attempt+1}/{max_retries}")
                        last_error = reason
                        if attempt < max_retries - 1:
                            import time
                            time.sleep(2)  
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"⚠️  [Thread-{idx+1}] API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(5)  
                        continue
            
            self.logger.error(f"❌ [Thread-{idx+1}] All retries failed, last error: {last_error}")
            return idx, {'predict': ''}
        
        results_dict = {}  
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_single_cot, (idx, item)): idx
                for idx, item in enumerate(inputs)
            }
            
            completed_count = 0
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results_dict[idx] = result
                    completed_count += 1
                    
                    if completed_count % 10 == 0:
                        self.logger.info(f"📊 Progress: {completed_count}/{len(inputs)} corrected CoTs completed")
                        
                except Exception as e:
                    self.logger.error(f"❌ Task failed: {e}")
        
        results = [results_dict[i] for i in range(len(inputs))]
        
        with open(batch_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ API concurrent generation completed: {len(results)}/{len(inputs)}")
        return True
    
    def _generate_corrected_cots_for_wrong_problems(
        self,
        wrong_problems: List[Dict[str, Any]],
        round_dir: Path
    ) -> Dict[str, str]:
        """
        Batch generate corrected CoT for wrong problems
        
        Args:
            wrong_problems: wrong problems list
            round_dir: round directory
            
        Returns:
            dictionary: problem_id -> corrected_cot
        """
        if not self.llm_generator:
            self.logger.warning("LLM Generator not available, skipping corrected CoT generation")
            return {}
        
        total = len(wrong_problems)
        self.logger.info(f"Generating corrected CoT for {total} wrong problems...")
        
        batch_inputs, problem_ids = self._prepare_batch_inputs(wrong_problems)
        
        valid_inputs = [(idx, inp) for idx, inp in enumerate(batch_inputs) if inp is not None]
        if not valid_inputs:
            self.logger.warning("No valid inputs for corrected CoT generation")
            return {}
        
        sft_dir = round_dir / "sft"
        temp_dir = sft_dir / "sft_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        batch_input_file = temp_dir / "batch_reflection_input.json"
        batch_output_file = temp_dir / "batch_reflection_output.json"
        
        valid_inputs_list = [inp for _, inp in valid_inputs]
        with open(batch_input_file, 'w', encoding='utf-8') as f:
            json.dump(valid_inputs_list, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"🔄 Generating corrected CoT for {len(valid_inputs_list)} problems (batch mode)...")
        self.logger.info(f"⚠️  Using API for corrected CoT generation (not local model)")
        import time
        start_time = time.time()
        
        corrected_cot_max_tokens = self.config.corrected_cot_max_tokens
        self.logger.info(f"Using max_tokens={corrected_cot_max_tokens} for corrected CoT generation")

        self.logger.info("🌐 Using API to generate corrected CoT...")
        success = self._generate_corrected_cots_with_api(
            batch_input_file=batch_input_file,
            batch_output_file=batch_output_file,
            max_tokens=corrected_cot_max_tokens
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"⏱️  Batch corrected CoT generation took {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        if not success or not batch_output_file.exists():
            self.logger.warning("Failed to generate corrected CoT batch")
            return {}
        
        with open(batch_output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        temp_corrected_cots = {}
        initial_skipped = 0
        failed_reasons = {}  
        
        for i, (original_idx, _) in enumerate(valid_inputs):
            if i >= len(results):
                problem_id = problem_ids[original_idx]
                self.logger.warning(f"No result for problem {problem_id}")
                failed_reasons[problem_id] = "No result from API"
                initial_skipped += 1
                continue
            
            corrected_cot = results[i].get('predict', '').strip()
            problem_id = problem_ids[original_idx]
            
            if not corrected_cot:
                self.logger.warning(f"Empty corrected CoT for problem {problem_id} (all retries failed)")
                failed_reasons[problem_id] = "Empty corrected CoT (retries exhausted)"
                initial_skipped += 1
                continue
            
            
            problem = wrong_problems[original_idx]
            wrong_answer = problem.get('model_prediction', '')
            
            is_valid, skip_reason = self._validate_corrected_cot_format(
                corrected_cot, problem_id, wrong_answer
            )
            
            if not is_valid:
                self.logger.warning(f"Problem {problem_id}: {skip_reason}, skipping")
                failed_reasons[problem_id] = skip_reason
                initial_skipped += 1
                continue
            
            temp_corrected_cots[problem_id] = corrected_cot
        
        self.logger.info(f"Format validation: {len(temp_corrected_cots)}/{len(valid_inputs)} passed")
        if initial_skipped > 0:
            self.logger.warning(f"⚠️  {initial_skipped} problems failed format validation")
            reason_counts = {}
            for problem_id, reason in failed_reasons.items():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            self.logger.warning("📋 Format validation failure summary:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                self.logger.warning(f"   - {reason}: {count} problem(s)")
            
            if len(failed_reasons) <= 10:  
                self.logger.warning("   Details:")
                for problem_id, reason in failed_reasons.items():
                    self.logger.warning(f"     • Problem {problem_id}: {reason}")
        
        if not temp_corrected_cots:
            return {}
        
        self.logger.info("Starting LLM-as-Judge validation for corrected CoTs...")
        corrected_cots = self._validate_corrected_cots_with_judge(
            temp_corrected_cots, wrong_problems, problem_ids, valid_inputs, round_dir
        )
        
        judge_skipped = len(temp_corrected_cots) - len(corrected_cots)
        self.logger.info(f"✅ Successfully validated corrected CoT for {len(corrected_cots)}/{len(valid_inputs)} problems")
        self.logger.info(f"   Format validation passed: {len(temp_corrected_cots)}")
        self.logger.info(f"   Judge validation passed: {len(corrected_cots)}")
        if judge_skipped > 0:
            self.logger.warning(f"⚠️  {judge_skipped} problems failed judge validation (wrong answers)")
            failed_judge_ids = set(temp_corrected_cots.keys()) - set(corrected_cots.keys())
            if failed_judge_ids and len(failed_judge_ids) <= 10:
                self.logger.warning("   Failed judge validation:")
                for problem_id in sorted(failed_judge_ids):
                    self.logger.warning(f"     • Problem {problem_id}: Answer is incorrect")
        
        return corrected_cots
    
    def _run_sft_training(
        self,
        round_num: int,
        wrong_problems: List[Dict[str, Any]],
        round_dir: Path,
        corrected_cots: Optional[Dict[str, str]] = None
    ) -> bool:
        """Run SFT training (cumulative training mode: train the initial model with all historical data in each round)"""
        sft_dir = round_dir / "sft"
        sft_dir.mkdir(parents=True, exist_ok=True)
        
        sft_data_file = sft_dir / "sft_data.json"
        num_sft_points, num_wrong = self.sft_builder.build_sft_dataset(
            wrong_problems, 
            sft_data_file,
            corrected_cots=corrected_cots
        )
        
        if num_sft_points == 0:
            self.logger.warning("No SFT data generated, skipping SFT training")
            return False
        
        self.logger.info(f"Created {num_sft_points} SFT data points from {num_wrong} wrong problems")
        
        self.sft_data_files.append(sft_data_file)
        self.logger.info(f"📊 Accumulated SFT data files: {len(self.sft_data_files)} rounds")
        
        if len(self.sft_data_files) > 1:
            self.logger.info(f"🔄 Merging SFT data from {len(self.sft_data_files)} rounds for cumulative training...")
            merged_data = []
            total_points = 0
            
            for i, data_file in enumerate(self.sft_data_files, 1):
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        round_data = json.load(f)
                    merged_data.extend(round_data)
                    total_points += len(round_data)
                    self.logger.info(f"   Round {i}: {len(round_data)} data points")
                except Exception as e:
                    self.logger.error(f"Failed to load SFT data from {data_file}: {e}")
                    return False
            
            filtered_data = []
            removed_count = 0
            for item in merged_data:
                image_token_count = 0
                for conv in item.get('conversations', []):
                    if conv.get('from') == 'user':
                        image_token_count = conv.get('value', '').count('<image>')
                
                images_count = len(item.get('images', []))
                
                if image_token_count == images_count:
                    filtered_data.append(item)
                else:
                    removed_count += 1
            
            self.logger.info(f"🔍 Filtering out data with mismatched images: removed {removed_count} items, kept {len(filtered_data)} items")
            
            merged_data_file = sft_dir / "sft_data_merged.json"
            with open(merged_data_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Merged {len(filtered_data)} total SFT data points from {len(self.sft_data_files)} rounds (filtered {removed_count} mismatched)")
            training_data_file = merged_data_file
        else:
            self.logger.info("📝 Round 1: Using single round data (no merging needed)")
            training_data_file = sft_data_file
        
        wait_time = self.config.dpo_memory_wait_time
        self.logger.info(f"Waiting {wait_time} seconds for GPU memory to fully release before SFT training...")
        time.sleep(wait_time)
        
        self.logger.info("Ready to start SFT training.")
        
        dataset_name = f"sft_round_{round_num}_cumulative"
        model_path = Path(self.initial_model_path)  
        output_dir = sft_dir / f"sft_model_round{round_num}"
        
        self.logger.info(f"🎯 Training strategy: Cumulative training")
        self.logger.info(f"   - Base model: {self.initial_model_path}")
        self.logger.info(f"   - Training data: {len(self.sft_data_files)} rounds merged")
        self.logger.info(f"   - Output model: {output_dir}")
        
        success = self.sft_trainer.train(
            dataset_name=dataset_name,
            sft_data_file=training_data_file,
            model_path=model_path,
            output_dir=output_dir,
            round_num=round_num,
            learning_rate=self._get_sft_learning_rate_for_round(round_num),
        )
        
        if not success:
            return False

        trained_model_path = output_dir
        self.logger.info(f"✅ SFT training completed. Model saved to: {trained_model_path}")
        
        return True
    
    def _upgrade_difficulty(
        self,
        round_num: int,
        problems: List[Dict[str, Any]],
        round_dir: Path,
        upgrade_mode: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Upgrade difficulty (parallel processing)
        
        Args:
            round_num: round number
            problems: problems to upgrade
            round_dir: round directory
            upgrade_mode: upgrade mode
                - "both": generate reasoning and visual two variants (default, for correct problems)
                - "error_based": upgrade based on error type (for wrong problems)
        
        Returns:
            upgraded problems list
        """
        upgraded_dir = round_dir / "upgraded"
        upgraded_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Upgrading difficulty for {len(problems)} problems (mode: {upgrade_mode})...")
        self.logger.info(f"Using concurrent generation with max_workers={min(100, len(problems))}")
        
        upgraded_problems = []
        
        def upgrade_single_problem(idx_and_problem):
            """Upgrade single problem"""
            idx, problem = idx_and_problem
            problem_id = problem.get('id', '')
            error_type = problem.get('error_type', None)  
            
            self.logger.info(f"🤖 [Thread-{idx+1}] Upgrading problem {idx+1}/{len(problems)}: ID {problem_id}")
            
            problem_text = problem.get("problem", "")
            answer_text = problem.get("reference_answer", "") or problem.get("answer", "")
            category_id = problem.get("category", 0)
            category_name = problem.get("category_name", "")
            image_path = problem.get("image_path")
            if isinstance(image_path, list) and image_path:
                image_path = image_path[0]
            elif not image_path:
                image_path = ""
            
            results = []
            current_reasoning_level = self._normalize_reasoning_level(problem.get("reasoning_level"))
            current_visual_level = self._normalize_visual_level(problem.get("visual_level"))
            
            if upgrade_mode == "error_based" and error_type:
                if error_type == "caption":
                    difficulty_aspects = ["visual", "similar"]
                    self.logger.info(f"  [Thread-{idx+1}] Caption error detected, upgrading VISUAL difficulty + generating SIMILAR problem for diversity")
                elif error_type == "reasoning":
                    difficulty_aspects = ["reasoning", "similar"]
                    self.logger.info(f"  [Thread-{idx+1}] Reasoning error detected, upgrading REASONING difficulty + generating SIMILAR problem for diversity")
                else:
                    difficulty_aspects = ["reasoning", "visual", "similar"]
                    self.logger.warning(f"  [Thread-{idx+1}] Unknown error type '{error_type}', upgrading reasoning + visual + similar (comprehensive improvement)")
            else:
                difficulty_aspects = ["reasoning", "visual"]
            
            for difficulty_aspect in difficulty_aspects:
                try:
                    target_reasoning_level, target_visual_level = self._compute_target_levels(
                        difficulty_aspect,
                        current_reasoning_level,
                        current_visual_level
                    )
                    self.logger.debug(
                        "  [Thread-%d] Levels for %s (aspect=%s): reasoning %.1f→%.1f, visual %d→%d",
                        idx + 1,
                        problem_id,
                        difficulty_aspect,
                        current_reasoning_level,
                        target_reasoning_level,
                        current_visual_level,
                        target_visual_level,
                    )
                    upgraded = self.gemini_generator.upgrade_problem_difficulty(
                        problem=problem_text,
                        answer=answer_text,
                        image_path=image_path,
                        category_id=category_id,
                        category_name=category_name,
                        difficulty_aspect=difficulty_aspect,
                        current_reasoning_level=current_reasoning_level,
                        current_visual_level=current_visual_level,
                        target_reasoning_level=target_reasoning_level,
                        target_visual_level=target_visual_level,
                    )
                    
                    if upgraded:
                        generated_id = self.next_generated_id
                        self.next_generated_id += 1
                        
                        new_problem_text = upgraded.get("question", "").strip()
                        if not new_problem_text.startswith("<image>"):
                            new_problem_text = "<image> " + new_problem_text
                        
                        new_problem = {
                            "id": generated_id,  
                            "problem": new_problem_text,  
                            "answer": upgraded.get("answer", "").strip(),
                            "category": category_id,  
                            "category_name": category_name,
                            "image_path": [],
                            "image_code": upgraded.get("image_code", ""),
                            "difficulty_type": difficulty_aspect,
                            "original_problem_id": problem_id,
                            "reasoning_level": target_reasoning_level,
                            "visual_level": target_visual_level,
                        }
                        
                        if error_type:
                            new_problem["source_error_type"] = error_type
                        
                        results.append(new_problem)
                    else:
                        self.logger.warning(f"  [Thread-{idx+1}] Failed to upgrade {difficulty_aspect} for {problem_id}")
                
                except Exception as e:
                    self.logger.error(f"  [Thread-{idx+1}] Error upgrading {difficulty_aspect} for {problem_id}: {e}")
            
            self.logger.info(f"✅ [Thread-{idx+1}] Generated {len(results)} upgraded problems for {problem_id}")
            return results
        
        if not problems:
            self.logger.warning("No problems to upgrade, returning empty list")
            upgraded_file = upgraded_dir / "upgraded_problems.json"
            with open(upgraded_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        max_workers = min(50, len(problems))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(upgrade_single_problem, (idx, problem)): idx
                for idx, problem in enumerate(problems)
            }
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    upgraded_problems.extend(results)
                except Exception as e:
                    self.logger.error(f"❌ Failed to process problem: {e}")
        
        self.logger.info(f"✅ Successfully generated {len(upgraded_problems)} upgraded problems")
        
        upgraded_file = upgraded_dir / "upgraded_problems.json"
        with open(upgraded_file, 'w', encoding='utf-8') as f:
            json.dump(upgraded_problems, f, ensure_ascii=False, indent=2)
        
        return upgraded_problems
    


    def _materialise_problem_images(
        self,
        round_num: int,
        problems: List[Dict[str, Any]],
        round_dir: Path
    ) -> List[Dict[str, Any]]:
        """Generate images for problems with image_code, up to 3 retries, discard problems if failed"""
        upgraded_dir = round_dir / "upgraded"
        
        problems_with_code = [p for p in problems if p.get("image_code")]
        problems_without_code = [p for p in problems if not p.get("image_code")]
        
        if not problems_with_code:
            self.logger.info("No problems require image generation")
            return problems
        
        self.logger.info(f"Generating images for {len(problems_with_code)} problems...")
        
        image_gen_times = []
        image_gen_start = time.time()
        
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import numpy as np
            
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception as exc:
            self.logger.error(f"matplotlib/numpy not available: {exc}")
            self.logger.warning("Discarding all problems that require images")
            return problems_without_code
        
        image_dir = round_dir / "generated_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        max_retries = 0  
        successful_problems = []
        failed_problems = []
        
        for problem in problems_with_code:
            problem_start_time = time.time()
            problem_id = problem.get("id", "")
            image_code = problem.get("image_code", "")
            
            output_path = image_dir / f"{problem_id}.png"

            existing_paths = problem.get("image_path") or []
            if output_path.exists() and any(str(output_path) == p for p in existing_paths):
                successful_problems.append(problem)
                continue
            
            current_code = image_code
            success = False
            error_history = []
            
            for attempt in range(max_retries + 1):
                namespace = {"plt": plt, "np": np}
                original_savefig = plt.savefig
                original_fig_savefig = Figure.savefig
                
                def plt_savefig_override(path, *args, **kwargs):
                    return original_savefig(output_path, *args, **kwargs)
                
                def fig_savefig_override(self, path, *args, **kwargs):
                    return original_fig_savefig(self, output_path, *args, **kwargs)
                
                plt.savefig = plt_savefig_override
                Figure.savefig = fig_savefig_override
                
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                    
                    exec(current_code, namespace)
                    
                    signal.alarm(0)
                    
                    if output_path.exists():
                        problem["image_path"] = [str(output_path)]
                        self.logger.debug(f"✓ Generated image for {problem_id}")
                        success = True
                    else:
                        error_msg = "Code executed but no image file created"
                        error_history.append({
                            "attempt": attempt + 1,
                            "error": error_msg,
                            "code": current_code  
                        })
                        self.logger.warning(f"✗ Image generation failed for {problem_id} (attempt {attempt+1}): {error_msg}")
                
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {str(exc)}"
                    error_history.append({
                        "attempt": attempt + 1,
                        "error": error_message,
                        "code": current_code  
                    })
                    self.logger.warning(f"✗ Image generation failed for {problem_id} (attempt {attempt+1}): {error_message}")
                    
                    if output_path.exists():
                        output_path.unlink(missing_ok=True)
                    
                    if attempt < max_retries:
                        self.logger.info(f"  Asking Gemini to fix code for {problem_id}...")
                        try:
                            fixed_code = self.gemini_generator.fix_image_code(
                                question=problem.get("problem", ""),
                                answer=problem.get("answer", ""),
                                original_code=current_code,
                                error_message=error_message,
                            )
                            
                            if fixed_code:
                                self.logger.info(f"  Got fixed code from Gemini, retrying...")
                                current_code = fixed_code
                                problem["image_code"] = fixed_code  
                            else:
                                self.logger.warning(f"  Gemini could not fix code, giving up")
                                break
                        except Exception as fix_exc:
                            self.logger.error(f"  Error calling Gemini to fix code: {fix_exc}")
                            break
                
                finally:
                    signal.alarm(0)
                    plt.savefig = original_savefig
                    Figure.savefig = original_fig_savefig
                    plt.close("all")
                
                if success:
                    break
            
            if success:
                problem_elapsed = time.time() - problem_start_time
                image_gen_times.append(problem_elapsed)
                successful_problems.append(problem)
            else:
                self.logger.error(f"✗ All attempts failed for {problem_id}, discarding this problem")
                failed_problems.append({
                    "id": problem_id,  
                    "category": problem.get("category"),
                    "category_name": problem.get("category_name"),
                    "problem": problem.get("problem"),
                    "answer": problem.get("answer"),
                    "original_problem_id": problem.get("original_problem_id"),
                    "difficulty_type": problem.get("difficulty_type"),
                    "original_image_code": image_code,  
                    "error_history": error_history,
                    "total_attempts": len(error_history)
                })
        
        if failed_problems:
            failed_file = upgraded_dir / "failed_image_generation.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_problems, f, ensure_ascii=False, indent=2)
            self.logger.warning(f"Saved {len(failed_problems)} failed image generation records to {failed_file}")
        
        all_successful = successful_problems + problems_without_code
        
        image_gen_total_time = time.time() - image_gen_start
        
        self.logger.info(f"Image generation completed:")
        self.logger.info(f"  Success: {len(successful_problems)}")
        self.logger.info(f"  Failed (discarded): {len(failed_problems)}")
        self.logger.info(f"  No image required: {len(problems_without_code)}")
        self.logger.info(f"  Total remaining: {len(all_successful)}")
        
        if image_gen_times:
            avg_time = sum(image_gen_times) / len(image_gen_times)
            max_time = max(image_gen_times)
            min_time = min(image_gen_times)
            self.logger.info(f"⏱️  Image generation stats:")
            self.logger.info(f"    Total time: {image_gen_total_time:.2f}s ({image_gen_total_time/60:.2f}m)")
            self.logger.info(f"    Avg per image: {avg_time:.2f}s")
            self.logger.info(f"    Min/Max: {min_time:.2f}s / {max_time:.2f}s")
        
        return all_successful
    
    def _save_round_summary(self, round_num: int) -> None:
        """Save round summary"""
        round_dir = self.config.output_dir / f"round_{round_num}"
        summary_file = round_dir / "round_summary.json"
        
        summary = {
            "round": round_num,
            "model_path": self.current_model_path,
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.round_history.append(summary)

