"""
Error analyzer - use LLM to determine if the error is a caption problem or a reasoning problem
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Use LLM to analyze error types"""
    
    def __init__(self, config, llm_generator):
        """
        Args:
            config: configuration object
            llm_generator: LLM Generator instance (for calling Qwen3VL30b)
        """
        self.config = config
        self.llm_generator = llm_generator
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_errors_batch(
        self,
        wrong_problems: List[Dict[str, Any]],
        temp_dir: Path
    ) -> List[Tuple[str, str]]:
        """
        Batch analyze multiple wrong problems' error types (parallel processing,大幅提升速度)
        
        Args:
            wrong_problems: wrong problems list, each element contains problem, model_prediction, reference_answer etc.
            temp_dir: temporary directory for saving intermediate files
            
        Returns:
            list of (error type, analysis reason), in the same order as wrong_problems
        """
        if not wrong_problems:
            return []
        
        self.logger.info(f"Start batch analyzing {len(wrong_problems)} wrong problems' error types...")
        
        # 1. collect all problems to analyze
        analysis_data = []
        problem_ids = []
        
        for problem in wrong_problems:
            # parse model output
            caption, reasoning, answer = self._parse_model_output(
                problem.get("model_prediction", "")
            )
            
            # more lenient check: if there is caption or reasoning, try to analyze
            # even if the output is truncated (missing answer tag), still can determine the error type based on caption and reasoning
            if not caption and not reasoning:
                self.logger.warning(f"Problem {problem.get('id', 'unknown')} completely cannot parse model output (caption and reasoning are both empty), will be treated as reasoning problem")
                continue
            
            # if some parts are missing, give a hint
            if not caption:
                self.logger.debug(f"Problem {problem.get('id', 'unknown')} missing caption part")
                caption = "[Caption not found - output may be truncated]"
            if not reasoning:
                self.logger.debug(f"Problem {problem.get('id', 'unknown')} missing reasoning part")
                reasoning = "[Reasoning not found - output may be truncated]"
            if not answer:
                self.logger.debug(f"Problem {problem.get('id', 'unknown')} missing answer part (output may be truncated)")
                answer = "[Answer not found - output may be truncated]"
            
            # build analysis prompt
            analysis_prompt = self._build_analysis_prompt(
                problem=problem.get("problem", ""),
                model_caption=caption,
                model_reasoning=reasoning,
                model_answer=answer,
                correct_answer=problem.get("reference_answer", "")
            )
            
            analysis_data.append({
                "id": problem.get("id", f"unknown_{len(analysis_data)}"),
                "problem": analysis_prompt,
                "image_path": problem.get("image_path", [])
            })
            problem_ids.append(problem.get("id", "unknown"))
        
        if not analysis_data:
            self.logger.warning("No problems to analyze (all problems cannot parse model output)")
            return [("reasoning", "Cannot parse model output") for _ in wrong_problems]
        
        # 2. save batch input file
        batch_input_file = temp_dir / "error_analysis_batch_input.json"
        batch_output_file = temp_dir / "error_analysis_batch_output.json"
        
        with open(batch_input_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Batch analysis input file: {batch_input_file}")
        self.logger.info(f"Using model: {self.config.error_analyzer_model_path}")
        self.logger.info(f"vLLM will process {len(analysis_data)} requests in parallel...")
        
        success = self.llm_generator.generate_predictions(
            test_file=batch_input_file,
            output_file=batch_output_file,
            model_path=self.config.error_analyzer_model_path,
            max_tokens=8192,  # reduce token number, error analysis does not need too long output
            disable_thinking=False,  # keep thinking mode, allow model to analyze in detail
            is_error_analysis=True 
        )
        
        if not success or not batch_output_file.exists():
            self.logger.error("Batch error analysis failed, all problems will be treated as reasoning problems")
            return [("reasoning", "LLM batch analysis failed") for _ in wrong_problems]
        
        # 4. parse batch results
        try:
            with open(batch_output_file, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
            
            # build ID to result mapping
            id_to_result = {}
            for result in batch_results:
                result_id = result.get("id", "unknown")
                analysis_text = result.get("predict", "")
                error_type, reason = self._parse_analysis_result(analysis_text)
                id_to_result[result_id] = (error_type, reason)
            

            results = []
            for i, problem in enumerate(wrong_problems):
                problem_id = problem.get("id", "unknown")
                
                if problem_id in id_to_result:
                    error_type, reason = id_to_result[problem_id]
                    self.logger.info(f"问题 {problem_id}: {error_type} - {reason[:80]}...")
                    results.append((error_type, reason))
                else:
                    # cannot parse problem, will be treated as reasoning problem
                    self.logger.warning(f"Problem {problem_id} not found analysis result, will be treated as reasoning problem")
                    results.append(("reasoning", "Cannot parse model output"))
            
            self.logger.info(f"✅ Batch analysis completed, processed {len(results)} problems")
            return results
            
        except Exception as e:
            self.logger.error(f"Parse batch analysis results failed: {e}")
            return [("reasoning", f"Parse failed: {str(e)}") for _ in wrong_problems]
    
    def _parse_model_output(self, output: str) -> Tuple[str, str, str]:

        caption = ""
        reasoning = ""
        answer = ""
        
        try:
            # directly use output content
            content = output
            
            # try to extract caption (prefer matching complete tag pairs, otherwise match to next tag or text end)
            caption_match = re.search(r'<caption>(.*?)</caption>', content, re.DOTALL | re.IGNORECASE)
            if not caption_match:
                # if there is no end tag, try to match to next tag or text end
                caption_match = re.search(r'<caption>(.*?)(?=<(?:reasoning|answer)|$)', content, re.DOTALL | re.IGNORECASE)
            
            if caption_match:
                caption = caption_match.group(1).strip()
            
            # try to extract reasoning (also support incomplete tags)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:
                # if there is no end tag, try to match to next tag or text end
                reasoning_match = re.search(r'<reasoning>(.*?)(?=<answer|$)', content, re.DOTALL | re.IGNORECASE)
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # try to extract answer (also support incomplete tags)
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if not answer_match:
                # if there is no end tag, try to match to text end
                answer_match = re.search(r'<answer>(.*?)$', content, re.DOTALL | re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # if there is no <answer> tag, try to extract \boxed{} from reasoning as answer
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
                if boxed_match:
                    answer = boxed_match.group(0)  # keep \boxed{} format
                
        except Exception as e:
            self.logger.error(f"Parse model output failed: {e}")
        
        return caption, reasoning, answer
    
    def _build_analysis_prompt(
        self,
        problem: str,
        model_caption: str,
        model_reasoning: str,
        model_answer: str,
        correct_answer: str
    ) -> str:
        """Build error analysis prompt (simplified version, task description is handled by system prompt)"""
        # parse correct_answer (also three-part format)
        correct_caption, correct_reasoning, correct_final_answer = self._parse_model_output(correct_answer)
        
        # if parse failed, use original text
        if not correct_caption and not correct_reasoning and not correct_final_answer:
            correct_caption = "[Unable to parse]"
            correct_reasoning = correct_answer  # put the whole answer into reasoning part
            correct_final_answer = "[Unable to parse]"
        
        return f"""**Problem:**
{problem}

**Student's Caption (Image Description):**
{model_caption}

**Correct Caption (Reference Image Description):**
{correct_caption}

**Student's Reasoning:**
{model_reasoning}

**Correct Reasoning (Reference Solution):**
{correct_reasoning}

**Student's Final Answer:**
{model_answer}

**Correct Final Answer:**
{correct_final_answer}

---

Please analyze the student's solution and determine whether the error is primarily a caption error or a reasoning error. Remember to end your response with <caption_error> or <reasoning_error>."""
    
    def _parse_analysis_result(self, analysis_text: str) -> Tuple[str, str]:
        """
        Parse LLM's analysis result
        
        Returns:
            (error_type, reason)
            error_type: "caption", "reasoning", or "unknown"
        """
        # extract tags
        if "<caption_error>" in analysis_text.lower():
            error_type = "caption"
        elif "<reasoning_error>" in analysis_text.lower():
            error_type = "reasoning"
        else:
            # no tags found, try to infer from text
            text_lower = analysis_text.lower()
            if "caption" in text_lower and "error" in text_lower:
                error_type = "caption"
            elif "reasoning" in text_lower and "error" in text_lower:
                error_type = "reasoning"
            else:
                error_type = "unknown"
                self.logger.warning("Cannot identify error type from analysis result, default to unknown")
        
        # extract reason (text after removing tags)
        reason = re.sub(r'<(caption_error|reasoning_error)>', '', analysis_text, flags=re.IGNORECASE).strip()
        
        return error_type, reason



