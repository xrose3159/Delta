错误分析器 - 使用 LLM 判断错误是 caption 问题还是 reasoning 问题

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ErrorAnalyzer:

    def __init__(self, config, llm_generator):
        Args:
            config: 配置对象
            llm_generator: LLM Generator 实例（用于调用 Qwen3VL30b）
        self.config = config
        self.llm_generator = llm_generator
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_errors_batch(
        self,
        wrong_problems: List[Dict[str, Any]],
        temp_dir: Path
    ) -> List[Tuple[str, str]]:
        批量分析多个错题的错误类型（并行处理，大幅提升速度）

        Args:
            wrong_problems: 错题列表，每个元素包含 problem, model_prediction, reference_answer 等
            temp_dir: 临时目录用于保存中间文件

        Returns:
            list of (错误类型, 分析理由)，顺序与 wrong_problems 一致
        if not wrong_problems:
            return []

        self.logger.info(f"开始批量分析 {len(wrong_problems)} 个错题的错误类型...")


        analysis_data = []
        problem_ids = []

        for problem in wrong_problems:

            caption, reasoning, answer = self._parse_model_output(
                problem.get("model_prediction", "")
            )



            if not caption and not reasoning:
                self.logger.warning(f"问题 {problem.get('id', 'unknown')} 完全无法解析模型输出（caption 和 reasoning 都为空），将默认为 reasoning 问题")
                continue


            if not caption:
                self.logger.debug(f"问题 {problem.get('id', 'unknown')} 缺少 caption 部分")
                caption = "[Caption not found - output may be truncated]"
            if not reasoning:
                self.logger.debug(f"问题 {problem.get('id', 'unknown')} 缺少 reasoning 部分")
                reasoning = "[Reasoning not found - output may be truncated]"
            if not answer:
                self.logger.debug(f"问题 {problem.get('id', 'unknown')} 缺少 answer 部分（可能输出被截断）")
                answer = "[Answer not found - output may be truncated]"


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
            self.logger.warning("没有可分析的问题（所有问题都无法解析模型输出）")
            return [("reasoning", "无法解析模型输出") for _ in wrong_problems]


        batch_input_file = temp_dir / "error_analysis_batch_input.json"
        batch_output_file = temp_dir / "error_analysis_batch_output.json"

        with open(batch_input_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"批量分析输入文件: {batch_input_file}")
        self.logger.info(f"使用模型: {self.config.error_analyzer_model_path}")
        self.logger.info(f"vLLM 将并行处理 {len(analysis_data)} 个请求...")



        success = self.llm_generator.generate_predictions(
            test_file=batch_input_file,
            output_file=batch_output_file,
            model_path=self.config.error_analyzer_model_path,
            max_tokens=8192,
            disable_thinking=False,
            is_error_analysis=True
        )

        if not success or not batch_output_file.exists():
            self.logger.error("批量错误分析失败，所有问题默认为 reasoning 问题")
            return [("reasoning", "LLM 批量分析失败") for _ in wrong_problems]


        try:
            with open(batch_output_file, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)


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

                    self.logger.warning(f"问题 {problem_id} 未找到分析结果，默认为 reasoning")
                    results.append(("reasoning", "无法解析模型输出"))

            self.logger.info(f"✅ 批量分析完成，共处理 {len(results)} 个错题")
            return results

        except Exception as e:
            self.logger.error(f"解析批量分析结果失败: {e}")
            return [("reasoning", f"解析失败: {str(e)}") for _ in wrong_problems]




















































































    def _parse_model_output(self, output: str) -> Tuple[str, str, str]:
        解析模型输出，提取 caption, reasoning, answer 三部分

        支持以下情况：
        1. 完整的标签对：<caption>...</caption>
        2. 不完整的标签（输出被截断）：<caption>...（没有结束标签）

        Returns:
            (caption, reasoning, answer)
        caption = ""
        reasoning = ""
        answer = ""

        try:

            content = output


            caption_match = re.search(r'<caption>(.*?)</caption>', content, re.DOTALL | re.IGNORECASE)
            if not caption_match:

                caption_match = re.search(r'<caption>(.*?)(?=<(?:reasoning|answer)|$)', content, re.DOTALL | re.IGNORECASE)

            if caption_match:
                caption = caption_match.group(1).strip()


            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:

                reasoning_match = re.search(r'<reasoning>(.*?)(?=<answer|$)', content, re.DOTALL | re.IGNORECASE)

            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()


            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if not answer_match:

                answer_match = re.search(r'<answer>(.*?)$', content, re.DOTALL | re.IGNORECASE)

            if answer_match:
                answer = answer_match.group(1).strip()
            else:

                boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
                if boxed_match:
                    answer = boxed_match.group(0)

        except Exception as e:
            self.logger.error(f"解析模型输出失败: {e}")

        return caption, reasoning, answer

    def _build_analysis_prompt(
        self,
        problem: str,
        model_caption: str,
        model_reasoning: str,
        model_answer: str,
        correct_answer: str
    ) -> str:

        correct_caption, correct_reasoning, correct_final_answer = self._parse_model_output(correct_answer)


        if not correct_caption and not correct_reasoning and not correct_final_answer:
            correct_caption = "[Unable to parse]"
            correct_reasoning = correct_answer
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
        解析 LLM 的分析结果

        Returns:
            (error_type, reason)
            error_type: "caption", "reasoning", 或 "unknown"

        if "<caption_error>" in analysis_text.lower():
            error_type = "caption"
        elif "<reasoning_error>" in analysis_text.lower():
            error_type = "reasoning"
        else:

            text_lower = analysis_text.lower()
            if "caption" in text_lower and "error" in text_lower:
                error_type = "caption"
            elif "reasoning" in text_lower and "error" in text_lower:
                error_type = "reasoning"
            else:
                error_type = "unknown"
                self.logger.warning("无法从分析结果中识别错误类型，默认为 unknown")


        reason = re.sub(r'<(caption_error|reasoning_error)>', '', analysis_text, flags=re.IGNORECASE).strip()

        return error_type, reason
