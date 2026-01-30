问题验证器：使用指定模型验证新题的答案是否正确

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProblemValidator:

    def __init__(self, config, llm_generator, llm_judge):
        初始化问题验证器

        Args:
            config: 配置对象
            llm_generator: LLM Generator 实例
            llm_judge: LLM Judge 实例（可选）
        self.config = config
        self.llm_generator = llm_generator
        self.llm_judge = llm_judge
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_new_problems(
        self,
        round_num: int,
        problems: List[Dict[str, Any]],
        round_dir: Path
    ) -> List[Dict[str, Any]]:
        使用指定模型验证新题的答案是否正确

        将题目和 Gemini 答案一起提供给模型，让模型判断答案是否正确

        Args:
            round_num: 轮次编号
            problems: 待验证的新题列表
            round_dir: 轮次目录

        Returns:
            验证通过的题目列表

        validation_model_path = self.config.llm_generator_model_path
        if not problems:
            self.logger.info("No problems to validate")
            return []

        if not self.llm_generator:
            self.logger.warning("LLM Generator not available, skipping validation")
            return problems

        validation_dir = round_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Validating {len(problems)} new problems with model: {validation_model_path}")


        validation_inputs = []
        problem_ids = []

        for problem in problems:
            problem_id = problem.get('id')
            if problem_id is None:
                problem_id = hash(problem.get('problem', '')) % 100000
            problem_ids.append(problem_id)

            problem_text = problem.get('problem', '')
            gemini_answer = problem.get('answer', '').strip()


            image_path_raw = problem.get('image_path', '')
            if isinstance(image_path_raw, list):
                image_paths = image_path_raw
            elif isinstance(image_path_raw, str) and image_path_raw:
                image_paths = [image_path_raw]
            else:
                image_paths = []


            validation_prompt = self._create_validation_prompt(problem_text, gemini_answer)

            validation_inputs.append({
                "id": problem_id,
                "problem": validation_prompt,
                "image_path": image_paths
            })


        validation_input_file = validation_dir / "validation_input.json"
        with open(validation_input_file, 'w', encoding='utf-8') as f:
            json.dump(validation_inputs, f, ensure_ascii=False, indent=2)


        validation_output_file = validation_dir / "validation_output.json"
        self.logger.info("Asking model to judge if the answer is correct...")
        success = self.llm_generator.generate_predictions(
            test_file=validation_input_file,
            output_file=validation_output_file,
            model_path=validation_model_path,
            max_tokens=30000,
            disable_thinking=False,
            is_validation=True
        )

        if not success or not validation_output_file.exists():
            self.logger.error("Failed to generate validation judgments, keeping all problems")
            return problems


        with open(validation_output_file, 'r', encoding='utf-8') as f:
            validation_results = json.load(f)


        validated_problems = []
        for i, problem in enumerate(problems):
            problem_id = problem_ids[i]
            judgment_text = ""


            for result in validation_results:
                if result.get('id') == problem_id:
                    judgment_text = result.get('predict', '').strip()
                    break

            if not judgment_text:
                self.logger.warning(f"No judgment found for problem {problem_id}, will be discarded")
                continue


            is_correct = self._parse_judgment(judgment_text)

            if is_correct:
                validated_problems.append(problem)
                self.logger.debug(f"Problem {problem_id} passed validation (answer is correct)")
            else:
                self.logger.debug(f"Problem {problem_id} failed validation (answer is incorrect)")

        return validated_problems

    def _create_validation_prompt(self, problem_text: str, answer: str) -> str:
        创建验证 prompt：让模型判断给定的答案是否正确

        注意：输出要求已经在 System prompt 中定义，这里只提供题目和答案

        Args:
            problem_text: 题目文本
            answer: Gemini 生成的答案

        Returns:
            验证 prompt（只包含题目和答案，不包含重复的输出要求）
        prompt = f"""## PROBLEM
{problem_text}

## GIVEN ANSWER
{answer}

Analyze the problem and the given answer step by step. Determine if the given answer is mathematically correct and solves the problem correctly. After your analysis, end your response with exactly one of these tags: <correct> or <incorrect>."""
        return prompt

    def _parse_judgment(self, judgment_text: str) -> bool:
        解析判断结果，提取 <correct> 或 <incorrect> 标签

        Args:
            judgment_text: 模型生成的判断文本（可能包含 thinking 内容）

        Returns:
            True 如果判断为正确，False 如果判断为错误
        import re


        text_to_parse = judgment_text
        if text_to_parse.startswith("<think>"):
            text_to_parse = text_to_parse[len("<think>"):].strip()


        pattern_redacted = r'<think>(.*?)</think>'
        matches_redacted = re.findall(pattern_redacted, text_to_parse, re.DOTALL)

        if matches_redacted:

            text_without_tags = re.sub(pattern_redacted, '', text_to_parse, flags=re.DOTALL).strip()
            if text_without_tags:

                text_to_parse = text_without_tags
            else:

                text_to_parse = matches_redacted[-1] if matches_redacted else text_to_parse



        correct_tag_pattern = r'<correct>'
        incorrect_tag_pattern = r'<incorrect>'

        correct_tag_matches = list(re.finditer(correct_tag_pattern, text_to_parse, re.IGNORECASE))
        incorrect_tag_matches = list(re.finditer(incorrect_tag_pattern, text_to_parse, re.IGNORECASE))

        if correct_tag_matches or incorrect_tag_matches:

            last_correct_pos = max([m.start() for m in correct_tag_matches]) if correct_tag_matches else -1
            last_incorrect_pos = max([m.start() for m in incorrect_tag_matches]) if incorrect_tag_matches else -1


            if last_correct_pos > last_incorrect_pos:
                return True
            elif last_incorrect_pos > last_correct_pos:
                return False
            elif last_correct_pos != -1:
                return True
            elif last_incorrect_pos != -1:
                return False


        judgment_text_lower = text_to_parse.lower().strip()

        correct_positions = []
        incorrect_positions = []


        for match in re.finditer(r'\bcorrect\b', judgment_text_lower):
            correct_positions.append(match.start())


        for match in re.finditer(r'\bincorrect\b', judgment_text_lower):
            incorrect_positions.append(match.start())


        last_correct_pos = max(correct_positions) if correct_positions else -1
        last_incorrect_pos = max(incorrect_positions) if incorrect_positions else -1


        if last_correct_pos > last_incorrect_pos:
            return True
        elif last_incorrect_pos > last_correct_pos:
            return False
        elif last_correct_pos != -1:
            return True
        elif last_incorrect_pos != -1:
            return False


        self.logger.warning(f"Could not parse judgment (no <correct> or <incorrect> tag found): {judgment_text[-200:] if len(judgment_text) > 200 else judgment_text}")
        return False
