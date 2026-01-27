"""
问题验证器：使用指定模型验证新题的答案是否正确
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProblemValidator:
    """问题验证器类"""
    
    def __init__(self, config, llm_generator, llm_judge):
        """
        初始化问题验证器
        
        Args:
            config: 配置对象
            llm_generator: LLM Generator 实例
            llm_judge: LLM Judge 实例（可选）
        """
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
        """
        使用指定模型验证新题的答案是否正确
        
        将题目和 Gemini 答案一起提供给模型，让模型判断答案是否正确
        
        Args:
            round_num: 轮次编号
            problems: 待验证的新题列表
            round_dir: 轮次目录
            
        Returns:
            验证通过的题目列表
        """
        # 从配置中获取验证模型路径
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
        
        # 1. 准备验证输入数据（将题目和 Gemini 答案一起提供给模型）
        validation_inputs = []
        problem_ids = []
        
        for problem in problems:
            problem_id = problem.get('id')
            if problem_id is None:
                problem_id = hash(problem.get('problem', '')) % 100000
            problem_ids.append(problem_id)
            
            problem_text = problem.get('problem', '')
            gemini_answer = problem.get('answer', '').strip()
            
            # image_path 可能是字符串或列表，统一转换为列表
            image_path_raw = problem.get('image_path', '')
            if isinstance(image_path_raw, list):
                image_paths = image_path_raw
            elif isinstance(image_path_raw, str) and image_path_raw:
                image_paths = [image_path_raw]
            else:
                image_paths = []
            
            # 构建验证 prompt：让模型判断给定的答案是否正确
            validation_prompt = self._create_validation_prompt(problem_text, gemini_answer)
            
            validation_inputs.append({
                "id": problem_id,
                "problem": validation_prompt,
                "image_path": image_paths
            })
        
        # 2. 保存验证输入文件
        validation_input_file = validation_dir / "validation_input.json"
        with open(validation_input_file, 'w', encoding='utf-8') as f:
            json.dump(validation_inputs, f, ensure_ascii=False, indent=2)
        
        # 3. 使用指定模型生成判断结果
        validation_output_file = validation_dir / "validation_output.json"
        self.logger.info("Asking model to judge if the answer is correct...")
        success = self.llm_generator.generate_predictions(
            test_file=validation_input_file,
            output_file=validation_output_file,
            model_path=validation_model_path,
            max_tokens=30000,  # 允许模型完整思考过程
            disable_thinking=False,  # 允许模型思考，然后从输出中提取最后一个判断词
            is_validation=True  # 使用答案验证配置（与错误分析相同）
        )
        
        if not success or not validation_output_file.exists():
            self.logger.error("Failed to generate validation judgments, keeping all problems")
            return problems
        
        # 4. 读取生成的判断结果
        with open(validation_output_file, 'r', encoding='utf-8') as f:
            validation_results = json.load(f)
        
        # 5. 解析判断结果，筛选验证通过的题目
        validated_problems = []
        for i, problem in enumerate(problems):
            problem_id = problem_ids[i]
            judgment_text = ""
            
            # 找到对应的生成结果
            for result in validation_results:
                if result.get('id') == problem_id:
                    judgment_text = result.get('predict', '').strip()
                    break
            
            if not judgment_text:
                self.logger.warning(f"No judgment found for problem {problem_id}, will be discarded")
                continue
            
            # 解析判断结果
            is_correct = self._parse_judgment(judgment_text)
            
            if is_correct:
                validated_problems.append(problem)
                self.logger.debug(f"Problem {problem_id} passed validation (answer is correct)")
            else:
                self.logger.debug(f"Problem {problem_id} failed validation (answer is incorrect)")
        
        return validated_problems
    
    def _create_validation_prompt(self, problem_text: str, answer: str) -> str:
        """
        创建验证 prompt：让模型判断给定的答案是否正确
        
        注意：输出要求已经在 System prompt 中定义，这里只提供题目和答案
        
        Args:
            problem_text: 题目文本
            answer: Gemini 生成的答案
            
        Returns:
            验证 prompt（只包含题目和答案，不包含重复的输出要求）
        """
        prompt = f"""## PROBLEM
{problem_text}

## GIVEN ANSWER
{answer}

Analyze the problem and the given answer step by step. Determine if the given answer is mathematically correct and solves the problem correctly. After your analysis, end your response with exactly one of these tags: <correct> or <incorrect>."""
        return prompt
    
    def _parse_judgment(self, judgment_text: str) -> bool:
        """
        解析判断结果，提取 <correct> 或 <incorrect> 标签
        
        Args:
            judgment_text: 模型生成的判断文本（可能包含 thinking 内容）
            
        Returns:
            True 如果判断为正确，False 如果判断为错误
        """
        import re
        
        # 处理 <think> 前缀：移除前缀，只保留实际输出
        text_to_parse = judgment_text
        if text_to_parse.startswith("<think>"):
            text_to_parse = text_to_parse[len("<think>"):].strip()
        
        # 处理 <think>...</think> 标签：提取标签外的内容
        pattern_redacted = r'<think>(.*?)</think>'
        matches_redacted = re.findall(pattern_redacted, text_to_parse, re.DOTALL)
        
        if matches_redacted:
            # 如果找到标签，提取标签外的内容（如果有）
            text_without_tags = re.sub(pattern_redacted, '', text_to_parse, flags=re.DOTALL).strip()
            if text_without_tags:
                # 如果标签外有内容，优先使用标签外的内容
                text_to_parse = text_without_tags
            else:
                # 如果整个输出都在标签内，使用标签内的最后一部分
                text_to_parse = matches_redacted[-1] if matches_redacted else text_to_parse
        
        # 🔑 关键：查找 <correct> 或 <incorrect> 标签
        # 优先查找标签格式
        correct_tag_pattern = r'<correct>'
        incorrect_tag_pattern = r'<incorrect>'
        
        correct_tag_matches = list(re.finditer(correct_tag_pattern, text_to_parse, re.IGNORECASE))
        incorrect_tag_matches = list(re.finditer(incorrect_tag_pattern, text_to_parse, re.IGNORECASE))
        
        if correct_tag_matches or incorrect_tag_matches:
            # 找到最后一个标签的位置
            last_correct_pos = max([m.start() for m in correct_tag_matches]) if correct_tag_matches else -1
            last_incorrect_pos = max([m.start() for m in incorrect_tag_matches]) if incorrect_tag_matches else -1
            
            # 比较最后一个标签的位置
            if last_correct_pos > last_incorrect_pos:
                return True
            elif last_incorrect_pos > last_correct_pos:
                return False
            elif last_correct_pos != -1:  # 只有 <correct>，没有 <incorrect>
                return True
            elif last_incorrect_pos != -1:  # 只有 <incorrect>，没有 <correct>
                return False
        
        # 如果没有找到标签，向后兼容：查找最后一个 "correct" 或 "incorrect" 单词
        judgment_text_lower = text_to_parse.lower().strip()
        
        correct_positions = []
        incorrect_positions = []
        
        # 查找所有 "correct" 的位置（使用单词边界，避免匹配到 "incorrect" 中的 "correct"）
        for match in re.finditer(r'\bcorrect\b', judgment_text_lower):
            correct_positions.append(match.start())
        
        # 查找所有 "incorrect" 的位置
        for match in re.finditer(r'\bincorrect\b', judgment_text_lower):
            incorrect_positions.append(match.start())
        
        # 找到最后一个匹配的位置
        last_correct_pos = max(correct_positions) if correct_positions else -1
        last_incorrect_pos = max(incorrect_positions) if incorrect_positions else -1
        
        # 比较最后一个 "correct" 和最后一个 "incorrect" 的位置
        if last_correct_pos > last_incorrect_pos:
            return True
        elif last_incorrect_pos > last_correct_pos:
            return False
        elif last_correct_pos != -1:  # 只有 correct，没有 incorrect
            return True
        elif last_incorrect_pos != -1:  # 只有 incorrect，没有 correct
            return False
        
        # 默认：如果没有明确判断，认为不正确（保守策略）
        self.logger.warning(f"Could not parse judgment (no <correct> or <incorrect> tag found): {judgment_text[-200:] if len(judgment_text) > 200 else judgment_text}")
        return False
    

