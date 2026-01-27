"""
错误分析器 - 使用 LLM 判断错误是 caption 问题还是 reasoning 问题
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """使用 LLM 分析错误类型"""
    
    def __init__(self, config, llm_generator):
        """
        Args:
            config: 配置对象
            llm_generator: LLM Generator 实例（用于调用 Qwen3VL30b）
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
        批量分析多个错题的错误类型（并行处理，大幅提升速度）
        
        Args:
            wrong_problems: 错题列表，每个元素包含 problem, model_prediction, reference_answer 等
            temp_dir: 临时目录用于保存中间文件
            
        Returns:
            list of (错误类型, 分析理由)，顺序与 wrong_problems 一致
        """
        if not wrong_problems:
            return []
        
        self.logger.info(f"开始批量分析 {len(wrong_problems)} 个错题的错误类型...")
        
        # 1. 收集所有需要分析的问题
        analysis_data = []
        problem_ids = []
        
        for problem in wrong_problems:
            # 解析模型输出
            caption, reasoning, answer = self._parse_model_output(
                problem.get("model_prediction", "")
            )
            
            # 更宽容的检查：只要有 caption 或 reasoning，就尝试分析
            # 即使输出被截断（缺少 answer 标签），仍然可以基于 caption 和 reasoning 判断错误类型
            if not caption and not reasoning:
                self.logger.warning(f"问题 {problem.get('id', 'unknown')} 完全无法解析模型输出（caption 和 reasoning 都为空），将默认为 reasoning 问题")
                continue
            
            # 如果某些部分缺失，给出提示
            if not caption:
                self.logger.debug(f"问题 {problem.get('id', 'unknown')} 缺少 caption 部分")
                caption = "[Caption not found - output may be truncated]"
            if not reasoning:
                self.logger.debug(f"问题 {problem.get('id', 'unknown')} 缺少 reasoning 部分")
                reasoning = "[Reasoning not found - output may be truncated]"
            if not answer:
                self.logger.debug(f"问题 {problem.get('id', 'unknown')} 缺少 answer 部分（可能输出被截断）")
                answer = "[Answer not found - output may be truncated]"
            
            # 构建分析提示
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
        
        # 2. 保存批量输入文件
        batch_input_file = temp_dir / "error_analysis_batch_input.json"
        batch_output_file = temp_dir / "error_analysis_batch_output.json"
        
        with open(batch_input_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"批量分析输入文件: {batch_input_file}")
        self.logger.info(f"使用模型: {self.config.error_analyzer_model_path}")
        self.logger.info(f"vLLM 将并行处理 {len(analysis_data)} 个请求...")
        
        # 3. 一次性调用 vLLM 处理所有请求（并行）
        # 注意：使用专门的错误分析 system prompt，确保模型输出 <caption_error> 或 <reasoning_error> 标签
        success = self.llm_generator.generate_predictions(
            test_file=batch_input_file,
            output_file=batch_output_file,
            model_path=self.config.error_analyzer_model_path,
            max_tokens=8192,  # 降低 token 数，错误分析不需要太长的输出
            disable_thinking=False,  # 保持思考模式，允许模型详细分析
            is_error_analysis=True  # 使用错误分析专用 system prompt
        )
        
        if not success or not batch_output_file.exists():
            self.logger.error("批量错误分析失败，所有问题默认为 reasoning 问题")
            return [("reasoning", "LLM 批量分析失败") for _ in wrong_problems]
        
        # 4. 解析批量结果
        try:
            with open(batch_output_file, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
            
            # 构建 ID 到结果的映射
            id_to_result = {}
            for result in batch_results:
                result_id = result.get("id", "unknown")
                analysis_text = result.get("predict", "")
                error_type, reason = self._parse_analysis_result(analysis_text)
                id_to_result[result_id] = (error_type, reason)
            
            # 按原始顺序返回结果
            results = []
            for i, problem in enumerate(wrong_problems):
                problem_id = problem.get("id", "unknown")
                
                if problem_id in id_to_result:
                    error_type, reason = id_to_result[problem_id]
                    self.logger.info(f"问题 {problem_id}: {error_type} - {reason[:80]}...")
                    results.append((error_type, reason))
                else:
                    # 无法解析的问题，默认为 reasoning
                    self.logger.warning(f"问题 {problem_id} 未找到分析结果，默认为 reasoning")
                    results.append(("reasoning", "无法解析模型输出"))
            
            self.logger.info(f"✅ 批量分析完成，共处理 {len(results)} 个错题")
            return results
            
        except Exception as e:
            self.logger.error(f"解析批量分析结果失败: {e}")
            return [("reasoning", f"解析失败: {str(e)}") for _ in wrong_problems]
    
    # def analyze_error(
    #     self,
    #     problem: Dict[str, Any],
    #     model_output: str,
    #     correct_answer: str,
    #     temp_dir: Path
    # ) -> Tuple[str, str]:
    #     """
    #     分析错误是 caption 问题还是 reasoning 问题
        
    #     Args:
    #         problem: 问题字典，包含 problem, image_path 等
    #         model_output: 模型的完整输出（包含 <caption>, <reasoning>, <answer>）
    #         correct_answer: 正确答案
    #         temp_dir: 临时目录用于保存中间文件
            
    #     Returns:
    #         (错误类型, 分析理由)
    #         错误类型可能是: "caption", "reasoning", 或 "unknown"
    #     """
    #     # 1. 解析模型输出，提取三部分
    #     caption, reasoning, answer = self._parse_model_output(model_output)
        
    #     if not caption or not reasoning or not answer:
    #         self.logger.warning("无法解析模型输出的三部分结构，默认为 reasoning 问题")
    #         return "reasoning", "无法解析模型输出格式"
        
    #     # 2. 构建分析提示
    #     analysis_prompt = self._build_analysis_prompt(
    #         problem=problem.get("problem", ""),
    #         model_caption=caption,
    #         model_reasoning=reasoning,
    #         model_answer=answer,
    #         correct_answer=correct_answer
    #     )
        
    #     # 3. 调用 LLM 进行分析
    #     problem_id = problem.get("id", "unknown")
    #     analysis_input_file = temp_dir / f"error_analysis_input_{problem_id}.json"
    #     analysis_output_file = temp_dir / f"error_analysis_output_{problem_id}.json"
        
    #     # 准备输入数据
    #     analysis_data = [{
    #         "problem": analysis_prompt,
    #         "image_path": problem.get("image_path", [])
    #     }]
        
    #     with open(analysis_input_file, 'w', encoding='utf-8') as f:
    #         json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
    #     # 调用 LLM Generator（使用 Qwen3VL30b 进行错误分析）
    #     self.logger.info(f"分析错误类型，使用模型: {self.config.error_analyzer_model_path}")
    #     success = self.llm_generator.generate_predictions(
    #         test_file=analysis_input_file,
    #         output_file=analysis_output_file,
    #         model_path=self.config.error_analyzer_model_path,
    #         max_tokens=4096,  # 较短的回答
    #         disable_thinking=False,  # 允许思考
    #         is_error_analysis=True  # 使用错误分析专用 system prompt
    #     )
        
    #     if not success or not analysis_output_file.exists():
    #         self.logger.warning("错误分析调用失败，默认为 reasoning 问题")
    #         return "reasoning", "LLM 分析失败"
        
    #     # 4. 解析分析结果
    #     try:
    #         with open(analysis_output_file, 'r', encoding='utf-8') as f:
    #             analysis_results = json.load(f)
            
    #         if not analysis_results or len(analysis_results) == 0:
    #             return "reasoning", "分析结果为空"
            
    #         analysis_text = analysis_results[0].get("predict", "")
    #         error_type, reason = self._parse_analysis_result(analysis_text)
            
    #         self.logger.info(f"问题 {problem_id} 的错误类型: {error_type}, 原因: {reason[:100]}")
    #         return error_type, reason
            
    #     except Exception as e:
    #         self.logger.error(f"解析分析结果失败: {e}")
    #         return "reasoning", f"解析失败: {str(e)}"
    
    def _parse_model_output(self, output: str) -> Tuple[str, str, str]:
        """
        解析模型输出，提取 caption, reasoning, answer 三部分
        
        支持以下情况：
        1. 完整的标签对：<caption>...</caption>
        2. 不完整的标签（输出被截断）：<caption>...（没有结束标签）
        
        Returns:
            (caption, reasoning, answer)
        """
        caption = ""
        reasoning = ""
        answer = ""
        
        try:
            # 直接使用输出内容
            content = output
            
            # 尝试提取 caption（优先匹配完整标签对，否则匹配到下一个标签或文本末尾）
            caption_match = re.search(r'<caption>(.*?)</caption>', content, re.DOTALL | re.IGNORECASE)
            if not caption_match:
                # 如果没有结束标签，尝试匹配到下一个标签或文本末尾
                caption_match = re.search(r'<caption>(.*?)(?=<(?:reasoning|answer)|$)', content, re.DOTALL | re.IGNORECASE)
            
            if caption_match:
                caption = caption_match.group(1).strip()
            
            # 尝试提取 reasoning（同样支持不完整标签）
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL | re.IGNORECASE)
            if not reasoning_match:
                # 如果没有结束标签，尝试匹配到下一个标签或文本末尾
                reasoning_match = re.search(r'<reasoning>(.*?)(?=<answer|$)', content, re.DOTALL | re.IGNORECASE)
            
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # 尝试提取 answer（同样支持不完整标签）
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if not answer_match:
                # 如果没有结束标签，尝试匹配到文本末尾
                answer_match = re.search(r'<answer>(.*?)$', content, re.DOTALL | re.IGNORECASE)
            
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # 如果没有 <answer> 标签，尝试从 reasoning 后面提取 \boxed{} 作为答案
                boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
                if boxed_match:
                    answer = boxed_match.group(0)  # 保留 \boxed{} 格式
                
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
        """构建错误分析的提示（简化版，任务说明由 system prompt 负责）"""
        # 解析 correct_answer（也是三段式格式）
        correct_caption, correct_reasoning, correct_final_answer = self._parse_model_output(correct_answer)
        
        # 如果解析失败，使用原始文本
        if not correct_caption and not correct_reasoning and not correct_final_answer:
            correct_caption = "[Unable to parse]"
            correct_reasoning = correct_answer  # 将整个答案放入 reasoning 部分
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
        解析 LLM 的分析结果
        
        Returns:
            (error_type, reason)
            error_type: "caption", "reasoning", 或 "unknown"
        """
        # 提取标签
        if "<caption_error>" in analysis_text.lower():
            error_type = "caption"
        elif "<reasoning_error>" in analysis_text.lower():
            error_type = "reasoning"
        else:
            # 没有找到标签，尝试从文本中推断
            text_lower = analysis_text.lower()
            if "caption" in text_lower and "error" in text_lower:
                error_type = "caption"
            elif "reasoning" in text_lower and "error" in text_lower:
                error_type = "reasoning"
            else:
                error_type = "unknown"
                self.logger.warning("无法从分析结果中识别错误类型，默认为 unknown")
        
        # 提取理由（移除标签后的文本）
        reason = re.sub(r'<(caption_error|reasoning_error)>', '', analysis_text, flags=re.IGNORECASE).strip()
        
        return error_type, reason



