"""
DPO (Direct Preference Optimization) 数据准备模块

功能：
1. 从测试结果中提取错误的题目
2. 构建 DPO 训练数据对：(chosen: 正确答案, rejected: 错误答案)
3. 转换为 LLaMA-Factory 需要的格式
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# DPO 训练的 system prompt（与推理时保持一致）
DPO_SYSTEM_PROMPT = (
    "Solve the following problem carefully and thoroughly. "
    "Your response MUST follow this EXACT three-part structure using special tokens:\n\n"
    "**REQUIRED OUTPUT FORMAT:**\n"
    "<caption>Your image description here</caption>\n"
    "<reasoning>Your step-by-step solution here</reasoning>\n"
    "<answer>Your final answer here</answer>\n\n"
    "**DETAILED INSTRUCTIONS:**\n\n"
    "1. **<caption> Section** (Image Description):\n"
    "   - Start your response with the <caption> opening tag\n"
    "   - IMPORTANT: This problem contains an image. You MUST describe what you see in the image(s)\n"
    "   - Describe all relevant visual elements, diagrams, graphs, shapes, numbers, labels, or figures in detail\n"
    "   - Focus on the visual information that is relevant to solving the problem\n"
    "   - End this section with the </caption> closing tag\n\n"
    "2. **<reasoning> Section** (Step-by-Step Solution):\n"
    "   - Start this section with the <reasoning> opening tag\n"
    "   - Solve the problem using a clear, structured chain of thought\n"
    "   - Number each step explicitly: Step 1, Step 2, Step 3, etc.\n"
    "   - Break down the problem into smaller parts\n"
    "   - Show your reasoning clearly but concisely for each step\n"
    "   - Verify intermediate steps for correctness\n"
    "   - End this section with the </reasoning> closing tag\n\n"
    "3. **<answer> Section** (Final Answer):\n"
    "   - Start this section with the <answer> opening tag\n"
    "   - Provide the final answer in \\boxed{} format\n"
    "   - End this section with the </answer> closing tag\n"
    "   - Stop immediately after the closing tag. Do not add extra text.\n\n"
    "**CRITICAL**: You MUST use ALL THREE tags in your response: <caption></caption>, <reasoning></reasoning>, and <answer></answer>. "
    "Missing any tag will result in an invalid response."
)


def normalize_image_paths(image_path_raw: Union[str, List[str], None]) -> List[str]:
    """
    标准化图像路径为列表格式
    
    Args:
        image_path_raw: 图像路径（可能是字符串、列表或None）
        
    Returns:
        图像路径列表
    """
    if isinstance(image_path_raw, list):
        return image_path_raw
    elif isinstance(image_path_raw, str) and image_path_raw:
        return [image_path_raw]
    else:
        return []


def get_problem_id(problem: Dict[str, Any]) -> str:
    """
    获取问题ID，确保与 _generate_corrected_cots_for_wrong_problems 一致
    
    Args:
        problem: 问题字典
        
    Returns:
        问题ID字符串
    """
    problem_id_val = problem.get('id')
    if problem_id_val is None or problem_id_val == 0:
        return str(hash(problem.get('problem', '')) % 100000)
    else:
        return str(problem_id_val)


@dataclass
class DPODataPoint:
    """DPO 数据点"""
    instruction: str  # 问题/指令
    chosen: str  # 正确答案（chosen response）
    rejected: str  # 错误答案（rejected response）
    images: List[str] = None  # 图像路径列表
    system: str = None  # system prompt（可选）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 ShareGPT DPO 格式"""
        # ShareGPT DPO 格式：使用简单的字符串格式
        # user message 直接用文本（包含 <image> token）
        # 图片通过顶层的 images 字段传递
        data = {
            "conversations": [
                {
                    "from": "user",
                    "value": self.instruction  # 直接使用字符串（包含 <image> token）
                }
            ],
            "chosen": {
                "from": "assistant",
                "value": self.chosen
            },
            "rejected": {
                "from": "assistant",
                "value": self.rejected
            }
        }
        
        # 添加 system prompt（如果提供）
        if self.system:
            data["system"] = self.system
        
        # 添加 images 字段（LLaMA-Factory 需要）
        if self.images:
            data["images"] = self.images
        
        return data


class DPODataBuilder:
    """DPO 数据构建器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_generator = None
        self._init_llm_generator()
    
    def _init_llm_generator(self):
        """初始化 LLM Generator（延迟导入）"""
        if not self.config.use_llm_generator:
            self.logger.warning("LLM Generator not enabled, will use reference_answer as chosen")
            return
        
        try:
            from ..processors.llm_wrapper import LLMasGenerator
            self.llm_generator = LLMasGenerator(self.config)
            self.logger.info("✅ LLM Generator initialized for DPO chosen generation")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM Generator: {e}, will use reference_answer as chosen")
            self.llm_generator = None
    
    def build_dpo_dataset(
        self,
        eval_results: List[Dict[str, Any]],
        output_path: Path,
        corrected_cots: Optional[Dict[str, str]] = None
    ) -> Tuple[int, int]:
        """
        从评估结果构建 DPO 数据集
        
        Args:
            eval_results: 评估结果列表，每个包含:
                - problem: 题目
                - reference_answer: 参考答案（正确）
                - model_prediction: 模型预测（可能错误）
                - matched: 是否正确
                - image_path: 图像路径
            output_path: 输出文件路径
            corrected_cots: 预先生成的修正CoT字典，key为problem_id，value为corrected_cot
        
        Returns:
            (总数据点数, 错误题目数)
        """
        dpo_data = []
        skipped_data = []  # 保存被跳过的数据
        total_wrong = sum(1 for item in eval_results if not item.get('matched', False))
        
        self.logger.info(f"Building DPO dataset: {total_wrong} wrong problems to process")
        if corrected_cots:
            self.logger.info(f"Using pre-generated corrected CoT for {len(corrected_cots)} problems")
        
        processed = 0
        for item in eval_results:
            # 只处理错误的题目
            if not item.get('matched', False):
                processed += 1
                # 确保 problem_id 生成方式与 _generate_corrected_cots_for_wrong_problems 一致
                problem_id = get_problem_id(item)
                self.logger.debug(f"Processing wrong problem {processed}/{total_wrong} (ID: {problem_id})...")
                
                # 检查是否有corrected_cot
                corrected_cot = corrected_cots.get(problem_id) if corrected_cots else None
                
                dpo_point, skip_reason = self._create_dpo_point_with_reason(item, corrected_cot=corrected_cot)
                if dpo_point:
                    dpo_data.append(dpo_point.to_dict())
                else:
                    # 保存被跳过的数据（包含跳过原因）
                    skipped_item = {
                        'problem_id': problem_id,
                        'problem': item.get('problem', ''),
                        'model_prediction': item.get('model_prediction', ''),
                        'has_corrected_cot': corrected_cot is not None,
                        'corrected_cot': corrected_cot if corrected_cot else None,
                        'skip_reason': skip_reason,
                        'original_item': item  # 保存原始数据以便调试
                    }
                    skipped_data.append(skipped_item)
                    self.logger.debug(f"⚠️  Skipped problem {processed}/{total_wrong} (ID: {problem_id}): {skip_reason}")
        
        # 保存成功的数据
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)
        
        # 保存被跳过的数据
        skipped_output_path = output_path.parent / f"{output_path.stem}_skipped.json"
        if skipped_data:
            with open(skipped_output_path, 'w', encoding='utf-8') as f:
                json.dump(skipped_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(skipped_data)} skipped items to: {skipped_output_path}")
        
        # 输出简化统计
        self.logger.info(f"Created {len(dpo_data)} DPO data points from {total_wrong} wrong problems")
        self.logger.info(f"Saved {len(dpo_data)} DPO data points to: {output_path}")
        
        return len(dpo_data), total_wrong
    
    def _create_dpo_point_with_reason(self, eval_item: Dict[str, Any], corrected_cot: Optional[str] = None) -> Tuple[Optional[DPODataPoint], str]:
        """
        创建单个 DPO 数据点，并返回跳过原因（如果失败）
        
        Returns:
            (DPODataPoint 或 None, 跳过原因字符串)
        """
        dpo_point = self._create_dpo_point(eval_item, corrected_cot)
        if dpo_point:
            return dpo_point, None
        
        # 确定跳过原因
        if not eval_item.get('problem') or not eval_item.get('model_prediction'):
            return None, 'missing_fields'
        if not corrected_cot:
            return None, 'no_corrected_cot'
        
        # 注意：corrected_cot 在生成阶段已经通过 _validate_corrected_cot 验证，
        # 包括是否有 \boxed{} 答案、答案是否与错误答案相同等
        # 如果 _create_dpo_point 返回 None，可能是其他原因（如格式问题等）
        return None, 'other_errors'
    
    def _create_dpo_point(self, eval_item: Dict[str, Any], corrected_cot: Optional[str] = None) -> Optional[DPODataPoint]:
        """
        创建单个 DPO 数据点
        
        Args:
            eval_item: 评估结果项
            corrected_cot: 预先生成的修正CoT（如果提供，优先使用）
        
        Returns:
            DPODataPoint 或 None
        """
        try:
            problem = eval_item.get('problem', '')
            model_prediction = eval_item.get('model_prediction', '')  # rejected: 错误的 CoT（长答案）
            
            if not problem or not model_prediction:
                self.logger.warning("Missing required fields in eval item")
                return None
            
            # chosen 必须使用 Qwen 生成的 corrected CoT
            # rejected 是 model_prediction（错误的 CoT，长答案）
            # 如果 corrected_cot 不存在，跳过这个数据点（不使用 reference_answer，因为格式不匹配）
            if corrected_cot:
                chosen_answer = corrected_cot
                self.logger.debug("Using pre-generated corrected CoT as chosen")
            else:
                self.logger.warning("No corrected CoT provided, skipping this DPO point (chosen must be corrected CoT, not reference_answer)")
                return None
            
            # 标准化图像路径
            image_paths = normalize_image_paths(eval_item.get('image_path', ''))
            
            # 如果有图像，确保问题文本中有 <image> token
            instruction = problem
            if image_paths:
                if '<image>' not in instruction:
                    instruction = '<image> ' + instruction
            
            # 清理 chosen 和 rejected 中的 <image> token
            # LLaMA-Factory 只处理 instruction 中的图片，chosen/rejected 中的 <image> token 会导致错误
            def remove_image_tokens(text: str) -> str:
                """移除文本中的所有 <image> token（包括可能的空格）"""
                # 移除 <image> 及其前后的空格
                text = re.sub(r'\s*<image>\s*', ' ', text)
                # 清理多余的空格
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            cleaned_chosen = remove_image_tokens(chosen_answer)
            cleaned_rejected = remove_image_tokens(model_prediction)
            
            # 过滤掉过长的rejected（防止显存OOM）
            # 设置最大长度为 10000 字符（约 2500 tokens）
            # 超长rejected通常是生成bug（文本重复），训练价值低，直接过滤
            max_length = 20000
            if len(cleaned_rejected) > max_length:
                self.logger.warning(f"Rejected too long ({len(cleaned_rejected)} chars), skipping this sample (likely generation bug)")
                return None  # 直接过滤掉，而不是截断
            
            return DPODataPoint(
                instruction=instruction,  
                chosen=cleaned_chosen,  
                rejected=cleaned_rejected,  
                images=image_paths if image_paths else None,
                system=DPO_SYSTEM_PROMPT  # 添加 system prompt
            )
        
        except Exception as e:
            self.logger.error(f"Error creating DPO point: {e}")
            return None
    
    def validate_dpo_dataset(self, dataset_path: Path) -> bool:
        """
        验证 DPO 数据集格式
        
        Args:
            dataset_path: 数据集路径
        
        Returns:
            是否有效
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                self.logger.error("DPO dataset must be a list")
                return False
            
            required_fields = ['instruction', 'input', 'chosen', 'rejected']
            
            for i, item in enumerate(data):
                for field in required_fields:
                    if field not in item:
                        self.logger.error(f"Item {i} missing required field: {field}")
                        return False
                
                # 验证 chosen 和 rejected 类型
                if not isinstance(item['chosen'], str) or not isinstance(item['rejected'], str):
                    self.logger.error(f"Item {i}: chosen and rejected must be strings")
                    return False
                
                # 验证 images 字段（如果存在）应该是列表
                if 'images' in item and not isinstance(item['images'], (list, type(None))):
                    self.logger.error(f"Item {i}: images must be a list or None")
                    return False
            
            self.logger.info(f"✅ DPO dataset validation passed: {len(data)} items")
            return True
        
        except Exception as e:
            self.logger.error(f"Error validating DPO dataset: {e}")
            return False
    
    def create_dpo_config_for_llamafactory(
        self,
        dataset_name: str,
        dataset_path: Path,
        output_dir: Path
    ) -> Path:
        """
        创建 LLaMA-Factory DPO 训练配置
        
        Args:
            dataset_name: 数据集名称
            dataset_path: 数据集路径
            output_dir: 输出目录
        
        Returns:
            配置文件路径
        """
        config = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "stage": "dpo",
            "model_name_or_path": str(self.config.model_base_path / self.config.base_model_name),
            "do_train": True,
            "finetuning_type": "full",
            
            # DPO 特定参数
            "dpo_beta": self.config.dpo_beta,
            "dpo_loss": "sigmoid",  # sigmoid, hinge, ipo
            "dpo_ftx": 0.0,  # 监督学习权重
            
            # 训练参数
            "num_train_epochs": self.config.dpo_epochs,
            "per_device_train_batch_size": self.config.dpo_batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": self.config.dpo_learning_rate,
            "max_length": self.config.dpo_max_length,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            
            # 输出配置
            "output_dir": str(output_dir),
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 2,
            
            # 模板
            "template": self.config.dpo_template,
            
            # 其他
            "bf16": True,
            "ddp_timeout": 180000000,
            "val_size": 0.1,
            "eval_steps": 100,
            "per_device_eval_batch_size": 1,
        }
        
        config_path = output_dir / "dpo_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Created DPO config: {config_path}")
        
        return config_path


def extract_wrong_problems(
    eval_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    从评估结果中提取错误的题目
    
    Args:
        eval_results: 评估结果列表
    
    Returns:
        错误题目列表
    """
    wrong_problems = []
    
    for item in eval_results:
        if not item.get('matched', False):
            wrong_problems.append(item)
    
    return wrong_problems


def extract_correct_problems(
    eval_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    从评估结果中提取正确的题目
    
    Args:
        eval_results: 评估结果列表
    
    Returns:
        正确题目列表
    """
    correct_problems = []
    
    for item in eval_results:
        if item.get('matched', False):
            correct_problems.append(item)
    
    return correct_problems


if __name__ == "__main__":
    # 测试代码
    from config import get_default_config
    
    config = get_default_config()
    builder = DPODataBuilder(config)
    
    # 模拟评估结果
    eval_results = [
        {
            "problem": "<image> What is 2+2?",
            "reference_answer": "<think>2+2=4</think> \\boxed{4}",
            "model_prediction": "<think>2+2=5</think> \\boxed{5}",
            "matched": False,
            "image_path": ["/path/to/image.png"]
        },
        {
            "problem": "<image> What is 3+3?",
            "reference_answer": "<think>3+3=6</think> \\boxed{6}",
            "model_prediction": "<think>3+3=6</think> \\boxed{6}",
            "matched": True,
            "image_path": ["/path/to/image2.png"]
        }
    ]
    
    # 构建 DPO 数据集
    output_path = Path("./test_dpo_dataset.json")
    num_points, num_wrong = builder.build_dpo_dataset(eval_results, output_path)
    
    print(f"Created {num_points} DPO points from {num_wrong} wrong problems")
    
    # 验证
    is_valid = builder.validate_dpo_dataset(output_path)
    print(f"Validation result: {is_valid}")

