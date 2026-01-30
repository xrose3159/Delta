import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass


logger = logging.getLogger(__name__)


def normalize_image_paths(image_path_raw: Union[str, List[str], None]) -> List[str]:
    标准化图像路径为列表格式

    Args:
        image_path_raw: 图像路径（可能是字符串、列表或None）

    Returns:
        图像路径列表
    if isinstance(image_path_raw, list):
        return image_path_raw
    elif isinstance(image_path_raw, str) and image_path_raw:
        return [image_path_raw]
    else:
        return []


def get_problem_id(problem: Dict[str, Any]) -> str:
    获取问题ID

    Args:
        problem: 问题字典

    Returns:
        问题ID字符串
    problem_id_val = problem.get('id')
    if problem_id_val is None or problem_id_val == 0:
        return str(hash(problem.get('problem', '')) % 100000)
    else:
        return str(problem_id_val)


def extract_boxed_answer(text: str) -> Optional[str]:
    提取 \\boxed{} 中的答案

    Args:
        text: 包含 \\boxed{} 的文本

    Returns:
        提取的答案字符串，如果没有找到则返回 None
    if not text:
        return None

    results = []
    i = 0
    while i < len(text):
        start = text.find('\\boxed{', i)
        if start == -1:
            break

        brace_start = start + 7
        brace_count = 1
        j = brace_start

        while j < len(text) and brace_count > 0:
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
            j += 1

        if brace_count == 0:
            results.append(text[brace_start:j-1])
            i = j
        else:
            i = brace_start

    return results[-1].strip() if results else None


@dataclass
class EasyR1DataPoint:
    problem: str
    answer: str
    images: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "problem": self.problem,
            "answer": self.answer,
        }
        if self.images:
            data["images"] = self.images
        return data


class GRPODataBuilder:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_generator = None
        self.llm_judge = None
        self._init_llm_components()

    def _init_llm_components(self):
        if not self.config.use_llm_generator:
            self.logger.warning("LLM Generator not enabled")
            return

        try:
            from ..processors.llm_wrapper import LLMasGenerator, LLMasJudge
            self.llm_generator = LLMasGenerator(self.config)
            if self.config.use_llm_judge:
                self.llm_judge = LLMasJudge(self.config)
            self.logger.info("✅ LLM Generator and Judge initialized for GRPO")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM components: {e}")
            self.llm_generator = None
            self.llm_judge = None

    def generate_single_answer(
        self,
        problem: Dict[str, Any],
        round_dir: Path,
        model_path: Optional[str] = None
    ) -> Optional[str]:
        为一道题目生成一个 ground truth 答案（用于 EasyR1 的 reward function）

        Args:
            problem: 题目字典
            round_dir: 轮次目录（用于保存临时文件）
            model_path: 模型路径

        Returns:
            生成的答案字符串，如果失败返回 None
        if not self.llm_generator:
            self.logger.warning("LLM Generator not available, cannot generate answer")
            return None

        problem_text = problem.get('problem', '')
        image_paths = normalize_image_paths(problem.get('image_path', ''))

        if not problem_text:
            self.logger.warning("Problem text is empty")
            return None


        input_item = {
            "problem": problem_text,
            "image_path": image_paths
        }


        temp_dir = round_dir / "grpo_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        problem_id = get_problem_id(problem)
        temp_input_file = temp_dir / f"answer_{problem_id}_input.json"
        temp_output_file = temp_dir / f"answer_{problem_id}_output.json"

        with open(temp_input_file, 'w', encoding='utf-8') as f:
            json.dump([input_item], f, ensure_ascii=False, indent=2)


        self.logger.debug(f"Generating ground truth answer for problem {problem_id}")
        if model_path is None:
            model_path = getattr(self.config, 'model_path', None)

        ground_truth_temperature = getattr(self.config, 'grpo_ground_truth_temperature', 0.2)
        self.logger.debug(f"Using temperature: {ground_truth_temperature} for ground truth generation")
        success = self.llm_generator.generate_predictions(
            test_file=temp_input_file,
            output_file=temp_output_file,
            model_path=model_path,
            max_tokens=self.config.eval_max_tokens,
            temperature=ground_truth_temperature
        )

        if not success or not temp_output_file.exists():
            self.logger.warning(f"Failed to generate answer for problem {problem_id}")
            return None


        with open(temp_output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)


        if results and len(results) > 0:
            answer = results[0].get('predict', '').strip()
            return answer if answer else None

        return None

    def check_answer_correctness(
        self,
        generated_answer: str,
        reference_answer: str,
        problem: Dict[str, Any]
    ) -> bool:
        检查生成的答案是否正确

        Args:
            generated_answer: 生成的答案
            reference_answer: 参考答案
            problem: 题目字典

        Returns:
            是否正确

        if self.llm_judge:
            try:

                eval_record = {
                    "id": get_problem_id(problem),
                    "question": problem.get('problem', ''),
                    "answer": reference_answer,
                    "category": problem.get('category', 0),
                    "category_name": problem.get('category_name', '')
                }


                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    eval_file = Path(f.name)
                    json.dump([eval_record], f, ensure_ascii=False, indent=2)

                predictions = [{"id": get_problem_id(problem), "predict": generated_answer}]
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    pred_file = Path(f.name)
                    json.dump(predictions, f, ensure_ascii=False, indent=2)

                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                    judge_file = Path(f.name)


                success = self.llm_judge.evaluate(
                    predictions_file=pred_file,
                    eval_records_file=eval_file,
                    output_file=judge_file
                )

                if success and judge_file.exists():
                    with open(judge_file, 'r', encoding='utf-8') as f:
                        judge_results = json.load(f)


                    if isinstance(judge_results, dict) and "eval_data" in judge_results:
                        judge_data = judge_results["eval_data"]
                    else:
                        judge_data = judge_results

                    if judge_data:
                        matched = judge_data[0].get('matched', False)

                        eval_file.unlink(missing_ok=True)
                        pred_file.unlink(missing_ok=True)
                        judge_file.unlink(missing_ok=True)
                        return matched


                eval_file.unlink(missing_ok=True)
                pred_file.unlink(missing_ok=True)
                judge_file.unlink(missing_ok=True)

            except Exception as e:
                self.logger.warning(f"Error using LLM Judge: {e}, falling back to boxed answer comparison")


        gen_box = extract_boxed_answer(generated_answer)
        ref_box = extract_boxed_answer(reference_answer)

        if gen_box and ref_box:
            return gen_box.strip() == ref_box.strip()

        return False

    def batch_check_answers_correctness(
        self,
        problems_with_answers: List[Tuple[Dict[str, Any], List[str]]],
        round_dir: Path
    ) -> Dict[str, List[bool]]:
        批量检查所有答案的正确性

        Args:
            problems_with_answers: 列表，每个元素是 (problem, answers) 元组
            round_dir: 轮次目录

        Returns:
            字典：problem_id -> [is_correct1, is_correct2, ...]
        if not self.llm_judge:

            results = {}
            for problem, answers in problems_with_answers:
                problem_id = get_problem_id(problem)
                reference_answer = problem.get('reference_answer', '') or problem.get('answer', '')
                ref_box = extract_boxed_answer(reference_answer)

                correctness = []
                for answer in answers:
                    gen_box = extract_boxed_answer(answer)
                    if gen_box and ref_box:
                        correctness.append(gen_box.strip() == ref_box.strip())
                    else:
                        correctness.append(False)

                results[problem_id] = correctness
            return results


        try:

            eval_records = []
            predictions = []
            problem_answer_mapping = []

            for problem, answers in problems_with_answers:
                problem_id = get_problem_id(problem)
                reference_answer = problem.get('reference_answer', '') or problem.get('answer', '')


                for idx, answer in enumerate(answers):
                    eval_records.append({
                        "id": f"{problem_id}_ans_{idx}",
                        "question": problem.get('problem', ''),
                        "answer": reference_answer,
                        "category": problem.get('category', 0),
                        "category_name": problem.get('category_name', '')
                    })
                    predictions.append({
                        "id": f"{problem_id}_ans_{idx}",
                        "predict": answer
                    })
                    problem_answer_mapping.append((problem_id, idx))

            if not eval_records:
                return {}


            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                eval_file = Path(f.name)
                json.dump(eval_records, f, ensure_ascii=False, indent=2)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                pred_file = Path(f.name)
                json.dump(predictions, f, ensure_ascii=False, indent=2)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                judge_file = Path(f.name)


            self.logger.info(f"Batch evaluating {len(predictions)} answers with LLM Judge...")
            success = self.llm_judge.evaluate(
                predictions_file=pred_file,
                eval_records_file=eval_file,
                output_file=judge_file
            )

            if not success or not judge_file.exists():
                self.logger.warning("Batch evaluation failed, falling back to individual checks")

                eval_file.unlink(missing_ok=True)
                pred_file.unlink(missing_ok=True)
                judge_file.unlink(missing_ok=True)
                return {}


            with open(judge_file, 'r', encoding='utf-8') as f:
                judge_results = json.load(f)


            if isinstance(judge_results, dict) and "eval_data" in judge_results:
                judge_data = judge_results["eval_data"]
            else:
                judge_data = judge_results


            id_to_result = {}
            for item in judge_data:
                item_id = item.get('id', '')
                matched = item.get('matched', False)
                id_to_result[item_id] = matched


            results = {}
            for (problem_id, answer_idx), eval_record in zip(problem_answer_mapping, eval_records):
                if problem_id not in results:
                    results[problem_id] = []

                eval_id = eval_record['id']
                is_correct = id_to_result.get(eval_id, False)
                results[problem_id].append(is_correct)


            eval_file.unlink(missing_ok=True)
            pred_file.unlink(missing_ok=True)
            judge_file.unlink(missing_ok=True)

            self.logger.info(f"✅ Batch evaluation completed for {len(results)} problems")
            return results

        except Exception as e:
            self.logger.warning(f"Error in batch evaluation: {e}, falling back to individual checks")
            return {}

    def build_grpo_dataset(
        self,
        wrong_problems: List[Dict[str, Any]],
        output_path: Path,
        round_dir: Path,
        model_path: Optional[str] = None
    ) -> Tuple[int, int, int]:
        从错题构建 EasyR1 格式的数据集（简化版）

        EasyR1 只需要 problem 和 answer（ground truth），不需要 chosen/rejected 对
        EasyR1 会在训练时通过 rollout 自动生成多个答案，并使用 reward function 评估

        Args:
            wrong_problems: 错题列表
            output_path: 输出文件路径（EasyR1 格式）
            round_dir: 轮次目录
            model_path: 当前模型路径

        Returns:
            (总数据点数, 处理的错题数, 跳过的错题数)
            注意：第三个返回值是跳过的题目数（skipped），不是丢弃数（discarded）
        easyr1_data = []
        skipped_data = []

        total_wrong = len(wrong_problems)

        self.logger.info(f"Building EasyR1 dataset: {total_wrong} wrong problems to process")
        self.logger.info("EasyR1 format: {problem, answer, images}")
        self.logger.info("Note: EasyR1 will generate multiple answers via rollout during training")


        processed = 0
        for problem in wrong_problems:
            processed += 1
            problem_id = get_problem_id(problem)
            self.logger.info(f"Processing problem {processed}/{total_wrong} (ID: {problem_id})...")


            reference_answer = problem.get('reference_answer', '') or problem.get('answer', '')


            if not reference_answer:
                self.logger.debug(f"No reference answer for problem {problem_id}, generating one...")
                generated_answer = self.generate_single_answer(problem, round_dir, model_path=model_path)

                if not generated_answer:
                    self.logger.warning(f"Failed to generate answer for problem {problem_id}, skipping")
                    skipped_data.append({
                        'problem_id': problem_id,
                        'problem': problem.get('problem', ''),
                        'skip_reason': 'failed_to_generate_answer'
                    })
                    continue

                reference_answer = generated_answer


            easyr1_point = self._create_easyr1_point(problem, reference_answer)
            if easyr1_point:
                easyr1_data.append(easyr1_point.to_dict())
                self.logger.debug(f"Created EasyR1 point for problem {problem_id}")
            else:
                skipped_data.append({
                    'problem_id': problem_id,
                    'problem': problem.get('problem', ''),
                    'skip_reason': 'failed_to_create_easyr1_point'
                })


        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(easyr1_data, f, ensure_ascii=False, indent=2)


        skipped_output_path = output_path.parent / f"{output_path.stem}_skipped.json"
        if skipped_data:
            with open(skipped_output_path, 'w', encoding='utf-8') as f:
                json.dump(skipped_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved {len(skipped_data)} skipped items to: {skipped_output_path}")


        self.logger.info(f"Created {len(easyr1_data)} EasyR1 data points from {total_wrong} wrong problems")
        self.logger.info(f"  - Successfully processed: {len(easyr1_data)}")
        self.logger.info(f"  - Skipped: {len(skipped_data)}")
        self.logger.info(f"Saved {len(easyr1_data)} EasyR1 data points to: {output_path}")

        return len(easyr1_data), total_wrong, len(skipped_data)

    def _create_easyr1_point(
        self,
        problem: Dict[str, Any],
        answer: str
    ) -> Optional[EasyR1DataPoint]:
        创建单个 EasyR1 数据点

        Args:
            problem: 题目字典
            answer: Ground truth 答案

        Returns:
            EasyR1DataPoint 或 None
        try:
            problem_text = problem.get('problem', '')

            if not problem_text or not answer:
                self.logger.warning("Missing required fields in problem")
                return None


            image_paths = normalize_image_paths(problem.get('image_path', ''))


            if image_paths:
                if '<image>' not in problem_text:
                    problem_text = '<image> ' + problem_text

            return EasyR1DataPoint(
                problem=problem_text,
                answer=answer.strip(),
                images=image_paths if image_paths else None
            )

        except Exception as e:
            self.logger.error(f"Error creating EasyR1 point: {e}")
            return None
