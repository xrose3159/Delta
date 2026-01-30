from __future__ import annotations

import copy
import json
import math
import logging
import os
import random
import re
import subprocess
import gc
import time
import sys
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import ViTImageProcessor, ViTModel

from ..config import AdaptiveConfig
from ..processors.gemini_generator import GeminiHardProblemGenerator

logger = logging.getLogger(__name__)


@dataclass
class ProblemRecord:

    pid: str
    category_id: int
    category_name: str
    question: str
    answer: str
    image_path: str
    source: str = "original"
    last_prediction: Optional[str] = None
    image_code: Optional[str] = None
    idx: Optional[int] = None
    model_answer: Optional[str] = None


class MultiRoundFineTuner:

    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config
        self.random_seed = config.random_seed
        self.rng = random.Random(self.random_seed)

        self.problem_map: Dict[str, ProblemRecord] = {}
        self.available_problem_ids: List[str] = []
        self.prev_wrong_ids: List[str] = []
        self.generated_bank: List[ProblemRecord] = []
        self.generated_counter: int = 0
        self.history: List[Dict[str, object]] = []
        self.current_model_path: Optional[str] = config.model_path


        self.next_generated_id: int = 5844

        self.gemini_generator = GeminiHardProblemGenerator(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            base_url=config.gemini_base_url,
            max_tokens=config.gemini_max_tokens,
            max_output_tokens=config.gemini_max_output_tokens,
        )

        self._load_problems()




    def run(self) -> None:
        start_round = self.config.start_round
        logger.info("Starting multi-round fine-tuning pipeline – rounds=%s, starting from round %s",
                    self.config.max_rounds, start_round)

        for round_index in range(start_round, self.config.max_rounds + 1):
            round_dir = self.config.get_round_dir(round_index)
            dataset_dir = round_dir / "dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            logger.info("\n%s\n[Round %s] Preparing datasets\n%s", "=" * 70, round_index, "=" * 70)
            train_records, eval_records = self._prepare_round_datasets(round_index)


            train_file, eval_file = self._write_dataset_files(dataset_dir, train_records, eval_records)
            self._write_dataset_info(dataset_dir, train_file.name, eval_file.name)

            train_yaml = self._write_train_yaml(round_index, dataset_dir)
            eval_yaml = self._write_eval_yaml(round_index, dataset_dir)




            logger.info("[Round %s] Launching training via LLaMA-Factory", round_index)
            success = self._run_llamafactory_command(
                self.config.train_command_prefix + ["llamafactory-cli", "train", str(train_yaml)]
            )
            if not success:
                logger.error("Training failed for round %s – aborting multi-round pipeline.", round_index)
                break

            round_model_dir = self._get_round_model_dir(round_index)
            self.current_model_path = str(round_model_dir)




            logger.info("[Round %s] Running evaluation with vLLM (Apptainer)", round_index)


            logger.info("Waiting for training process to release GPU memory...")
            time.sleep(15)
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")


            eval_output_dir = round_model_dir / "eval"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = eval_output_dir / "generated_predictions.json"


            eval_dataset_file = dataset_dir / "eval.json"
            if not eval_dataset_file.exists():
                logger.error("Evaluation dataset not found: %s", eval_dataset_file)
                logger.error("Evaluation failed for round %s – aborting pipeline.", round_index)
                break


            llm_generator_script = Path(__file__).parent.parent / "processors" / "llmasgenerator.py"


            apptainer_image = os.getenv("APPTAINER_IMAGE", self.config.apptainer_image if hasattr(self.config, 'apptainer_image') else "")

            if not apptainer_image:
                logger.error("APPTAINER_IMAGE not set. Please set it in environment or config.")
                logger.error("Evaluation failed for round %s – aborting pipeline.", round_index)
                break


            vllm_cmd = [
                "apptainer", "exec", "--nv",
                "--cleanenv",
                "--bind", "/share:/share,/mnt:/mnt",
                "--env", "TRITON_CACHE_DIR=/tmp/triton_cache",
                "--env", "PATH=/opt/py312/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                "--env", "CC=/usr/bin/gcc",
                "--env", "CXX=/usr/bin/g++",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--env", f"PYTHONPATH={Path(__file__).parent.parent.parent}",
                apptainer_image,
                "python", str(llm_generator_script),
                "--model-path", str(round_model_dir),
                "--input", str(eval_dataset_file),
                "--output", str(predictions_path),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--temperature", str(self.config.eval_temperature),
                "--top-p", str(self.config.eval_top_p),
                "--max-tokens", str(self.config.eval_max_tokens),
                "--tensor-parallel-size", str(self.config.eval_tensor_parallel_size),
            ]

            logger.info("Running vLLM evaluation: %s", " ".join(map(str, vllm_cmd)))

            try:
                result = subprocess.run(
                    vllm_cmd,
                    check=True,
                    stdout=None,
                    stderr=None,
                    text=True,
                )
                logger.info("✅ vLLM evaluation completed successfully")
            except subprocess.CalledProcessError as exc:
                logger.error("❌ vLLM evaluation failed with exit code %d", exc.returncode)
                logger.error("Evaluation failed for round %s – aborting pipeline.", round_index)
                break


            if not predictions_path.exists():
                logger.error("Prediction file not found: %s", predictions_path)
                logger.error("Evaluation failed for round %s – aborting pipeline.", round_index)
                break


            logger.info("Evaluation complete.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            logger.info("GPU memory released. Waiting 20 seconds for GPU to fully release memory...")
            time.sleep(20)
            logger.info("Ready to start LLM Judge.")


            wrong_records, category_stats = self._analyse_predictions(predictions_path, eval_records)


            hard_generated = self._generate_hard_problems(category_stats, wrong_records, round_index)
            refresh_generated = self._generate_refresh_problems(category_stats, wrong_records, eval_records, round_index)
            all_generated = hard_generated + refresh_generated


            if all_generated:


                logger.info("Generating images for %d newly generated problems.", len(all_generated))
                failed_pids = self._materialise_problem_images(dataset_dir, all_generated)


                if failed_pids:
                    failed_set = set(failed_pids)
                    all_generated = [r for r in all_generated if r.pid not in failed_set]
                    logger.info("Filtered out %d problems with failed image generation", len(failed_pids))
                    logger.info("Remaining: %d generated problems", len(all_generated))


                logger.info("Generating COT answers.")
                self._generate_model_answers(dataset_dir, all_generated)


                if self.config.enable_quality_check and all_generated:
                    logger.info("\n%s\n[Round %s] Quality check for newly generated problems\n%s",
                               "=" * 70, round_index, "=" * 70)
                    logger.info("Testing newly generated problems with current model.")

                    qualified_problems = self._quality_check_generated_problems(
                        dataset_dir, all_generated, round_model_dir, round_index
                    )

                    filtered_count = len(all_generated) - len(qualified_problems)
                    if filtered_count > 0:
                        logger.info("Quality check filtered out %d problems (too easy or too hard)", filtered_count)
                        logger.info("Remaining: %d qualified problems", len(qualified_problems))
                        all_generated = qualified_problems
                    else:
                        logger.info("All %d problems passed quality check", len(all_generated))

                logger.info("Successfully processed %d newly generated problems", len(all_generated))
            else:
                logger.info("No new problems generated in this round")

            self._update_state_after_round(
                round_index,
                train_records,
                eval_records,
                wrong_records,
                all_generated,
            )

            self._write_round_summary(
                round_dir,
                train_records,
                eval_records,
                wrong_records,
                category_stats,
                all_generated,
            )

        self._write_overall_summary()




    def _load_problems(self) -> None:
        logger.info("Loading WeMath standard dataset for multi-round pipeline")

        wemath_records = self._load_wemath_standard(Path(self.config.dataset_path))


        has_builtin_categories = False
        if wemath_records:
            first_record = wemath_records[0]
            has_builtin_categories = "category" in first_record and "category_name" in first_record


        classification_map: Dict[str, int] = {}
        if not has_builtin_categories:
            logger.info("Dataset does not contain built-in category info, loading from classification file")
            classification_map = self._load_classification_map(Path(self.config.classification_file))
            logger.info("Loaded %s category mappings from external file", len(classification_map))
        else:
            logger.info("Dataset contains built-in category info, will use it directly")

        skipped = 0
        for idx, record in enumerate(wemath_records):
            pid = str(record.get("id", ""))
            if idx < 3:
                logger.info("Record %d: raw_id=%s, pid=%s, type=%s", idx, record.get("id"), pid, type(record.get("id")))
            if not pid:
                skipped += 1
                continue


            if has_builtin_categories:
                category_id = record.get("category", 0)
                category_name = record.get("category_name", "Unknown")
            else:

                category_id = classification_map.get(pid, 0)
                category_name = self.config.categories.get(category_id, "未知类别")


            answer_text = record.get("model_answer", "")
            if not answer_text or not isinstance(answer_text, str):
                answer_text = record.get("answer", "")
            answer_text = str(answer_text).strip()

            problem = ProblemRecord(
                pid=pid,
                category_id=category_id,
                category_name=category_name,
                question=record.get("problem", ""),
                answer=answer_text,
                image_path=record.get("image_path", ""),
                source="original",
                image_code=None,
                idx=record.get("idx"),
                model_answer=None,
            )
            self.problem_map[pid] = problem

        if skipped > 0:
            logger.warning("Skipped %s records with empty ID", skipped)

        self.available_problem_ids = list(self.problem_map.keys())
        self.rng.shuffle(self.available_problem_ids)
        logger.info("Loaded %s problems from dataset", len(self.problem_map))

    @staticmethod
    def _load_classification_map(path: Path) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        if not path.exists():
            logger.warning("Classification file not found at %s", path)
            return mapping

        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    pid, label = line.split("\t", 1)
                    cat_id = int(label.split("-", 1)[0].strip())
                    mapping[str(pid).strip()] = cat_id
                except ValueError:
                    continue
        return mapping

    @staticmethod
    def _load_wemath_standard(path: Path) -> List[Dict[str, object]]:
        logger.info("Loading from path: %s", path)
        logger.info("Path exists: %s", path.exists())

        if not path.exists():
            logger.error("WeMath standard dataset not found at %s", path)
            return []

        with path.open("r", encoding="utf-8") as fin:
            try:
                data = json.load(fin)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse WeMath standard dataset: %s", exc)
                return []

        if not isinstance(data, list):
            logger.error("WeMath standard dataset should be a JSON array")
            return []

        logger.info("Loaded %s records from %s", len(data), path)
        if data:
            logger.info("First record keys: %s", list(data[0].keys()))
            logger.info("First record id: %s (type: %s)", data[0].get("id"), type(data[0].get("id")))
        return data




    def _prepare_round_datasets(self, round_index: int) -> Tuple[List[ProblemRecord], List[ProblemRecord]]:
        准备每轮的训练和验证数据集
        1. 第一轮：使用原始数据集，按比例划分
        2. 后续轮：上一轮验证集的全部错题 + 生成新题
        if round_index == 1:
            return self._prepare_first_round_datasets()
        else:
            return self._prepare_adaptive_round_datasets(round_index)

    def _prepare_first_round_datasets(self) -> Tuple[List[ProblemRecord], List[ProblemRecord]]:
        if not self.available_problem_ids:
            raise RuntimeError("No data available for the first round.")


        all_records = [self.problem_map[pid] for pid in self.available_problem_ids]


        if self.config.round_total_samples > 0:
            all_records = all_records[:self.config.round_total_samples]

        logger.info("Round 1 total data: %d samples", len(all_records))


        self.rng.shuffle(all_records)


        train_size = max(1, int(len(all_records) * self.config.train_ratio))
        train_records = all_records[:train_size]
        eval_records = all_records[train_size:]

        logger.info(
            "Round 1 dataset split – train=%d (%.1f%%), eval=%d (%.1f%%)",
            len(train_records),
            len(train_records) / len(all_records) * 100,
            len(eval_records),
            len(eval_records) / len(all_records) * 100,
        )


        self.available_problem_ids = []

        return train_records, eval_records

    def _prepare_adaptive_round_datasets(self, round_index: int) -> Tuple[List[ProblemRecord], List[ProblemRecord]]:


        base_records = [self.problem_map[pid] for pid in self.prev_wrong_ids if pid in self.problem_map]

        if not base_records:
            logger.warning("No wrong problems from previous round, cannot continue.")
            raise RuntimeError("No wrong problems available for adaptive training.")

        logger.info("Round %d base data: %d wrong problems from previous round", round_index, len(base_records))


        all_records = list(self.generated_bank) + base_records

        logger.info("Round %d total data: %d samples (generated=%d, wrong=%d)",
                   round_index, len(all_records), len(self.generated_bank), len(base_records))


        self.rng.shuffle(all_records)


        train_size = max(1, int(len(all_records) * self.config.train_ratio))
        train_records = all_records[:train_size]
        eval_records = all_records[train_size:]

        logger.info(
            "Round %d dataset split – train=%d (%.1f%%), eval=%d (%.1f%%)",
            round_index,
            len(train_records),
            len(train_records) / len(all_records) * 100,
            len(eval_records),
            len(eval_records) / len(all_records) * 100,
        )

        return train_records, eval_records

    def _pop_available_ids(self, count: int) -> List[str]:
        count = max(0, min(count, len(self.available_problem_ids)))
        selected = self.available_problem_ids[:count]
        self.available_problem_ids = self.available_problem_ids[count:]
        return selected




    def _write_dataset_files(
        self,
        dataset_dir: Path,
        train_records: List[ProblemRecord],
        eval_records: List[ProblemRecord],
    ) -> Tuple[Path, Path]:
        train_path = dataset_dir / "train.json"
        eval_path = dataset_dir / "eval.json"

        self._dump_alpaca_format(train_path, train_records)
        self._dump_alpaca_format(eval_path, eval_records)

        return train_path, eval_path

    def _dump_alpaca_format(self, path: Path, records: List[ProblemRecord]) -> None:
        serialisable = []
        fixed_count = 0

        for problem in records:

            if problem.pid.isdigit():
                id_value = int(problem.pid)
            elif "_" in problem.pid:

                parts = problem.pid.split("_")
                id_value = int(parts[-1]) if parts[-1].isdigit() else hash(problem.pid) % (10**9)
            else:

                id_value = hash(problem.pid) % (10**9)


            image_path_list = [problem.image_path] if problem.image_path else []


            question_text = problem.question
            if image_path_list and "<image>" not in question_text:

                question_text = "<image> " + question_text
                fixed_count += 1
                logger.debug("Auto-fixed: added <image> token for problem %s", problem.pid)
            elif not image_path_list and "<image>" in question_text:

                question_text = question_text.replace("<image>", "").strip()
                fixed_count += 1
                logger.debug("Auto-fixed: removed <image> token from problem %s", problem.pid)

            entry = {
                "id": id_value,
                "problem": question_text,
                "answer": problem.answer,
                "image_path": image_path_list,
                "category": problem.category_id,
                "category_name": problem.category_name,
                "system": (
                    "Solve the following problem carefully and thoroughly. Use a long, detailed chain of thought to reason step-by-step. "
                    "Make sure to: "
                    "1. Explicitly break down the problem into smaller parts. "
                    "2. Explore alternative approaches when relevant. "
                    "3. Verify intermediate steps for correctness. "
                    "4. Clearly summarize the reasoning before giving the final answer. "
                    "5. Enclose the final answer in \\boxed{}."
                ),
            }


            if problem.idx is not None:
                entry["idx"] = problem.idx


            entry["_metadata"] = {
                "source": problem.source,
            }

            if problem.image_code:
                entry["_metadata"]["image_code"] = problem.image_code

            serialisable.append(entry)

        if fixed_count > 0:
            logger.info("Auto-fixed %d problems with missing <image> tokens in %s", fixed_count, path.name)

        with path.open("w", encoding="utf-8") as fout:
            json.dump(serialisable, fout, ensure_ascii=False, indent=2)


    @staticmethod
    def _write_dataset_info(dataset_dir: Path, train_filename: str, eval_filename: str) -> None:
        dataset_info = {
            "round_train": {
                "file_name": train_filename,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "problem",
                    "response": "answer",
                    "images": "image_path",
                    "system": "system"
                },
            },
            "round_eval": {
                "file_name": eval_filename,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "problem",
                    "response": "answer",
                    "images": "image_path",
                    "system": "system"
                },
            },
        }

        info_path = dataset_dir / "dataset_info.json"
        with info_path.open("w", encoding="utf-8") as fout:
            json.dump(dataset_info, fout, ensure_ascii=False, indent=2)

    def _write_train_yaml(self, round_index: int, dataset_dir: Path) -> Path:
        with open(self.config.train_yaml_template, "r", encoding="utf-8") as fin:
            config = yaml.safe_load(fin)

        output_dir = self._get_round_model_dir(round_index)
        output_dir.mkdir(parents=True, exist_ok=True)

        config["model_name_or_path"] = self.current_model_path
        config["dataset_dir"] = str(dataset_dir)
        config["dataset"] = "round_train"
        config.pop("eval_dataset", None)
        config["output_dir"] = str(output_dir)


        if self.config.round_total_samples <= 0:
            config.pop("max_samples", None)
        else:
            config["max_samples"] = self.config.round_total_samples

        yaml_path = dataset_dir.parent / f"round_{round_index:02d}_train.yaml"
        with yaml_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(config, fout, sort_keys=False, allow_unicode=True)
        return yaml_path

    def _write_eval_yaml(self, round_index: int, dataset_dir: Path) -> Path:
        with open(self.config.eval_yaml_template, "r", encoding="utf-8") as fin:
            config = yaml.safe_load(fin)

        eval_output = self._get_round_model_dir(round_index) / "eval"
        eval_output.mkdir(parents=True, exist_ok=True)

        config["model_name_or_path"] = str(self._get_round_model_dir(round_index))
        config["dataset_dir"] = str(dataset_dir)
        config.pop("dataset", None)
        config.pop("dataset_mixer", None)
        config["eval_dataset"] = ["round_eval"]
        config["output_dir"] = str(eval_output)
        config["stage"] = "sft"
        config["do_train"] = False
        config["do_eval"] = False
        config["do_predict"] = True
        config["predict_with_generate"] = True
        config.pop("deepspeed", None)

        yaml_path = dataset_dir.parent / f"round_{round_index:02d}_eval.yaml"
        with yaml_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(config, fout, sort_keys=False, allow_unicode=True)
        return yaml_path

    def _get_round_model_dir(self, round_index: int) -> Path:
        return self.config.workspace_dir / "models" / f"round_{round_index:02d}"




    def _run_llamafactory_command(self, command: List[str]) -> bool:
        logger.info("Executing command: %s", " ".join(command))
        try:
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Command failed with exit code %s: %s", exc.returncode, command)
            return False




    def _analyse_predictions(
        self,
        predictions_path: Path,
        eval_records: List[ProblemRecord],
    ) -> Tuple[List[ProblemRecord], Dict[int, Dict[str, float]]]:

        if self.config.use_llm_judge:
            logger.info("Using LLM-as-Judge for evaluation")
            return self._analyse_predictions_with_llm_judge(predictions_path, eval_records)
        else:
            logger.info("Using rule-based matching for evaluation")
            return self._analyse_predictions_with_rules(predictions_path, eval_records)

    def _analyse_predictions_with_llm_judge(
        self,
        predictions_path: Path,
        eval_records: List[ProblemRecord],
    ) -> Tuple[List[ProblemRecord], Dict[int, Dict[str, float]]]:

        eval_records_dict = []
        for record in eval_records:
            eval_records_dict.append({
                'pid': record.pid,
                'question': record.question,
                'answer': record.answer,
                'category_id': record.category_id,
                'category_name': record.category_name,
            })


        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            eval_records_file = Path(f.name)
            json.dump(eval_records_dict, f, ensure_ascii=False, indent=2)


        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            judge_output = Path(f.name)


        judge_script = Path(__file__).parent.parent / "processors" / "run_llm_judge.py"


        judge_config = {
            'model_path': self.config.judge_model_path,
            'tensor_parallel_size': self.config.judge_tensor_parallel_size,
            'gpu_memory_utilization': self.config.judge_gpu_memory_utilization,
            'temperature': self.config.judge_temperature,
            'max_tokens': self.config.judge_max_tokens,
            'max_model_len': self.config.judge_max_model_len,
        }
        judge_config_json = json.dumps(judge_config)

        logger.info("Running LLM Judge in separate process to avoid CUDA conflicts...")
        logger.info("Judge script: %s", judge_script)
        logger.info("Predictions: %s", predictions_path)
        logger.info("Eval records: %d items", len(eval_records))



        try:
            import subprocess
            import select
            import time

            process = subprocess.Popen(
                [
                    sys.executable,
                    str(judge_script),
                    str(eval_records_file),
                    str(predictions_path),
                    str(judge_output),
                    judge_config_json,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )


            logger.info("=== LLM Judge subprocess output ===")
            last_output_time = time.time()
            no_output_timeout = 300
            graceful_termination = False

            while True:

                if process.poll() is not None:

                    remaining = process.stdout.read()
                    if remaining:
                        for line in remaining.splitlines():
                            logger.info("  [Judge] %s", line.rstrip())
                    break


                import threading
                line_container = []

                def read_line():
                    try:
                        line = process.stdout.readline()
                        if line:
                            line_container.append(line)
                    except:
                        pass

                reader_thread = threading.Thread(target=read_line)
                reader_thread.daemon = True
                reader_thread.start()
                reader_thread.join(timeout=1.0)

                if line_container:
                    line = line_container[0]
                    logger.info("  [Judge] %s", line.rstrip())
                    last_output_time = time.time()
                else:

                    if time.time() - last_output_time > no_output_timeout:
                        logger.warning("⚠️  Judge subprocess has no output for %d seconds, but process is still running", no_output_timeout)
                        logger.warning("⚠️  This might indicate the process completed but stdout wasn't closed properly")
                        logger.warning("⚠️  Checking if output file exists...")


                        if judge_output.exists():
                            logger.info("✅ Output file exists! Judge likely completed. Terminating subprocess.")
                            graceful_termination = True
                            process.terminate()
                            time.sleep(2)
                            if process.poll() is None:
                                process.kill()
                            break


                        last_output_time = time.time()

                    time.sleep(0.5)


            return_code = process.wait(timeout=10)
            logger.info("=== LLM Judge subprocess finished with code %d ===", return_code)


            if return_code != 0 and not graceful_termination:
                raise RuntimeError(f"LLM Judge failed with exit code {return_code}")

        except subprocess.TimeoutExpired:
            logger.error("LLM Judge timed out after 1 hour")
            process.kill()
            raise RuntimeError("LLM Judge timed out")
        except Exception as e:
            logger.error("LLM Judge failed: %s", e)
            raise
        finally:

            eval_records_file.unlink(missing_ok=True)


        if not judge_output.exists():
            raise RuntimeError("LLM Judge did not produce output file")

        with open(judge_output, 'r') as f:
            results = json.load(f)

        eval_data = results['eval_data']
        judge_stats = results['judge_stats']


        judge_output.unlink(missing_ok=True)


        logger.info("=" * 70)
        logger.info("LLM Judge Statistics:")
        logger.info("  Total: %d", judge_stats.get('total', 0))
        logger.info("  Correct: %d", judge_stats.get('correct', 0))
        logger.info("  Wrong: %d", judge_stats.get('wrong', 0))
        logger.info("  Unknown: %d", judge_stats.get('unknown', 0))
        logger.info("  Accuracy: %.2f%%", judge_stats.get('accuracy', 0.0) * 100)
        logger.info("=" * 70)


        wrong_records: List[ProblemRecord] = []
        stats: Dict[int, Dict[str, float]] = {}

        for record, eval_item in zip(eval_records, eval_data):

            record.last_prediction = eval_item.get('model_prediction', '')


            total = stats.setdefault(record.category_id, {"total": 0, "wrong": 0})
            total["total"] += 1


            matched = eval_item.get('matched')
            if matched != True:
                total["wrong"] += 1
                wrong_records.append(record)


        for category_id, value in stats.items():
            total = value.get("total", 1)
            wrong = value.get("wrong", 0)
            value["ratio"] = wrong / total if total else 0.0
            value["category_name"] = self.config.categories.get(category_id, "未知类别")

        logger.info(
            "LLM Judge evaluation summary: %s wrong out of %s examples (%.2f%%)",
            len(wrong_records),
            len(eval_records),
            (len(wrong_records) / len(eval_records)) * 100 if eval_records else 0,
        )


        judge_results_path = predictions_path.parent / "judge_results.json"
        judge_summary_path = predictions_path.parent / "judge_summary.json"

        with judge_results_path.open("w", encoding="utf-8") as fout:
            json.dump(eval_data, fout, ensure_ascii=False, indent=2)
        logger.info("Saved detailed LLM Judge results to %s", judge_results_path)


        summary_data = {
            "overall": judge_stats,
            "by_category": stats,
        }
        with judge_summary_path.open("w", encoding="utf-8") as fout:
            json.dump(summary_data, fout, ensure_ascii=False, indent=2)
        logger.info("Saved LLM Judge summary to %s", judge_summary_path)

        return wrong_records, stats

    def _analyse_predictions_with_rules(
        self,
        predictions_path: Path,
        eval_records: List[ProblemRecord],
    ) -> Tuple[List[ProblemRecord], Dict[int, Dict[str, float]]]:
        predictions: List[Dict[str, str]] = []


        try:
            with predictions_path.open("r", encoding="utf-8") as fin:
                predictions = json.load(fin)
                logger.debug("Loaded %d predictions from JSON file", len(predictions))
        except Exception as exc:
            logger.error("Failed to read predictions from %s: %s", predictions_path, exc)
            logger.warning("Returning empty predictions list")

        if len(predictions) != len(eval_records):
            logger.warning(
                "Mismatch between predictions (%s) and evaluation records (%s)",
                len(predictions),
                len(eval_records),
            )

        wrong_records: List[ProblemRecord] = []
        stats: Dict[int, Dict[str, float]] = {}

        for record, pred in zip(eval_records, predictions):
            predicted_answer = pred.get("predict", "")
            label_answer = pred.get("label", "")

            prediction_text = str(predicted_answer) if predicted_answer is not None else ""
            label_text = str(label_answer) if label_answer is not None else ""

            record.last_prediction = prediction_text

            total = stats.setdefault(record.category_id, {"total": 0, "wrong": 0})
            total["total"] += 1

            if not self._answers_match(label_text, prediction_text):
                total["wrong"] += 1
                wrong_records.append(record)

        for category_id, value in stats.items():
            total = value.get("total", 1)
            wrong = value.get("wrong", 0)
            value["ratio"] = wrong / total if total else 0.0
            value["category_name"] = self.config.categories.get(category_id, "未知类别")


        total_count = len(eval_records)
        correct_count = total_count - len(wrong_records)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        logger.info("=" * 70)
        logger.info("Rule-based Evaluation Statistics:")
        logger.info("  Total: %d", total_count)
        logger.info("  Correct: %d", correct_count)
        logger.info("  Wrong: %d", len(wrong_records))
        logger.info("  Accuracy: %.2f%%", accuracy * 100)
        logger.info("=" * 70)

        logger.info(
            "Rule-based evaluation summary: %s wrong out of %s examples (%.2f%%)",
            len(wrong_records),
            len(eval_records),
            (len(wrong_records) / len(eval_records)) * 100 if eval_records else 0,
        )


        rule_summary_path = predictions_path.parent / "rule_summary.json"
        summary_data = {
            "overall": {
                "total": total_count,
                "correct": correct_count,
                "wrong": len(wrong_records),
                "accuracy": accuracy,
            },
            "by_category": stats,
        }
        with rule_summary_path.open("w", encoding="utf-8") as fout:
            json.dump(summary_data, fout, ensure_ascii=False, indent=2)
        logger.info("Saved rule-based evaluation summary to %s", rule_summary_path)

        return wrong_records, stats

    @staticmethod
    def _extract_boxed_answer(text: str) -> str:








































































































        如果代码执行失败，会调用 Gemini 重新修复代码并重试

        Returns:
            需要删除的问题 ID 列表（图片生成失败的问题）
        image_records = [record for record in records if record.image_code]
        if not image_records:
            return []

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import numpy as np


            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except Exception as exc:
            logger.error("matplotlib/numpy not available for image generation: %s", exc)
            return []


        image_dir = Path("/path/to/user/We-Math2.0-Standard/images_generated")
        image_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating images for %s problems with image_code", len(image_records))

        max_retries = 2
        failed_pids: List[str] = []
        failed_problems: List[Dict] = []

        for record in image_records:

            output_path = image_dir / f"{record.pid}.png"

            current_code = record.image_code
            success = False
            error_history = []

            for attempt in range(max_retries + 1):
                namespace: Dict[str, object] = {"plt": plt, "np": np}
                original_savefig = plt.savefig
                original_fig_savefig = Figure.savefig

                def plt_savefig_override(path: object, *args: object, **kwargs: object) -> object:
                    return original_savefig(output_path, *args, **kwargs)

                def fig_savefig_override(self: Figure, path: object, *args: object, **kwargs: object) -> object:
                    return original_fig_savefig(self, output_path, *args, **kwargs)

                plt.savefig = plt_savefig_override
                Figure.savefig = fig_savefig_override

                try:
                    exec(current_code, namespace, {})
                    if output_path.exists():

                        record.image_path = str(output_path)
                        logger.debug("Generated image for problem %s at %s", record.pid, output_path)
                        success = True
                    else:
                        error_msg = "Code executed but no image file created"
                        error_history.append({
                            "attempt": attempt + 1,
                            "error": error_msg,
                            "code": current_code
                        })
                        logger.warning("Image generation did not create file for problem %s (attempt %d)",
                                     record.pid, attempt + 1)
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {str(exc)}"
                    error_history.append({
                        "attempt": attempt + 1,
                        "error": error_message,
                        "code": current_code
                    })
                    logger.warning("Failed to render image for problem %s (attempt %d/%d): %s",
                                 record.pid, attempt + 1, max_retries + 1, error_message)

                    if output_path.exists():
                        output_path.unlink(missing_ok=True)


                    if attempt < max_retries:
                        logger.info("Asking Gemini to fix the code for problem %s", record.pid)
                        fixed_code = self.gemini_generator.fix_image_code(
                            question=record.question,
                            answer=record.answer,
                            original_code=current_code,
                            error_message=error_message,
                        )

                        if fixed_code:
                            logger.info("Got fixed code from Gemini for problem %s, retrying...", record.pid)
                            current_code = fixed_code

                            record.image_code = fixed_code
                        else:
                            logger.warning("Gemini could not fix the code for problem %s", record.pid)
                            break
                finally:
                    plt.savefig = original_savefig
                    Figure.savefig = original_fig_savefig
                    plt.close("all")

                if success:
                    break


            if not success:
                logger.error("All attempts failed for problem %s, will remove this problem from dataset", record.pid)
                failed_pids.append(record.pid)


                failed_problem = {
                    "id": record.pid,
                    "category_id": record.category_id,
                    "category_name": record.category_name,
                    "question": record.question,
                    "answer": record.answer,
                    "source": record.source,
                    "original_image_code": record.image_code,
                    "error_history": error_history,
                    "total_attempts": len(error_history)
                }
                failed_problems.append(failed_problem)

        if failed_pids:
            logger.warning("Image generation failed for %d problems, they will be removed: %s",
                         len(failed_pids), ", ".join(failed_pids))


            self._save_failed_problems(dataset_dir, failed_problems)

        logger.info("Image generation completed: %d succeeded, %d failed",
                   len(image_records) - len(failed_pids), len(failed_pids))

        return failed_pids

    def _save_failed_problems(self, dataset_dir: Path, failed_problems: List[Dict]) -> None:
        if not failed_problems:
            return

        failed_file = dataset_dir / "failed_problems.json"


        existing_failed = []
        if failed_file.exists():
            try:
                with failed_file.open("r", encoding="utf-8") as f:
                    existing_failed = json.load(f)
            except Exception as exc:
                logger.warning("Could not load existing failed problems: %s", exc)


        all_failed = existing_failed + failed_problems


        try:
            with failed_file.open("w", encoding="utf-8") as f:
                json.dump(all_failed, f, ensure_ascii=False, indent=2)
            logger.info("Saved %d failed problems to %s", len(failed_problems), failed_file)
        except Exception as exc:
            logger.error("Failed to save failed problems: %s", exc)

    def _quality_check_generated_problems(
        self,
        dataset_dir: Path,
        records: List[ProblemRecord],
        model_dir: Path,
        round_index: int,
    ) -> List[ProblemRecord]:
        使用当前轮训练的模型测试新生成的题目，每题测5次，过滤掉太简单或太难的题目

        过滤规则：
        - 5次全对（太简单）-> 过滤
        - 5次全错（太难）-> 过滤
        - 1-4次正确（难度合适）-> 保留

        注意：直接使用训练后的模型进行测试，通过简单的字符串匹配判断正确性

        Args:
            dataset_dir: 数据集目录
            records: 新生成的问题记录列表
            model_dir: 当前轮训练的模型目录
            round_index: 当前轮次

        Returns:
            通过质量检测的问题列表
        if not records:
            return []

        logger.info("Starting quality check for %d newly generated problems", len(records))


        quality_check_dir = dataset_dir / "quality_check"
        quality_check_dir.mkdir(exist_ok=True)

        test_data = []
        pid_to_record = {}

        for record in records:
            pid_to_record[record.pid] = record


            for attempt in range(1, 6):
                test_id = f"{record.pid}_attempt{attempt}"


                image_path_list = [record.image_path] if record.image_path else []


                if record.pid.isdigit():
                    id_value = int(record.pid) * 10 + attempt
                else:

                    id_value = (hash(record.pid) % (10**8)) * 10 + attempt


                question_text = record.question
                if image_path_list and "<image>" not in question_text:

                    question_text = "<image> " + question_text
                elif not image_path_list and "<image>" in question_text:

                    question_text = question_text.replace("<image>", "").strip()


                test_data.append({
                    "id": id_value,
                    "problem_id": record.pid,
                    "problem": question_text,
                    "answer": record.answer,
                    "image_path": image_path_list,
                    "category": record.category_id,
                    "category_name": record.category_name,


                    "_metadata": {
                        "test_id": test_id,
                        "original_pid": record.pid,
                        "attempt": attempt,
                    }
                })


        test_file = quality_check_dir / "quality_check_test.json"
        with test_file.open("w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        logger.info("Created quality check dataset with %d test instances (%d problems × 5 attempts)",
                   len(test_data), len(records))


        dataset_info = {
            "quality_check_test": {
                "file_name": test_file.name,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "problem",
                    "response": "answer",
                    "images": "image_path",
                    "system": "system"
                },
            }
        }

        dataset_info_file = quality_check_dir / "dataset_info.json"
        with dataset_info_file.open("w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)


        eval_yaml_path = quality_check_dir / f"quality_check_round_{round_index}.yaml"
        eval_config = {
            "model_name_or_path": str(model_dir),
            "stage": "sft",
            "do_predict": True,
            "finetuning_type": "lora",
            "dataset_dir": str(quality_check_dir),
            "dataset": "quality_check_test",
            "eval_dataset": "quality_check_test",
            "template": "qwen2_vl",
            "cutoff_len": 32000,
            "max_samples": len(test_data),
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "output_dir": str(quality_check_dir / "eval_output"),
            "per_device_eval_batch_size": 1,
            "predict_with_generate": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "max_new_tokens": 8192,
            "bf16": True,
            "ddp_timeout": 180000000,
            "val_size": 0.0,
            "eval_num_beams": 1,
            "vllm_maxlen": 32000,
        }

        with eval_yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(eval_config, f, allow_unicode=True, default_flow_style=False)

        logger.info("Created quality check evaluation config: %s", eval_yaml_path)


        logger.info("Running quality check evaluation with vLLM (this may take 2-5 minutes)...")


        predictions_path = quality_check_dir / "eval_output" / "generated_predictions.json"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        llm_generator_script = Path(__file__).parent.parent / "processors" / "llmasgenerator.py"


        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "false").lower() == "true"
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")

        if use_apptainer and apptainer_image:
            vllm_cmd = [
                "apptainer", "exec", "--nv",
                "--cleanenv",
                "--bind", "/share:/share,/mnt:/mnt",
                "--env", "TRITON_CACHE_DIR=/tmp/triton_cache",
                "--env", "PATH=/opt/py312/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                "--env", "CC=/usr/bin/gcc",
                "--env", "CXX=/usr/bin/g++",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--env", f"PYTHONPATH={Path(__file__).parent.parent.parent}",
                apptainer_image,
                "python", str(llm_generator_script),
                "--input", str(test_file),
                "--output", str(predictions_path),
                "--model-path", str(model_dir),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--batch-size", "8",
                "--temperature", "0.6",
                "--top-p", "0.9",
                "--max-tokens", "16384",
            ]
        else:
            vllm_cmd = [
                "python", str(llm_generator_script),
                "--input", str(test_file),
                "--output", str(predictions_path),
                "--model-path", str(model_dir),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--batch-size", "8",
                "--temperature", "0.6",
                "--top-p", "0.9",
                "--max-tokens", "16384",
            ]

        if hasattr(self.config, 'quality_check_tensor_parallel_size'):
            vllm_cmd.extend(["--tensor-parallel-size", str(self.config.quality_check_tensor_parallel_size)])
        elif hasattr(self.config, 'eval_tensor_parallel_size'):
            vllm_cmd.extend(["--tensor-parallel-size", str(self.config.eval_tensor_parallel_size)])

        logger.info("Running vLLM quality check: %s", " ".join(map(str, vllm_cmd)))

        try:
            result = subprocess.run(
                vllm_cmd,
                check=True,
                stdout=None,
                stderr=None,
                text=True,
            )
            logger.info("✅ vLLM quality check completed successfully")
        except subprocess.CalledProcessError as exc:
            logger.error("❌ vLLM quality check failed with exit code %d", exc.returncode)
            logger.warning("Quality check evaluation failed, keeping all problems")
            return records


        if not predictions_path.exists():
            logger.warning("Quality check predictions not found, keeping all problems")
            return records


        predictions = []
        try:
            with predictions_path.open("r", encoding="utf-8") as f:
                predictions = json.load(f)
                logger.debug("Loaded %d predictions from JSON file", len(predictions))
        except Exception as exc:
            logger.error("Failed to read quality check predictions: %s", exc)

            logger.warning("Quality check predictions read failed, keeping all problems")
            return records


        logger.info("Evaluating predictions with LLM Judge...")


        eval_records_for_judge = []
        for record in records:
            eval_records_for_judge.append({
                "pid": record.pid,
                "question": record.question,
                "answer": record.answer,
                "category_id": record.category_id,
                "category_name": record.category_name,
            })


        eval_records_file = quality_check_dir / "eval_records.json"
        with eval_records_file.open("w", encoding="utf-8") as f:
            json.dump(eval_records_for_judge, f, ensure_ascii=False, indent=2)


        judge_output_file = quality_check_dir / "judge_results.json"
        judge_config = {
            "model_path": self.config.judge_model_path,
            "tensor_parallel_size": self.config.judge_tensor_parallel_size,
            "gpu_memory_utilization": self.config.judge_gpu_memory_utilization,
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
            "max_model_len": self.config.judge_max_model_len,
        }

        judge_cmd = [
            "python",
            str(Path(__file__).parent.parent / "processors" / "run_llm_judge.py"),
            str(eval_records_file),
            str(predictions_path),
            str(judge_output_file),
            json.dumps(judge_config),
        ]

        logger.info("Running LLM Judge for quality check (this may take 5-15 minutes)...")
        judge_success = self._run_llamafactory_command(judge_cmd)

        if not judge_success or not judge_output_file.exists():
            logger.warning("LLM Judge failed for quality check, keeping all problems")
            return records


        try:
            with judge_output_file.open("r", encoding="utf-8") as f:
                judge_results = json.load(f)
            eval_data = judge_results["eval_data"]
            judge_stats = judge_results["judge_stats"]

            logger.info("LLM Judge stats: %s", judge_stats)
        except Exception as exc:
            logger.error("Failed to read Judge results: %s", exc)
            return records


        pid_to_test_items = {}
        for test_item in test_data:
            original_pid = test_item["_metadata"]["original_pid"]
            attempt = test_item["_metadata"]["attempt"]
            if original_pid not in pid_to_test_items:
                pid_to_test_items[original_pid] = {}
            pid_to_test_items[original_pid][attempt] = test_item


        pid_attempts = {}

        for eval_item in eval_data:

            original_pid = eval_item.get("pid", "")
            attempt = eval_item.get("attempt", 1)

            if not original_pid:
                logger.warning("Eval item missing pid field")
                continue

            if original_pid not in pid_to_record:
                logger.warning("Original pid %s not found in records", original_pid)
                continue

            record = pid_to_record[original_pid]


            is_correct = eval_item.get("matched", None)

            if is_correct is None:
                logger.warning("Problem %s (attempt %d) has no judge result (matched field)", original_pid, attempt)
                is_correct = False

            if original_pid not in pid_attempts:
                pid_attempts[original_pid] = []

            pid_attempts[original_pid].append({
                "attempt": attempt,
                "prediction": eval_item.get("model_prediction", ""),
                "label": record.answer,
                "correct": is_correct,
                "judge_analysis": eval_item.get("match_analysis", ""),
            })


        qualified_records = []
        filtered_too_easy = []
        filtered_too_hard = []
        upgraded_problems = []

        for record in records:
            attempts = pid_attempts.get(record.pid, [])

            if len(attempts) < 5:
                logger.warning("Problem %s has only %d attempts, keeping it", record.pid, len(attempts))
                qualified_records.append(record)
                continue

            correct_count = sum(1 for a in attempts if a["correct"])

            if correct_count == 0:

                filtered_too_hard.append(record.pid)
                logger.info("❌ Filtered (too hard): %s - 0/5 correct", record.pid)
            elif correct_count == 5:

                filtered_too_easy.append(record.pid)
                logger.info("⚠️  Too easy: %s - 5/5 correct, will upgrade difficulty", record.pid)


                upgraded_problems.append(record)
            else:

                qualified_records.append(record)
                logger.info("✅ Qualified: %s - %d/5 correct", record.pid, correct_count)


        max_upgrade_iterations = 3
        upgrade_iteration = 0

        while upgraded_problems and upgrade_iteration < max_upgrade_iterations:
            upgrade_iteration += 1
            logger.info("\n🔄 Upgrade iteration %d: Processing %d problems that were too easy...",
                       upgrade_iteration, len(upgraded_problems))


            upgraded_records = self._upgrade_problem_difficulty(
                upgraded_problems, dataset_dir, round_index
            )

            if not upgraded_records:
                logger.warning("Failed to upgrade any problems in iteration %d", upgrade_iteration)
                break

            logger.info("✅ Successfully upgraded %d problems", len(upgraded_records))


            logger.info("Generating images for upgraded problems...")
            failed_pids = self._materialise_problem_images(dataset_dir, upgraded_records)
            if failed_pids:
                logger.warning("%d upgraded problems failed image generation", len(failed_pids))

                upgraded_records = [r for r in upgraded_records if r.pid not in failed_pids]

            if not upgraded_records:
                logger.warning("All upgraded problems failed image generation")
                break


            logger.info("Generating COT answers for upgraded problems...")
            self._generate_model_answers(dataset_dir, upgraded_records)


            if upgrade_iteration < max_upgrade_iterations:
                logger.info("Testing upgraded problems to check if they are still too easy...")


                still_too_easy = self._test_upgraded_problems(
                    dataset_dir, upgraded_records, model_dir, round_index, quality_check_dir
                )

                if still_too_easy:
                    logger.info("⚠️  %d upgraded problems are still too easy, will upgrade again",
                               len(still_too_easy))

                    upgraded_problems = still_too_easy

                    qualified_upgraded = [r for r in upgraded_records if r not in still_too_easy]
                    if qualified_upgraded:
                        qualified_records.extend(qualified_upgraded)
                        logger.info("Added %d qualified upgraded problems to training set",
                                   len(qualified_upgraded))
                else:
                    logger.info("✅ All upgraded problems now have appropriate difficulty")
                    qualified_records.extend(upgraded_records)
                    logger.info("Added %d upgraded problems to qualified list", len(upgraded_records))
                    break
            else:

                logger.info("Reached max upgrade iterations (%d), accepting all upgraded problems",
                           max_upgrade_iterations)
                qualified_records.extend(upgraded_records)
                logger.info("Added %d upgraded problems to qualified list", len(upgraded_records))
                break


        quality_report = {
            "round": round_index,
            "total_problems": len(records),
            "qualified_problems": len(qualified_records),
            "filtered_too_easy": len(filtered_too_easy),
            "filtered_too_hard": len(filtered_too_hard),
            "filtered_too_easy_pids": filtered_too_easy,
            "filtered_too_hard_pids": filtered_too_hard,
            "judge_stats": judge_stats if 'judge_stats' in locals() else {},
            "details": {
                pid: {
                    "attempts": attempts,
                    "correct_count": sum(1 for a in attempts if a["correct"]),
                    "qualified": pid in [r.pid for r in qualified_records],
                }
                for pid, attempts in pid_attempts.items()
            },
        }

        report_file = quality_check_dir / f"quality_report_round_{round_index}.json"
        with report_file.open("w", encoding="utf-8") as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)

        logger.info("Quality check report saved to: %s", report_file)
        logger.info("Quality check summary:")
        logger.info("  Total: %d problems", len(records))
        logger.info("  Qualified: %d problems (1-4/5 correct)", len(qualified_records))
        logger.info("  Filtered (too easy): %d problems (5/5 correct)", len(filtered_too_easy))
        logger.info("  Filtered (too hard): %d problems (0/5 correct)", len(filtered_too_hard))

        return qualified_records

    def _test_upgraded_problems(
        self,
        dataset_dir: Path,
        upgraded_records: List[ProblemRecord],
        model_dir: Path,
        round_index: int,
        quality_check_dir: Path,
    ) -> List[ProblemRecord]:
        测试升级后的题目是否仍然太简单（使用 vLLM）

        Args:
            dataset_dir: 数据集目录
            upgraded_records: 升级后的题目列表
            model_dir: 当前轮训练后的模型目录
            round_index: 当前轮次
            quality_check_dir: 质量检测目录

        Returns:
            仍然太简单的题目列表（5/5 正确）
        logger.info("Testing %d upgraded problems with current model (5 attempts each) using vLLM...",
                   len(upgraded_records))


        test_data = []
        pid_to_record = {record.pid: record for record in upgraded_records}

        for record in upgraded_records:
            for attempt in range(1, 6):
                test_id = f"{record.pid}_attempt{attempt}"


                image_path_list = []
                if record.image_path:
                    img_path = Path(record.image_path)
                    if not img_path.is_absolute():
                        img_path = dataset_dir / record.image_path
                    if img_path.exists():
                        image_path_list = [str(img_path)]


                question_text = record.question
                if image_path_list and "<image>" not in question_text:
                    question_text = "<image> " + question_text
                elif not image_path_list and "<image>" in question_text:
                    question_text = question_text.replace("<image>", "").strip()

                test_data.append({
                    "id": hash(test_id) % (10**8),
                    "problem_id": record.pid,
                    "problem": question_text,
                    "answer": record.answer,
                    "image_path": image_path_list,
                    "category": record.category_id,
                    "category_name": record.category_name,
                    "_metadata": {
                        "test_id": test_id,
                        "original_pid": record.pid,
                        "attempt": attempt,
                    }
                })


        test_file = quality_check_dir / f"upgraded_test_iteration.json"
        with test_file.open("w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        logger.info("Saved upgraded test data: %s (%d instances)", test_file, len(test_data))


        predictions_path = quality_check_dir / "upgraded_eval_output" / "generated_predictions.json"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

        llm_generator_script = Path(__file__).parent.parent / "processors" / "llmasgenerator.py"


        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "false").lower() == "true"
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")

        if use_apptainer and apptainer_image:
            vllm_cmd = [
                "apptainer", "exec", "--nv",
                "--cleanenv",
                "--bind", "/share:/share,/mnt:/mnt",
                "--env", "TRITON_CACHE_DIR=/tmp/triton_cache",
                "--env", "PATH=/opt/py312/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                "--env", "CC=/usr/bin/gcc",
                "--env", "CXX=/usr/bin/g++",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--env", f"PYTHONPATH={Path(__file__).parent.parent.parent}",
                apptainer_image,
                "python", str(llm_generator_script),
                "--input", str(test_file),
                "--output", str(predictions_path),
                "--model-path", str(model_dir),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--batch-size", "8",
                "--temperature", "0.6",
                "--top-p", "0.9",
                "--max-tokens", "16384",
            ]
        else:
            vllm_cmd = [
                "python", str(llm_generator_script),
                "--input", str(test_file),
                "--output", str(predictions_path),
                "--model-path", str(model_dir),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--batch-size", "8",
                "--temperature", "0.6",
                "--top-p", "0.9",
                "--max-tokens", "16384",
            ]

        if hasattr(self.config, 'quality_check_tensor_parallel_size'):
            vllm_cmd.extend(["--tensor-parallel-size", str(self.config.quality_check_tensor_parallel_size)])
        elif hasattr(self.config, 'eval_tensor_parallel_size'):
            vllm_cmd.extend(["--tensor-parallel-size", str(self.config.eval_tensor_parallel_size)])

        logger.info("Running vLLM for upgraded test: %s", " ".join(map(str, vllm_cmd)))

        try:
            result = subprocess.run(
                vllm_cmd,
                check=True,
                stdout=None,
                stderr=None,
                text=True,
            )
            logger.info("✅ vLLM upgraded test completed successfully")
        except subprocess.CalledProcessError as exc:
            logger.error("❌ vLLM upgraded test failed with exit code %d", exc.returncode)
            logger.warning("Upgraded problems evaluation failed, assuming all are qualified")
            return []


        if not predictions_path.exists():
            logger.warning("Upgraded test predictions not found, assuming all are qualified")
            return []


        eval_records_for_judge = []
        for record in upgraded_records:
            eval_records_for_judge.append({
                "pid": record.pid,
                "question": record.question,
                "answer": record.answer,
                "category_id": record.category_id,
                "category_name": record.category_name,
            })

        eval_records_file = quality_check_dir / "upgraded_eval_records.json"
        with eval_records_file.open("w", encoding="utf-8") as f:
            json.dump(eval_records_for_judge, f, ensure_ascii=False, indent=2)

        judge_output_file = quality_check_dir / "upgraded_judge_results.json"
        judge_config = {
            "model_path": self.config.judge_model_path,
            "tensor_parallel_size": self.config.judge_tensor_parallel_size,
            "gpu_memory_utilization": self.config.judge_gpu_memory_utilization,
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
            "max_model_len": self.config.judge_max_model_len,
        }

        judge_cmd = [
            "python",
            str(Path(__file__).parent.parent / "processors" / "run_llm_judge.py"),
            str(eval_records_file),
            str(predictions_path),
            str(judge_output_file),
            json.dumps(judge_config),
        ]

        logger.info("Running LLM Judge on upgraded problems...")
        judge_success = self._run_llamafactory_command(judge_cmd)

        if not judge_success or not judge_output_file.exists():
            logger.warning("LLM Judge failed for upgraded test, assuming all are qualified")
            return []


        try:
            with judge_output_file.open("r", encoding="utf-8") as f:
                judge_results = json.load(f)
            eval_data = judge_results["eval_data"]
        except Exception as exc:
            logger.error("Failed to read upgraded Judge results: %s", exc)
            return []


        index_to_info = {}
        for idx, test_item in enumerate(test_data):
            index_to_info[idx] = {
                "pid": test_item["_metadata"]["original_pid"],
                "attempt": test_item["_metadata"]["attempt"],
            }


        pid_correct_counts = {}

        for idx, eval_item in enumerate(eval_data):
            if idx not in index_to_info:
                continue

            info = index_to_info[idx]
            original_pid = info["pid"]


            is_correct = eval_item.get("matched", False)

            if original_pid not in pid_correct_counts:
                pid_correct_counts[original_pid] = 0

            if is_correct:
                pid_correct_counts[original_pid] += 1


        still_too_easy = []
        for record in upgraded_records:
            correct_count = pid_correct_counts.get(record.pid, 0)
            if correct_count == 5:
                logger.info("⚠️  Upgraded problem %s is still too easy: 5/5 correct", record.pid)
                still_too_easy.append(record)
            else:
                logger.info("✅ Upgraded problem %s has appropriate difficulty: %d/5 correct",
                          record.pid, correct_count)

        return still_too_easy

    def _upgrade_problem_difficulty(
        self,
        easy_problems: List[ProblemRecord],
        dataset_dir: Path,
        round_index: int,
    ) -> List[ProblemRecord]:
        对太简单的题目进行难度升级

        根据题目的 source 判断类型（reasoning 或 visual），
        调用 Gemini 生成难度升级版本

        Args:
            easy_problems: 太简单的题目列表
            dataset_dir: 数据集目录
            round_index: 当前轮次

        Returns:
            升级后的题目列表
        upgraded_records = []
        upgrade_details = []

        logger.info("Starting difficulty upgrade for %d problems", len(easy_problems))

        for record in easy_problems:
            try:

                if "reasoning" in record.source.lower():
                    difficulty_aspect = "reasoning"
                elif "visual" in record.source.lower():
                    difficulty_aspect = "visual"
                else:

                    difficulty_aspect = "reasoning"
                    logger.warning("Cannot determine difficulty type for %s (source: %s), defaulting to reasoning",
                                 record.pid, record.source)

                logger.info("Upgrading problem %s (%s difficulty)", record.pid, difficulty_aspect)


                upgraded = self.gemini_generator.upgrade_problem_difficulty(
                    problem=record.question,
                    answer=record.answer,
                    image_path=record.image_path,
                    category_id=record.category_id,
                    category_name=record.category_name,
                    difficulty_aspect=difficulty_aspect,
                )

                if not upgraded:
                    logger.warning("Failed to upgrade problem %s", record.pid)
                    continue


                new_id = self.next_generated_id
                self.next_generated_id += 1
                new_pid = str(new_id)


                new_record = ProblemRecord(
                    pid=new_pid,
                    category_id=record.category_id,
                    category_name=record.category_name,
                    question=upgraded.get("question", "").strip(),
                    answer=upgraded.get("answer", "").strip(),
                    image_path="",
                    source=f"{record.source}_upgraded",
                    image_code=upgraded.get("image_code", ""),
                )

                upgraded_records.append(new_record)


                upgrade_details.append({
                    "original_pid": record.pid,
                    "new_pid": new_pid,
                    "difficulty_aspect": difficulty_aspect,
                    "original_question": record.question,
                    "new_question": new_record.question,
                    "original_answer": record.answer,
                    "new_answer": new_record.answer,
                })

                logger.info("✅ Upgraded problem %s → %s (%s)",
                          record.pid, new_pid, difficulty_aspect)

            except Exception as exc:
                logger.error("Failed to upgrade problem %s: %s", record.pid, exc)


        if upgrade_details:
            upgrade_file = dataset_dir / f"upgraded_problems_round_{round_index}.json"
            try:
                with upgrade_file.open("w", encoding="utf-8") as f:
                    json.dump({
                        "round": round_index,
                        "total_upgraded": len(upgrade_details),
                        "upgrades": upgrade_details
                    }, f, ensure_ascii=False, indent=2)
                logger.info("Saved upgrade details to: %s", upgrade_file)
            except Exception as exc:
                logger.error("Failed to save upgrade details: %s", exc)

        return upgraded_records

    def _check_answer_correctness(self, prediction: str, ground_truth: str) -> bool:
        检查预测答案是否正确（简单的字符串匹配）

        从预测的 COT 答案中提取最终答案，与标准答案对比

        Args:
            prediction: 模型的预测结果（COT格式）
            ground_truth: 标准答案

        Returns:
            是否正确
        import re


        def normalize(text):
            if not text:
                return ""

            text = " ".join(text.split())

            text = text.lower()
            return text

        pred_normalized = normalize(prediction)
        gt_normalized = normalize(ground_truth)


        if gt_normalized in pred_normalized:
            return True


        answer_patterns = [
            r"答案是[：:]\s*(.+?)(?:\n|$|。)",
            r"答案是\s*(.+?)(?:\n|$|。)",
            r"answer\s*is[：:]\s*(.+?)(?:\n|$|\.|,)",
            r"answer[：:]\s*(.+?)(?:\n|$|\.|,)",
            r"因此[，,]\s*(.+?)(?:\n|$|。)",
            r"所以[，,]\s*(.+?)(?:\n|$|。)",
            r"最终答案[为是][：:]\s*(.+?)(?:\n|$|。)",
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, pred_normalized, re.IGNORECASE)
            if match:
                extracted_answer = normalize(match.group(1).strip())

                if gt_normalized in extracted_answer or extracted_answer in gt_normalized:
                    return True

                if extracted_answer == gt_normalized:
                    return True


        def extract_numbers(text):

            numbers = re.findall(r'-?\d+\.?\d*', text)
            return [float(n) for n in numbers if n]

        pred_numbers = extract_numbers(pred_normalized)
        gt_numbers = extract_numbers(gt_normalized)

        if pred_numbers and gt_numbers:

            for gt_num in gt_numbers:
                if any(abs(pred_num - gt_num) < 1e-6 for pred_num in pred_numbers):
                    return True


        return False

    def _check_answer_consistency(self, cot_answer: str, gemini_answer: str) -> bool:
        检查 COT 答案和 Gemini 答案是否一致

        使用 llmasjudge 的提取函数来提取和比较答案：
        1. 从 COT 答案提取最终答案（\boxed{}, <answer>, "answer is" 等）
        2. 从 Gemini 答案提取最终答案（支持 Gemini 特有的格式）
        3. 比较两者是否一致

        Args:
            cot_answer: Qwen3-VL-30B 生成的 COT 答案（包含 \boxed{}）
            gemini_answer: Gemini 生成的简短答案

        Returns:
            True 如果答案一致，False 如果不一致
        import re


        try:
            from ..processors.llmasjudge import extract_solution_text
        except ImportError:

            logger.warning("Failed to import extract_solution_text from llmasjudge, using fallback method")
            return self._check_answer_correctness(cot_answer, gemini_answer)


        def normalize(text):
            if not text:
                return ""
            text = " ".join(text.split())
            text = text.lower()

            text = re.sub(r'\*\*', '', text)
            return text


        cot_extracted = extract_solution_text(cot_answer)
        cot_normalized = normalize(cot_extracted) if cot_extracted else ""


        gemini_extracted = extract_solution_text(gemini_answer)
        gemini_normalized = normalize(gemini_extracted) if gemini_extracted else ""



        if len(gemini_normalized) > 100:

            gemini_specific_patterns = [
                r'therefore[,\s]+(?:only\s+)?(?:shape\s+|option\s+)?([a-z0-9]+)\s+is',
                r'thus[,\s]+(?:only\s+)?(?:shape\s+|option\s+)?([a-z0-9]+)\s+(?:is|corresponds)',
                r'only\s+(?:shape\s+|option\s+)?([a-z0-9]+)\s+(?:is|can)',
                r'(?:shape|option)\s+([a-z0-9]+)\s+is\s+(?:the\s+)?correct',
                r'答案(?:是|为)[：:\s]*([a-z0-9]+)',
                r'选项?\s*([a-z0-9]+)\s*(?:是|为)(?:正确|答案)',
            ]

            for pattern in gemini_specific_patterns:
                match = re.search(pattern, gemini_normalized, re.IGNORECASE)
                if match:
                    gemini_normalized = match.group(1).lower()
                    logger.debug("Extracted from Gemini using specific pattern: '%s'", gemini_normalized)
                    break

        logger.debug("Extracted answers - COT: '%s', Gemini: '%s'",
                    cot_normalized[:50], gemini_normalized[:50])


        if cot_normalized and gemini_normalized:

            if cot_normalized == gemini_normalized:
                return True


            if cot_normalized in gemini_normalized or gemini_normalized in cot_normalized:
                return True


            if len(cot_normalized) <= 5 and len(gemini_normalized) <= 20:

                pattern = rf'\b{re.escape(cot_normalized)}\b'
                if re.search(pattern, gemini_normalized, re.IGNORECASE):
                    return True


        if cot_normalized and len(cot_normalized) <= 10:

            gemini_full_normalized = normalize(gemini_answer)
            pattern = rf'\b{re.escape(cot_normalized)}\b'
            if re.search(pattern, gemini_full_normalized, re.IGNORECASE):
                return True


        return self._check_answer_correctness(cot_answer, gemini_answer)

    def _save_inconsistent_answers(self, dataset_dir: Path, inconsistent_details: List[Dict]) -> None:







        Args:
            dataset_dir: 数据集目录
            records: 问题记录列表

        generated_records = [
            r for r in records
            if (r.source.startswith("generated_hard") or r.source.startswith("generated_refresh")) and r.image_path
        ]

        logger.info("Generating COT answers for %d newly generated problems using Qwen3-VL-30B-A3B-Thinking",
                   len(generated_records))


        temp_input_path = dataset_dir / "new_problems_wo_cot.json"


        temp_data = []
        for record in generated_records:
            temp_data.append({
                "id": record.pid,
                "problem": record.question,
                "answer": record.answer,
                "image_path": record.image_path,
                "category_id": record.category_id,
            })


        with temp_input_path.open("w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)

        logger.info("Saved temporary input file for LLM generation: %s", temp_input_path)

        output_path = dataset_dir / "new_problems_with_cot.json"


        llm_generator_script = Path(__file__).parent.parent / "processors" / "llmasgenerator.py"


        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "false").lower() == "true"
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")
        if use_apptainer and apptainer_image:

            cmd = [
                "apptainer", "exec", "--nv",
                "--cleanenv",
                "--bind", "/share:/share,/mnt:/mnt",
                "--env", "TRITON_CACHE_DIR=/tmp/triton_cache",
                "--env", "PATH=/opt/py312/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                "--env", "CC=/usr/bin/gcc",
                "--env", "CXX=/usr/bin/g++",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--env", "PYTHONPATH=/path/to/models",
                apptainer_image,
                "python", str(llm_generator_script),
                "--input", str(temp_input_path),
                "--output", str(output_path),
                "--model-path", self.config.llm_generator_model_path,
                "--question-key", "problem",
                "--answer-key", "model_answer",
                "--image-key", "image_path",
                "--batch-size", "1",
                "--temperature", "0.6",
                "--top-p", "0.9",
                "--max-tokens", "32000",
            ]
        else:
            cmd = [
                "python", str(llm_generator_script),
                "--input", str(temp_input_path),
                "--output", str(output_path),
                "--model-path", self.config.llm_generator_model_path,
                "--question-key", "problem",
                "--answer-key", "model_answer",
                "--image-key", "image_path",
                "--batch-size", "1",
                "--temperature", "0.6",
                "--top-p", "0.9",
                "--max-tokens", "32000",
            ]

        if hasattr(self.config, 'llm_generator_tensor_parallel_size') and self.config.llm_generator_tensor_parallel_size:
            cmd.extend(["--tensor-parallel-size", str(self.config.llm_generator_tensor_parallel_size)])

        logger.info("Running LLM generator command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=None,
                stderr=None,
                text=True,
            )
            logger.info("✅ LLM generator completed successfully")
        except subprocess.CalledProcessError as exc:
            logger.error("❌ LLM generator failed with exit code %d", exc.returncode)
            temp_input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise RuntimeError("LLM generator failed") from exc


        try:
            with output_path.open("r", encoding="utf-8") as f:
                generated_data = json.load(f)


            pid_to_cot_answer = {
                item["id"]: item.get("model_answer", "")
                for item in generated_data
            }


            updated_count = 0
            inconsistent_pids = []
            inconsistent_details = []

            for record in generated_records:
                if record.pid in pid_to_cot_answer and pid_to_cot_answer[record.pid]:
                    cot_answer = pid_to_cot_answer[record.pid]
                    gemini_answer = record.answer


                    is_consistent = self._check_answer_consistency(cot_answer, gemini_answer)

                    if not is_consistent:
                        logger.warning("Answer inconsistency detected for problem %s", record.pid)
                        logger.warning("  Gemini answer: %s", gemini_answer[:100])
                        logger.warning("  COT answer: %s", cot_answer[:100])
                        inconsistent_pids.append(record.pid)
                        inconsistent_details.append({
                            "pid": record.pid,
                            "question": record.question[:200],
                            "gemini_answer": gemini_answer,
                            "cot_answer": cot_answer,
                            "category_id": record.category_id,
                            "category_name": record.category_name,
                        })

                        continue


                    record.answer = cot_answer
                    record.model_answer = cot_answer
                    updated_count += 1
                    logger.debug("Replaced answer for problem %s with COT answer (answers consistent)", record.pid)

            logger.info("Updated %d records: replaced short answers with COT answers", updated_count)

            if inconsistent_pids:
                logger.warning("Found %d problems with inconsistent answers between Gemini and COT",
                             len(inconsistent_pids))
                logger.warning("These problems will be filtered out: %s", ", ".join(inconsistent_pids))


                self._save_inconsistent_answers(dataset_dir, inconsistent_details)


                original_count = len(records)
                records[:] = [r for r in records if r.pid not in inconsistent_pids]
                logger.info("Removed %d problems with inconsistent answers from dataset",
                          original_count - len(records))

        except Exception as exc:
            logger.error("Failed to read or process generated COT answers: %s", exc)
            raise
        finally:

            temp_input_path.unlink(missing_ok=True)
