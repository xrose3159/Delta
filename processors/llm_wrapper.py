import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import os


logger = logging.getLogger(__name__)


class LLMasGenerator:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_predictions(
        self,
        test_file: Path,
        output_file: Path,
        model_path: Optional[str] = None,
        max_tokens: Optional[int] = None,
        disable_thinking: bool = False,
        is_error_analysis: bool = False,
        is_correction: bool = False,
        is_validation: bool = False
    ) -> bool:
        使用 vLLM 生成预测

        Args:
            test_file: 输入文件路径
            output_file: 输出文件路径
            model_path: 模型路径（可选，默认使用配置中的路径）
            max_tokens: 最大生成 token 数（可选，默认使用配置中的 eval_max_tokens）
            disable_thinking: 是否禁用思考模式
            is_error_analysis: 是否为错误分析任务（使用专门的 system prompt）
            is_correction: 是否为生成 corrected CoT 任务（使用专门的 system prompt）
            is_validation: 是否为答案验证任务（使用与错误分析相同的配置）

        Returns:
            是否成功
        if model_path is None:
            model_path = self.config.model_path
        if max_tokens is None:
            max_tokens = self.config.eval_max_tokens


        if is_error_analysis or is_validation:

            tensor_parallel_size = self.config.error_analyzer_tensor_parallel_size
            gpu_memory_utilization = self.config.error_analyzer_gpu_memory_utilization
            max_model_len = self.config.error_analyzer_max_model_len
            if max_tokens is None or max_tokens == self.config.eval_max_tokens:
                max_tokens = self.config.error_analyzer_max_tokens
        else:

            tensor_parallel_size = self.config.llm_generator_tensor_parallel_size
            gpu_memory_utilization = self.config.llm_generator_gpu_memory_utilization
            max_model_len = self.config.llm_generator_max_model_len

        script_path = Path(__file__).parent / "llmasgenerator.py"


        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "").lower() in ("true", "1", "yes")
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")

        if use_apptainer and apptainer_image:
            self.logger.info("Using Apptainer for vLLM generation")


            project_root = Path(__file__).parent.parent.parent


            triton_cache_dir = f"/tmp/triton_cache_{os.getenv('USER', 'default')}_{os.getpid()}"

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
                "--model-path", model_path,
                "--input", str(test_file),
                "--output", str(output_file),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--temperature", str(self.config.eval_temperature),
                "--top-p", str(self.config.eval_top_p),
                "--frequency-penalty", str(self.config.eval_frequency_penalty),
                "--max-tokens", str(max_tokens),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-model-len", str(max_model_len),
            ]

            if disable_thinking:
                cmd.append("--disable-thinking")
            if is_error_analysis:
                cmd.append("--is-error-analysis")
            if is_correction:
                cmd.append("--is-correction")

        else:
            self.logger.info("Using local Python for vLLM generation")
            cmd = [
                "python", "-u", str(script_path),
                "--model-path", model_path,
                "--input", str(test_file),
                "--output", str(output_file),
                "--question-key", "problem",
                "--answer-key", "predict",
                "--image-key", "image_path",
                "--temperature", str(self.config.eval_temperature),
                "--top-p", str(self.config.eval_top_p),
                "--frequency-penalty", str(self.config.eval_frequency_penalty),
                "--max-tokens", str(max_tokens),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-model-len", str(max_model_len),
            ]

            if disable_thinking:
                cmd.append("--disable-thinking")
            if is_error_analysis:
                cmd.append("--is-error-analysis")
            if is_correction:
                cmd.append("--is-correction")

        self.logger.info(f"Running LLM Generator: {' '.join(map(str, cmd[:6]))}...")

        try:

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )


            self.logger.info("=== LLM Generator subprocess output ===")


            important_keywords = [
                'ERROR', 'WARNING', 'CRITICAL', 'Failed', 'Exception',
                'Traceback', '❌', '✅', 'completed', 'Starting', 'Finished'
            ]


            skip_patterns = [
                'Found nccl', 'is using nccl', 'rank', 'TP rank', 'DP rank',
                'Loading safetensors checkpoint shards:', 'Completed |',
                'Using FlashInfer', 'Using Flash Attention', 'Starting to load model',
                'Capturing CUDA graphs',
                'it/s]',
                '% Completed',
                'Model loading took',
                'Loading weights took',
                'torch.compile',
                'Dynamo bytecode',
                'FutureWarning',
                'pynvml package is deprecated',
                'TORCH_CUDA_ARCH_LIST',
                'SymmMemCommunicator',
                'Reducing Torch parallelism',
                'OMP_NUM_THREADS',
                'torch/utils/cpp_extension.py',
                '[1;36m(EngineCore_DP0',
                '[1;36m(Worker_TP'
            ]

            import re
            for line in process.stdout:
                line_stripped = line.rstrip()


                line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                if not line_clean.strip():
                    continue


                should_print = any(keyword in line_stripped for keyword in important_keywords)

                should_skip = any(pattern in line_stripped for pattern in skip_patterns)

                if should_print or not should_skip:
                    self.logger.info(f"  [vLLM] {line_stripped}")


            return_code = process.wait()
            self.logger.info(f"=== LLM Generator subprocess finished with code {return_code} ===")

            if return_code != 0:
                self.logger.error(f"❌ LLM Generator failed with exit code {return_code}")
                return False

            self.logger.info("✅ LLM Generator completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ LLM Generator failed: {e}")
            return False


class LLMasJudge:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(
        self,
        predictions_file: Path,
        eval_records_file: Path,
        output_file: Path
    ) -> bool:
        使用 LLM Judge 评估预测

        Args:
            predictions_file: 预测文件路径
            eval_records_file: 评估记录文件路径
            output_file: 输出文件路径

        Returns:
            是否成功
        script_path = Path(__file__).parent / "run_llm_judge.py"


        judge_config = {
            "model_path": self.config.judge_model_path,
            "tensor_parallel_size": self.config.judge_tensor_parallel_size,
            "gpu_memory_utilization": self.config.judge_gpu_memory_utilization,
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
            "max_model_len": self.config.judge_max_model_len,
        }
        judge_config_json = json.dumps(judge_config)


        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "").lower() in ("true", "1", "yes")
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")

        if use_apptainer and apptainer_image:
            self.logger.info(f"Using Apptainer for Judge: {apptainer_image}")

            cmd = [
                "apptainer", "exec", "--nv",
                "--bind", "/mnt:/mnt",
                "--bind", "/tmp:/tmp",
                "--env", f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
                "--env", "PYTHONPATH=/path/to/models",
                "--env", f"HF_HOME={os.getenv('HF_HOME', '/tmp/huggingface')}",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",

                "--env", "VLLM_LOGGING_LEVEL=WARNING",

                "--env", "CC=/usr/bin/gcc",
                "--env", "CXX=/usr/bin/g++",
                "--env", "CUDAHOSTCXX=/usr/bin/g++",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--pwd", str(Path.cwd()),
                apptainer_image,
                "python3", str(script_path),
                str(eval_records_file),
                str(predictions_file),
                str(output_file),
                judge_config_json
            ]
        else:

            self.logger.info("Using host Python for Judge (Apptainer disabled)")
            cmd = [
                sys.executable, str(script_path),
                str(eval_records_file),
                str(predictions_file),
                str(output_file),
                judge_config_json
            ]

        self.logger.info(f"Running LLM Judge: {' '.join(map(str, cmd[:4]))}...")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )


            self.logger.info("=== LLM Judge subprocess output ===")


            important_keywords = [
                'ERROR', 'WARNING', 'CRITICAL', 'Failed', 'Exception',
                'Traceback', '❌', '✅', 'completed', 'Starting', 'Finished',
                'Evaluating', 'Total problems'
            ]


            skip_patterns = [
                'Found nccl', 'is using nccl', 'rank', 'TP rank', 'DP rank',
                'Loading safetensors checkpoint shards:', 'Completed |',
                'Using FlashInfer', 'Using Flash Attention', 'Starting to load model',
                'Loading model from scratch', 'Using cache directory',
                'Capturing CUDA graphs',
                'it/s]',
                '% Completed',
                'Model loading took',
                'Loading weights took',
                'torch.compile',
                'Dynamo bytecode',
                'FutureWarning',
                'pynvml package is deprecated',
                'TORCH_CUDA_ARCH_LIST',
                'SymmMemCommunicator',
                'Reducing Torch parallelism',
                'OMP_NUM_THREADS',
                'torch/utils/cpp_extension.py',
                '[1;36m(EngineCore_DP0',
                '[1;36m(Worker_TP'
            ]


            import select
            import time
            import re



            timeout_seconds = 1200
            last_output_time = time.time()

            while True:

                if process.poll() is not None:

                    for line in process.stdout:
                        line_stripped = line.rstrip()

                        line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                        if not line_clean.strip():
                            continue
                        should_print = any(keyword in line_stripped for keyword in important_keywords)
                        should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                        if should_print or not should_skip:
                            self.logger.info(f"  [Judge] {line_stripped}")
                    break



                try:
                    if sys.platform != 'win32':

                        readable, _, _ = select.select([process.stdout], [], [], 1.0)
                        if readable:
                            line = process.stdout.readline()
                            if line:
                                line_stripped = line.rstrip()

                                line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                                if line_clean.strip():
                                    should_print = any(keyword in line_stripped for keyword in important_keywords)
                                    should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                                    if should_print or not should_skip:
                                        self.logger.info(f"  [Judge] {line_stripped}")
                                last_output_time = time.time()
                    else:

                        line = process.stdout.readline()
                        if line:
                            line_stripped = line.rstrip()

                            line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                            if line_clean.strip():
                                should_print = any(keyword in line_stripped for keyword in important_keywords)
                                should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                                if should_print or not should_skip:
                                    self.logger.info(f"  [Judge] {line_stripped}")
                            last_output_time = time.time()
                except Exception as read_error:
                    self.logger.warning(f"Error reading subprocess output: {read_error}")


                if time.time() - last_output_time > timeout_seconds:
                    self.logger.error(f"❌ Subprocess timeout: no output for {timeout_seconds} seconds")
                    process.kill()
                    process.wait()
                    return False


            return_code = process.wait()
            self.logger.info(f"=== LLM Judge subprocess finished with code {return_code} ===")

            if return_code != 0:
                self.logger.error(f"❌ LLM Judge failed with exit code {return_code}")
                return False


            if not output_file.exists():
                self.logger.error(f"❌ Output file not found: {output_file}")
                return False

            self.logger.info("✅ LLM Judge completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ LLM Judge failed: {e}")

            try:
                if 'process' in locals() and process.poll() is None:
                    process.kill()
                    process.wait()
            except:
                pass
            return False
