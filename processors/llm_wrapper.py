"""
LLM Generator 和 Judge 的包装类
用于将脚本调用封装成类接口
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import os


logger = logging.getLogger(__name__)


class LLMasGenerator:
    """LLM Generator 包装类"""
    
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
        """
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
        """
        if model_path is None:
            model_path = self.config.model_path
        if max_tokens is None:
            max_tokens = self.config.eval_max_tokens
        
        # 根据任务类型选择配置参数
        if is_error_analysis or is_validation:
            # 错误分析和答案验证使用相同的专用配置（都是使用 Qwen3-VL-30B）
            tensor_parallel_size = self.config.error_analyzer_tensor_parallel_size
            gpu_memory_utilization = self.config.error_analyzer_gpu_memory_utilization
            max_model_len = self.config.error_analyzer_max_model_len
            if max_tokens is None or max_tokens == self.config.eval_max_tokens:
                max_tokens = self.config.error_analyzer_max_tokens
        else:
            # 默认使用 LLM Generator 配置
            tensor_parallel_size = self.config.llm_generator_tensor_parallel_size
            gpu_memory_utilization = self.config.llm_generator_gpu_memory_utilization
            max_model_len = self.config.llm_generator_max_model_len
        
        script_path = Path(__file__).parent / "llmasgenerator.py"
        
        # 检查是否使用 Apptainer
        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "").lower() in ("true", "1", "yes")
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")
        
        if use_apptainer and apptainer_image:
            self.logger.info("Using Apptainer for vLLM generation")
            
            # 获取项目根目录（用于 PYTHONPATH）
            project_root = Path(__file__).parent.parent.parent
            
            # 使用用户专属的 Triton 缓存目录，避免多进程/多用户权限冲突
            triton_cache_dir = f"/tmp/triton_cache_{os.getenv('USER', 'default')}_{os.getpid()}"
            
            cmd = [
                "apptainer", "exec", "--nv",
                "--cleanenv",
                "--bind", "/share:/share,/mnt:/mnt",
                # HuggingFace 离线模式（必须设置，避免路径验证错误）
                "--env", "HF_HUB_OFFLINE=1",
                "--env", "TRANSFORMERS_OFFLINE=1",
                "--env", "HF_DATASETS_OFFLINE=1",
                # Python 输出不缓冲（关键修复）
                "--env", "PYTHONUNBUFFERED=1",
                # CUDA 设备配置（关键：传递 Slurm 分配的 GPU）
                "--env", f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
                # vLLM 日志控制：只显示 WARNING 及以上级别
                "--env", "VLLM_LOGGING_LEVEL=WARNING",
                # Triton 和编译环境：使用用户+进程专属缓存，避免权限冲突
                "--env", f"TRITON_CACHE_DIR={triton_cache_dir}",
                "--env", "PATH=/opt/py312/bin:/usr/local/cuda/bin:/usr/bin:/bin",
                # 关键修复：强制使用容器内的 GCC，避免宿主机 GCC 依赖问题
                "--env", "CC=/usr/bin/gcc",              # C 编译器
                "--env", "CXX=/usr/bin/g++",             # C++ 编译器
                "--env", "CUDAHOSTCXX=/usr/bin/g++",     # CUDA 主机编译器（最关键！）
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                "--env", "CUDA_HOME=/usr/local/cuda",
                "--env", f"PYTHONPATH={project_root}",  # 添加 PYTHONPATH
                apptainer_image,
                "python", "-u", str(script_path),  # -u 参数强制不缓冲输出
                "--model-path", model_path,
                "--input", str(test_file),
                "--output", str(output_file),
                "--question-key", "problem",      # 与原项目一致
                "--answer-key", "predict",        # 与原项目一致：输出到 predict 字段
                "--image-key", "image_path",      # 与原项目一致
                "--temperature", str(self.config.eval_temperature),
                "--top-p", str(self.config.eval_top_p),
                "--frequency-penalty", str(self.config.eval_frequency_penalty),
                "--max-tokens", str(max_tokens),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-model-len", str(max_model_len),
            ]
            # 添加可选参数
            if disable_thinking:
                cmd.append("--disable-thinking")
            if is_error_analysis:
                cmd.append("--is-error-analysis")
            if is_correction:
                cmd.append("--is-correction")

        else:
            self.logger.info("Using local Python for vLLM generation")
            cmd = [
                "python", "-u", str(script_path),  # -u 参数强制不缓冲输出
                "--model-path", model_path,
                "--input", str(test_file),
                "--output", str(output_file),
                "--question-key", "problem",      # 与原项目一致
                "--answer-key", "predict",        # 与原项目一致：输出到 predict 字段
                "--image-key", "image_path",      # 与原项目一致
                "--temperature", str(self.config.eval_temperature),
                "--top-p", str(self.config.eval_top_p),
                "--frequency-penalty", str(self.config.eval_frequency_penalty),
                "--max-tokens", str(max_tokens),
                "--tensor-parallel-size", str(tensor_parallel_size),
                "--gpu-memory-utilization", str(gpu_memory_utilization),
                "--max-model-len", str(max_model_len),
            ]
            # 添加可选参数
            if disable_thinking:
                cmd.append("--disable-thinking")
            if is_error_analysis:
                cmd.append("--is-error-analysis")
            if is_correction:
                cmd.append("--is-correction")
        
        self.logger.info(f"Running LLM Generator: {' '.join(map(str, cmd[:6]))}...")
        
        try:
            # 使用 Popen 实现实时输出（与 Judge 一致）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # 实时输出日志（过滤掉详细的vLLM内部日志）
            self.logger.info("=== LLM Generator subprocess output ===")
            
            # 需要打印的关键词（错误、警告、重要进度）
            important_keywords = [
                'ERROR', 'WARNING', 'CRITICAL', 'Failed', 'Exception',
                'Traceback', '❌', '✅', 'completed', 'Starting', 'Finished'
            ]
            
            # 需要跳过的详细日志模式
            skip_patterns = [
                'Found nccl', 'is using nccl', 'rank', 'TP rank', 'DP rank',
                'Loading safetensors checkpoint shards:', 'Completed |',
                'Using FlashInfer', 'Using Flash Attention', 'Starting to load model',
                'Capturing CUDA graphs',  # CUDA graphs 捕获进度
                'it/s]',  # 进度条（通用）
                '% Completed',  # 百分比进度
                'Model loading took',  # 模型加载时间
                'Loading weights took',  # 权重加载时间
                'torch.compile',  # torch.compile 相关日志
                'Dynamo bytecode',  # Dynamo 编译日志
                'FutureWarning',  # Python 警告
                'pynvml package is deprecated',  # pynvml 警告
                'TORCH_CUDA_ARCH_LIST',  # CUDA 架构列表警告
                'SymmMemCommunicator',  # 对称内存通信器警告
                'Reducing Torch parallelism',  # Torch 并行度调整
                'OMP_NUM_THREADS',  # OpenMP 线程数
                'torch/utils/cpp_extension.py',  # C++ 扩展编译警告
                '[1;36m(EngineCore_DP0',  # 只有进程信息的空行
                '[1;36m(Worker_TP'  # 只有 Worker 信息的空行
            ]
            
            import re  # 用于移除 ANSI 颜色代码
            for line in process.stdout:
                line_stripped = line.rstrip()
                
                # 移除 ANSI 颜色代码后检查是否为空
                line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)  # 移除 ANSI 代码
                if not line_clean.strip():  # 如果移除颜色代码后为空，跳过
                    continue
                
                # 检查是否包含重要关键词
                should_print = any(keyword in line_stripped for keyword in important_keywords)
                # 或者不包含跳过模式
                should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                
                if should_print or not should_skip:
                    self.logger.info(f"  [vLLM] {line_stripped}")
            
            # 等待进程完成
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
    """LLM Judge 包装类"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate(
        self,
        predictions_file: Path,
        eval_records_file: Path,
        output_file: Path
    ) -> bool:
        """
        使用 LLM Judge 评估预测
        
        Args:
            predictions_file: 预测文件路径
            eval_records_file: 评估记录文件路径
            output_file: 输出文件路径
        
        Returns:
            是否成功
        """
        script_path = Path(__file__).parent / "run_llm_judge.py"
        
        # 准备 judge 配置（与原项目一致）
        judge_config = {
            "model_path": self.config.judge_model_path,
            "tensor_parallel_size": self.config.judge_tensor_parallel_size,
            "gpu_memory_utilization": self.config.judge_gpu_memory_utilization,
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
            "max_model_len": self.config.judge_max_model_len,  # 添加 max_model_len 配置
        }
        judge_config_json = json.dumps(judge_config)
        
        # 检查是否使用 Apptainer 容器
        use_apptainer = os.getenv("USE_APPTAINER_FOR_VLLM", "").lower() in ("true", "1", "yes")
        apptainer_image = os.getenv("APPTAINER_IMAGE", "")
        
        if use_apptainer and apptainer_image:
            self.logger.info(f"Using Apptainer for Judge: {apptainer_image}")
            # 使用 Apptainer 容器运行 Judge
            cmd = [
                "apptainer", "exec", "--nv",
                "--bind", "/mnt:/mnt",
                "--bind", "/tmp:/tmp",
                "--env", f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
                "--env", "PYTHONPATH=/mnt/petrelfs/shangxiaoran",
                "--env", f"HF_HOME={os.getenv('HF_HOME', '/tmp/huggingface')}",
                "--env", "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/.singularity.d/libs",
                # vLLM 日志控制：只显示 WARNING 及以上级别
                "--env", "VLLM_LOGGING_LEVEL=WARNING",
                # 关键修复：强制使用容器内的 GCC，避免宿主机 GCC 依赖问题
                "--env", "CC=/usr/bin/gcc",              # C 编译器
                "--env", "CXX=/usr/bin/g++",             # C++ 编译器
                "--env", "CUDAHOSTCXX=/usr/bin/g++",     # CUDA 主机编译器（最关键！）
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
            # 直接使用 Python 运行（宿主机环境）
            self.logger.info("Using host Python for Judge (Apptainer disabled)")
            cmd = [
                sys.executable, str(script_path),
                str(eval_records_file),
                str(predictions_file),
                str(output_file),
                judge_config_json  # JSON 字符串，不是文件
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
            
            # 实时输出日志（过滤掉详细的vLLM内部日志）
            self.logger.info("=== LLM Judge subprocess output ===")
            
            # 需要打印的关键词（错误、警告、重要进度）
            important_keywords = [
                'ERROR', 'WARNING', 'CRITICAL', 'Failed', 'Exception',
                'Traceback', '❌', '✅', 'completed', 'Starting', 'Finished',
                'Evaluating', 'Total problems'
            ]
            
            # 需要跳过的详细日志模式
            skip_patterns = [
                'Found nccl', 'is using nccl', 'rank', 'TP rank', 'DP rank',
                'Loading safetensors checkpoint shards:', 'Completed |',
                'Using FlashInfer', 'Using Flash Attention', 'Starting to load model',
                'Loading model from scratch', 'Using cache directory',
                'Capturing CUDA graphs',  # CUDA graphs 捕获进度
                'it/s]',  # 进度条（通用）
                '% Completed',  # 百分比进度
                'Model loading took',  # 模型加载时间
                'Loading weights took',  # 权重加载时间
                'torch.compile',  # torch.compile 相关日志
                'Dynamo bytecode',  # Dynamo 编译日志
                'FutureWarning',  # Python 警告
                'pynvml package is deprecated',  # pynvml 警告
                'TORCH_CUDA_ARCH_LIST',  # CUDA 架构列表警告
                'SymmMemCommunicator',  # 对称内存通信器警告
                'Reducing Torch parallelism',  # Torch 并行度调整
                'OMP_NUM_THREADS',  # OpenMP 线程数
                'torch/utils/cpp_extension.py',  # C++ 扩展编译警告
                '[1;36m(EngineCore_DP0',  # 只有进程信息的空行
                '[1;36m(Worker_TP'  # 只有 Worker 信息的空行
            ]
            
            # 🔑 关键修复：添加超时机制和非阻塞读取
            import select
            import time
            import re  # 用于移除 ANSI 颜色代码
            
            # 设置超时时间（20分钟无输出则认为卡住）
            # 注意：vLLM 资源清理（特别是多GPU环境下）可能需要较长时间
            timeout_seconds = 1200
            last_output_time = time.time()
            
            while True:
                # 检查进程是否结束
                if process.poll() is not None:
                    # 进程已结束，读取剩余输出
                    for line in process.stdout:
                        line_stripped = line.rstrip()
                        # 移除 ANSI 颜色代码后检查是否为空
                        line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                        if not line_clean.strip():
                            continue
                        should_print = any(keyword in line_stripped for keyword in important_keywords)
                        should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                        if should_print or not should_skip:
                            self.logger.info(f"  [Judge] {line_stripped}")
                    break
                
                # 非阻塞读取（使用 select，仅 Unix）
                # 对于 Windows，需要使用不同的方法
                try:
                    if sys.platform != 'win32':
                        # Unix 系统：使用 select
                        readable, _, _ = select.select([process.stdout], [], [], 1.0)
                        if readable:
                            line = process.stdout.readline()
                            if line:
                                line_stripped = line.rstrip()
                                # 移除 ANSI 颜色代码后检查是否为空
                                line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                                if line_clean.strip():  # 只处理非空行
                                    should_print = any(keyword in line_stripped for keyword in important_keywords)
                                    should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                                    if should_print or not should_skip:
                                        self.logger.info(f"  [Judge] {line_stripped}")
                                last_output_time = time.time()
                    else:
                        # Windows 系统：使用阻塞读取，但检查超时
                        line = process.stdout.readline()
                        if line:
                            line_stripped = line.rstrip()
                            # 移除 ANSI 颜色代码后检查是否为空
                            line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line_stripped)
                            if line_clean.strip():  # 只处理非空行
                                should_print = any(keyword in line_stripped for keyword in important_keywords)
                                should_skip = any(pattern in line_stripped for pattern in skip_patterns)
                                if should_print or not should_skip:
                                    self.logger.info(f"  [Judge] {line_stripped}")
                            last_output_time = time.time()
                except Exception as read_error:
                    self.logger.warning(f"Error reading subprocess output: {read_error}")
                
                # 检查超时
                if time.time() - last_output_time > timeout_seconds:
                    self.logger.error(f"❌ Subprocess timeout: no output for {timeout_seconds} seconds")
                    process.kill()
                    process.wait()
                    return False
            
            # 等待进程完成
            return_code = process.wait()
            self.logger.info(f"=== LLM Judge subprocess finished with code {return_code} ===")
            
            if return_code != 0:
                self.logger.error(f"❌ LLM Judge failed with exit code {return_code}")
                return False
            
            # 验证输出文件
            if not output_file.exists():
                self.logger.error(f"❌ Output file not found: {output_file}")
                return False
            
            self.logger.info("✅ LLM Judge completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ LLM Judge failed: {e}")
            # 确保清理子进程
            try:
                if 'process' in locals() and process.poll() is None:
                    process.kill()
                    process.wait()
            except:
                pass
            return False


