"""
DPO 训练器 - 直接调用 LLaMA-Factory

功能：
1. 准备 LLaMA-Factory 数据集配置
2. 生成 DPO 训练配置文件
3. 调用 LLaMA-Factory CLI 进行训练
"""

import json
import logging
import os
import subprocess
import yaml  # 使用 yaml 库生成配置文件（与原项目一致）
from pathlib import Path
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class DPOTrainer:
    """DPO 训练器 - 使用 LLaMA-Factory"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LLaMA-Factory 路径
        self.llamafactory_path = Path("/mnt/petrelfs/shangxiaoran/math_generation6/LLaMA-Factory")
        if not self.llamafactory_path.exists():
            self.logger.warning(f"LLaMA-Factory not found at {self.llamafactory_path}")
    
    def prepare_dataset_info(
        self,
        dataset_name: str,
        dataset_file: Path,
        dataset_info_path: Path
    ) -> None:
        """
        准备 LLaMA-Factory 的 dataset_info.json
        
        Args:
            dataset_name: 数据集名称
            dataset_file: DPO 数据文件路径
            dataset_info_path: dataset_info.json 保存路径
        """
        # LLaMA-Factory 数据集配置格式
        # 🔧 使用 sharegpt 格式，因为我们的数据使用 conversations
        dataset_info = {
            dataset_name: {
                "file_name": str(dataset_file),
                "formatting": "sharegpt",  # 使用 ShareGPT 格式
                "ranking": True,  # 启用 DPO
                "columns": {
                    "messages": "conversations",
                    "chosen": "chosen",
                    "rejected": "rejected",
                    "images": "images",
                    "system": "system"  # 添加 system prompt 映射
                },
                # 🔧 添加role tags以匹配我们的数据格式
                "tags": {
                    "role_tag": "from",
                    "content_tag": "value",
                    "user_tag": "user",
                    "assistant_tag": "assistant"
                }
            }
        }
        
        # 如果已存在 dataset_info.json，则更新它
        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    existing_info = json.load(f)
                existing_info.update(dataset_info)
                dataset_info = existing_info
            except Exception as e:
                self.logger.warning(f"Failed to load existing dataset_info.json: {e}, will create new one")
        
        # 保存
        dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Dataset info updated/created: {dataset_info_path}")
        self.logger.info(f"  Added/Updated dataset: {dataset_name}")
    
    def create_training_config(
        self,
        dataset_name: str,
        model_path: Path,
        output_dir: Path,
        config_path: Path
    ) -> Path:
        """
        创建 LLaMA-Factory 训练配置文件（YAML 格式）
        
        Args:
            dataset_name: 数据集名称
            model_path: 模型路径
            output_dir: 输出目录
            config_path: 配置文件保存路径
        
        Returns:
            配置文件路径
        """
        # 从环境变量获取 GPU 配置（支持测试脚本）
        import os
        num_gpus = int(os.environ.get('DPO_NUM_GPUS', self.config.dpo_num_gpus))
        batch_size = int(os.environ.get('DPO_BATCH_SIZE', self.config.dpo_batch_size))
        gradient_accumulation = int(os.environ.get('DPO_GRADIENT_ACCUMULATION', self.config.dpo_gradient_accumulation))
        num_epochs = int(os.environ.get('DPO_NUM_EPOCHS', self.config.dpo_epochs))
        learning_rate = float(os.environ.get('DPO_LEARNING_RATE', self.config.dpo_learning_rate))
        
        self.logger.info(f"DPO Training Config: GPUs={num_gpus}, batch_size={batch_size}, "
                        f"grad_accum={gradient_accumulation}, epochs={num_epochs}, lr={learning_rate}")
        
        # 构建训练参数（参考 qwen2_5vl_full_sft.yaml + qwen2_5vl_lora_dpo.yaml）
        training_args = {
            # === model ===
            "model_name_or_path": str(model_path),
            "image_max_pixels": 262144,
            "video_max_pixels": 16384,
            "trust_remote_code": True,
            "model_revision": "main",  # 本地模型使用main分支
            
            # === method ===
            "stage": "dpo",
            "do_train": True,
            "finetuning_type": "full",  
            "freeze_vision_tower": self.config.dpo_freeze_vision_tower,  # 是否冻结视觉编码器
            "freeze_multi_modal_projector": self.config.dpo_freeze_projector,  # 是否冻结投影层
            "freeze_language_model": self.config.dpo_freeze_llm,  # 是否冻结语言模型
            "pref_beta": self.config.dpo_beta,
            "pref_loss": "sigmoid", 
            "deepspeed": "/mnt/petrelfs/shangxiaoran/math_generation6/LLaMA-Factory/examples/deepspeed/ds_z3_config.json",  
            
            # === dataset ===
            "dataset": dataset_name,
            "template": self.config.dpo_template,
            "cutoff_len": self.config.dpo_max_length,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16, 
            "dataloader_num_workers": 4,
            
            # === output ===
            "output_dir": str(output_dir),
            "logging_steps": 10,
            "save_steps": 500,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": "none",  
            
            # === train ===
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "gradient_checkpointing": True,  
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": None,
        }
        

        
        
        # 保存为 YAML 格式（使用 yaml.dump，与原项目一致）
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(training_args, f, allow_unicode=True, default_flow_style=False)
        
        self.logger.info(f"Training config saved to: {config_path}")
        
        return config_path
    
    def train(
        self,
        dataset_name: str,
        dpo_data_file: Path,
        model_path: Path,
        output_dir: Path,
        round_num: int
    ) -> bool:
        """
        执行 DPO 训练
        
        Args:
            dataset_name: 数据集名称
            dpo_data_file: DPO 数据文件路径
            model_path: 模型路径
            output_dir: 输出目录
            round_num: 当前轮次
        
        Returns:
            是否成功
        """
        try:
            self.logger.info(f"=" * 80)
            self.logger.info(f"Starting DPO training for Round {round_num}")
            self.logger.info(f"=" * 80)
            
            # 1. 准备 dataset_info.json
            dataset_info_path = self.llamafactory_path / "data" / "dataset_info.json"
            self.prepare_dataset_info(dataset_name, dpo_data_file, dataset_info_path)
            
            # 2. 创建训练配置
            config_dir = output_dir / "dpo_configs"
            config_path = config_dir / f"dpo_config_round_{round_num}.yaml"
            self.create_training_config(dataset_name, model_path, output_dir, config_path)
            
            # 3. 调用 LLaMA-Factory 训练
            self.logger.info("Calling LLaMA-Factory for DPO training...")
            
            # 构建命令 - 直接调用本地LLaMA-Factory的Python模块，避免使用全局的llamafactory-cli
            # 这样确保使用的是math_generation6目录下的LLaMA-Factory，而不是system-wide安装的版本
            cmd = [
                "python3", "-m", "llamafactory.cli", "train",
                str(config_path)
            ]
            
            self.logger.info(f"Command: {' '.join(cmd)}")
            self.logger.info(f"Working directory: {self.llamafactory_path}")
            
            # 设置环境变量
            env = os.environ.copy()
            env['HF_HUB_OFFLINE'] = '1'  # 完全离线模式
            env['TRANSFORMERS_OFFLINE'] = '1'  # Transformers离线模式
            env['HF_DATASETS_OFFLINE'] = '1'  # Datasets离线模式
            env['DATASETS_VERBOSITY'] = 'error'  # 减少datasets日志
            # 🔧 禁用datasets缓存以避免多GPU竞态条件
            env['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache_dpo'  # 使用临时缓存目录
            os.makedirs('/tmp/hf_datasets_cache_dpo', exist_ok=True)
            
            # 🔧 设置PYTHONPATH确保使用本地的LLaMA-Factory
            llamafactory_src = str(self.llamafactory_path / "src")
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{llamafactory_src}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = llamafactory_src
            self.logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
            
            # 执行训练
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.llamafactory_path),
                env=env
            )
            
            # 实时输出日志
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.logger.info(line)
            
            # 等待完成
            return_code = process.wait()
            
            if return_code == 0:
                self.logger.info("✅ DPO training completed successfully")
                return True
            else:
                self.logger.error(f"❌ DPO training failed with code {return_code}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error during DPO training: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    # 测试代码
    from config import get_default_config
    
    config = get_default_config()
    trainer = DPOTrainer(config)
    
    # 测试数据集配置
    dataset_name = "test_dpo_round_1"
    dataset_file = Path("./test_dpo_data.json")
    dataset_info_path = Path("./test_dataset_info.json")
    
    trainer.prepare_dataset_info(dataset_name, dataset_file, dataset_info_path)
    
    print("✅ DPO Trainer test completed")

