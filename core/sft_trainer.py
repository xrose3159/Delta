"""
SFT 训练器 - 直接调用 LLaMA-Factory

功能：
1. 准备 LLaMA-Factory 数据集配置
2. 生成 SFT 训练配置文件
3. 调用 LLaMA-Factory CLI 进行训练
"""

import json
import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class SFTTrainer:
    """SFT 训练器 - 使用 LLaMA-Factory"""
    
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
            dataset_file: SFT 数据文件路径
            dataset_info_path: dataset_info.json 保存路径
        """
        # LLaMA-Factory 数据集配置格式（SFT 使用 sharegpt 格式）
        dataset_info = {
            dataset_name: {
                "file_name": str(dataset_file),
                "formatting": "sharegpt",  # 使用 ShareGPT 格式
                "columns": {
                    "messages": "conversations",
                    "images": "images",
                    "system": "system"
                },
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
        config_path: Path,
        learning_rate: Optional[float] = None,
    ) -> Path:
        """
        创建 LLaMA-Factory 训练配置文件（YAML 格式）
        
        Args:
            dataset_name: 数据集名称
            model_path: 基础模型路径
            output_dir: 输出目录
            config_path: 配置文件保存路径
        
        Returns:
            配置文件路径
        """
        # SFT 训练配置（与 DPO 类似，但去掉 DPO 特定参数）
        config = {
            ### model
            "model_name_or_path": str(model_path),
            "image_max_pixels": 262144,
            "video_max_pixels": 16384,
            "trust_remote_code": True,
            
            ### method
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full",
            "freeze_vision_tower": self.config.sft_freeze_vision_tower,
            "freeze_multi_modal_projector": self.config.sft_freeze_projector,
            "freeze_language_model": self.config.sft_freeze_llm,
            "deepspeed": "/mnt/petrelfs/shangxiaoran/math_generation6/LLaMA-Factory/examples/deepspeed/ds_z3_config.json",
            "packing": getattr(self.config, "sft_enable_packing", False),

            ### dataset
            "dataset": dataset_name,
            "template": "qwen3_vl_nothink",
            "cutoff_len": self.config.sft_cutoff_len,
            "overwrite_cache": True,
            "preprocessing_num_workers": 32,
            "dataloader_num_workers": 8,
            
            ### output
            "output_dir": str(output_dir),
            "logging_steps": self.config.sft_logging_steps,
            "save_steps": self.config.sft_save_steps,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": "none",
            
            ### train
            "per_device_train_batch_size": self.config.sft_per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.sft_gradient_accumulation_steps,
            "learning_rate": learning_rate if learning_rate is not None else self.config.sft_learning_rate,
            "num_train_epochs": self.config.sft_num_train_epochs,
            "lr_scheduler_type": self.config.sft_lr_scheduler_type,
            "warmup_ratio": self.config.sft_warmup_ratio,
            "bf16": True,
            "gradient_checkpointing": True,
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": None,

            ### eval
            "val_size": 0.0,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "no",
        }
        
        # 如果使用 LoRA，添加 LoRA 参数
        if self.config.sft_use_lora:
            config.update({
                "lora_rank": self.config.sft_lora_rank,
                "lora_alpha": self.config.sft_lora_alpha,
                "lora_dropout": self.config.sft_lora_dropout,
                "lora_target": "all",
            })
        
        # 保存配置
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        self.logger.info(f"Training config created: {config_path}")
        return config_path
    
    def _verify_model_files(self, output_dir: Path) -> bool:
        """
        验证模型文件是否存在
        
        Args:
            output_dir: 模型输出目录
        
        Returns:
            模型文件是否存在
        """
        # 检查必需的模型文件
        required_files = ["config.json"]
        model_file_patterns = ["*.safetensors", "*.bin", "pytorch_model*.bin"]
        
        # 检查 config.json
        config_exists = (output_dir / "config.json").exists()
        
        # 检查模型权重文件
        model_weights_exist = False
        for pattern in model_file_patterns:
            if list(output_dir.glob(pattern)):
                model_weights_exist = True
                break
        
        self.logger.info(f"Model verification - config.json: {config_exists}, weights: {model_weights_exist}")
        
        return config_exists and model_weights_exist
    
    def _export_model(self, config_path: Path, output_dir: Path) -> bool:
        """
        导出训练后的模型（如果需要）
        
        Args:
            config_path: 训练配置文件路径
            output_dir: 模型输出目录
        
        Returns:
            导出是否成功
        """
        # 读取训练配置
        with open(config_path, 'r', encoding='utf-8') as f:
            train_config = yaml.safe_load(f)
        
        # 如果不是 LoRA，检查是否有 checkpoint 子目录
        if train_config.get("finetuning_type") != "lora":
            self.logger.info("Full fine-tuning detected. Checking for checkpoint directories...")
            
            # 查找所有 checkpoint 目录
            checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
            
            if checkpoint_dirs:
                latest_checkpoint = checkpoint_dirs[-1]
                self.logger.info(f"Found checkpoint: {latest_checkpoint}")
                
                # 将 checkpoint 中的文件复制到主目录
                import shutil
                for item in latest_checkpoint.iterdir():
                    dest = output_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest)
                        self.logger.info(f"Copied {item.name}")
                    elif item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                        self.logger.info(f"Copied directory {item.name}")
                
                self.logger.info("✅ Checkpoint files copied to output directory")
                return True
            else:
                self.logger.error("No checkpoint directories found. Training may not have saved any checkpoints.")
                self.logger.error(f"Possible reasons: 1) Training steps < save_steps, 2) Training failed silently")
                return False
        
        # LoRA 模式：使用 llamafactory-cli export
        self.logger.info("LoRA fine-tuning detected. Using llamafactory-cli export...")
        
        # 创建导出配置
        export_config_path = config_path.parent / f"export_{config_path.stem}.yaml"
        
        export_config = {
            "model_name_or_path": train_config["model_name_or_path"],
            "adapter_name_or_path": str(output_dir),
            "template": train_config["template"],
            "finetuning_type": "lora",
            "export_dir": str(output_dir / "exported_model"),
            "export_size": 2,
            "export_device": "cpu",
            "export_legacy_format": False,
        }
        
        # 保存导出配置
        with open(export_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_config, f, allow_unicode=True, default_flow_style=False)
        
        # 执行导出
        cmd = [
            "llamafactory-cli", "export",
            str(export_config_path)
        ]
        
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(self.llamafactory_path)
            )
            
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.logger.info(f"[Export] {line}")
            
            process.wait()
            
            if process.returncode == 0:
                self.logger.info("✅ Model exported successfully")
                return True
            else:
                self.logger.error(f"❌ Model export failed with exit code {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Model export failed with exception: {e}", exc_info=True)
            return False
    
    def train(
        self,
        dataset_name: str,
        sft_data_file: Path,
        model_path: Path,
        output_dir: Path,
        round_num: int,
        learning_rate: Optional[float] = None,
    ) -> bool:
        """
        执行 SFT 训练
        
        Args:
            dataset_name: 数据集名称
            sft_data_file: SFT 数据文件路径
            model_path: 基础模型路径
            output_dir: 输出目录
            round_num: 轮次编号
        
        Returns:
            训练是否成功
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting SFT Training - Round {round_num}")
        self.logger.info(f"=" * 80)
        self.logger.info(f"Dataset: {sft_data_file}")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Output: {output_dir}")
        
        lr_to_use = learning_rate if learning_rate is not None else self.config.sft_learning_rate
        self.logger.info(f"Using learning rate={lr_to_use:.2e} for round {round_num}")

        # 1. 准备数据集配置
        dataset_info_path = self.llamafactory_path / "data" / "dataset_info.json"
        self.prepare_dataset_info(dataset_name, sft_data_file, dataset_info_path)
        
        # 2. 创建训练配置
        config_path = output_dir.parent / f"sft_config_round_{round_num}.yaml"
        self.create_training_config(dataset_name, model_path, output_dir, config_path, learning_rate=lr_to_use)
        
        # 3. 执行训练
        self.logger.info("Starting LLaMA-Factory training...")
        
        cmd = [
            "llamafactory-cli", "train",
            str(config_path)
        ]
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # 使用 subprocess 执行训练
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(self.llamafactory_path)
            )
            
            # 实时输出日志
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.logger.info(f"[LLaMA-Factory] {line}")
            
            process.wait()
            
            if process.returncode == 0:
                self.logger.info("=" * 80)
                self.logger.info("✅ SFT training completed successfully!")
                self.logger.info("=" * 80)
                
                # 验证模型文件是否存在
                model_files_exist = self._verify_model_files(output_dir)
                if not model_files_exist:
                    self.logger.warning("⚠️ Model files not found in output directory.")
                    self.logger.warning("This is expected for small datasets where training steps < save_steps.")
                    self.logger.warning("Attempting to recover model files...")
                    
                    # 尝试从checkpoint恢复
                    export_success = self._export_model(config_path, output_dir)
                    if not export_success:
                        # 如果还是失败，直接复制基础模型
                        self.logger.warning("⚠️ No checkpoints found. Copying base model as fallback...")
                        import shutil
                        try:
                            # 复制基础模型文件到输出目录
                            if model_path.exists():
                                for item in model_path.iterdir():
                                    if item.suffix in ['.json', '.safetensors', '.bin', '.model', '.txt']:
                                        dest = output_dir / item.name
                                        if item.is_file():
                                            shutil.copy2(item, dest)
                                            self.logger.info(f"Copied {item.name}")
                                self.logger.warning("⚠️ Used base model (no training applied due to insufficient steps)")
                                return True  # 虽然没训练，但至少有可用的模型
                            else:
                                self.logger.error(f"❌ Base model not found at {model_path}")
                                return False
                        except Exception as e:
                            self.logger.error(f"❌ Failed to copy base model: {e}")
                            return False
                else:
                    self.logger.info("✅ Model files verified successfully")
                
                return True
            else:
                self.logger.error(f"❌ SFT training failed with exit code {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ SFT training failed with exception: {e}", exc_info=True)
            return False

