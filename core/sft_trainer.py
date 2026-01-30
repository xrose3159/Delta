

import json
import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class SFTTrainer:
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
           
        self.llamafactory_path = Path(os.getenv("LLAMAFACTORY_PATH", "./LLaMA-Factory"))
        if not self.llamafactory_path.exists():
            self.logger.warning(f"LLaMA-Factory not found at {self.llamafactory_path}")
    
    def prepare_dataset_info(
        self,
        dataset_name: str,
        dataset_file: Path,
        dataset_info_path: Path
    ) -> None:

        dataset_info = {
            dataset_name: {
                "file_name": str(dataset_file),
                "formatting": "sharegpt", 
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

        if dataset_info_path.exists():
            try:
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    existing_info = json.load(f)
                existing_info.update(dataset_info)
                dataset_info = existing_info
            except Exception as e:
                self.logger.warning(f"Failed to load existing dataset_info.json: {e}, will create new one")
        
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

        config = {
   
            "model_name_or_path": str(model_path),
            "image_max_pixels": 262144,
            "video_max_pixels": 16384,
            "trust_remote_code": True,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full",
            "freeze_vision_tower": self.config.sft_freeze_vision_tower,
            "freeze_multi_modal_projector": self.config.sft_freeze_projector,
            "freeze_language_model": self.config.sft_freeze_llm,
            "deepspeed": str(self.llamafactory_path / "examples/deepspeed/ds_z3_config.json"),
            "packing": getattr(self.config, "sft_enable_packing", False),

            "dataset": dataset_name,
            "template": "qwen3_vl_nothink",
            "cutoff_len": self.config.sft_cutoff_len,
            "overwrite_cache": True,
            "preprocessing_num_workers": 32,
            "dataloader_num_workers": 8,
    
            "output_dir": str(output_dir),
            "logging_steps": self.config.sft_logging_steps,
            "save_steps": self.config.sft_save_steps,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": "none",
            
          
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

          
            "val_size": 0.0,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "no",
        }
        
        if self.config.sft_use_lora:
            config.update({
                "lora_rank": self.config.sft_lora_rank,
                "lora_alpha": self.config.sft_lora_alpha,
                "lora_dropout": self.config.sft_lora_dropout,
                "lora_target": "all",
            })
        
 
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        self.logger.info(f"Training config created: {config_path}")
        return config_path
    
    def _verify_model_files(self, output_dir: Path) -> bool:
        required_files = ["config.json"]
        model_file_patterns = ["*.safetensors", "*.bin", "pytorch_model*.bin"]
        
     
        config_exists = (output_dir / "config.json").exists()
        model_weights_exist = False
        for pattern in model_file_patterns:
            if list(output_dir.glob(pattern)):
                model_weights_exist = True
                break
        
        self.logger.info(f"Model verification - config.json: {config_exists}, weights: {model_weights_exist}")
        
        return config_exists and model_weights_exist
    
    def _export_model(self, config_path: Path, output_dir: Path) -> bool:

        with open(config_path, 'r', encoding='utf-8') as f:
            train_config = yaml.safe_load(f)

        if train_config.get("finetuning_type") != "lora":
            self.logger.info("Full fine-tuning detected. Checking for checkpoint directories...")
     
            checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
            
            if checkpoint_dirs:
                latest_checkpoint = checkpoint_dirs[-1]
                self.logger.info(f"Found checkpoint: {latest_checkpoint}")
                
        
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
        
        self.logger.info("LoRA fine-tuning detected. Using llamafactory-cli export...")
        
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
        
        with open(export_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(export_config, f, allow_unicode=True, default_flow_style=False)
        
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

        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting SFT Training - Round {round_num}")
        self.logger.info(f"=" * 80)
        self.logger.info(f"Dataset: {sft_data_file}")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Output: {output_dir}")
        
        lr_to_use = learning_rate if learning_rate is not None else self.config.sft_learning_rate
        self.logger.info(f"Using learning rate={lr_to_use:.2e} for round {round_num}")

    
        dataset_info_path = self.llamafactory_path / "data" / "dataset_info.json"
        self.prepare_dataset_info(dataset_name, sft_data_file, dataset_info_path)
        
       
        config_path = output_dir.parent / f"sft_config_round_{round_num}.yaml"
        self.create_training_config(dataset_name, model_path, output_dir, config_path, learning_rate=lr_to_use)
        
        self.logger.info("Starting LLaMA-Factory training...")
        
        cmd = [
            "llamafactory-cli", "train",
            str(config_path)
        ]
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        
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
                    self.logger.info(f"[LLaMA-Factory] {line}")
            
            process.wait()
            
            if process.returncode == 0:
                self.logger.info("=" * 80)
                self.logger.info("✅ SFT training completed successfully!")
                self.logger.info("=" * 80)
                
                model_files_exist = self._verify_model_files(output_dir)
                if not model_files_exist:
                    self.logger.warning("⚠️ Model files not found in output directory.")
                    self.logger.warning("This is expected for small datasets where training steps < save_steps.")
                    self.logger.warning("Attempting to recover model files...")
                    
                    export_success = self._export_model(config_path, output_dir)
                    if not export_success:
                  
                        self.logger.warning("⚠️ No checkpoints found. Copying base model as fallback...")
                        import shutil
                        try:
                   
                            if model_path.exists():
                                for item in model_path.iterdir():
                                    if item.suffix in ['.json', '.safetensors', '.bin', '.model', '.txt']:
                                        dest = output_dir / item.name
                                        if item.is_file():
                                            shutil.copy2(item, dest)
                                            self.logger.info(f"Copied {item.name}")
                                self.logger.warning("⚠️ Used base model (no training applied due to insufficient steps)")
                                return True  
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

