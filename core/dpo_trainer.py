

import json
import logging
import os
import subprocess
import yaml  
from pathlib import Path
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class DPOTrainer:
    """DPO trainer - use LLaMA-Factory"""
    
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
        """
        Prepare LLaMA-Factory's dataset_info.json
        
        Args:
            dataset_name: dataset name
            dataset_file: DPO data file path
            dataset_info_path: dataset_info.json save path
        """
        dataset_info = {
            dataset_name: {
                "file_name": str(dataset_file),
                "formatting": "sharegpt", 
                "ranking": True, 
                "columns": {
                    "messages": "conversations",
                    "chosen": "chosen",
                    "rejected": "rejected",
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
        config_path: Path
    ) -> Path:
        import os
        num_gpus = int(os.environ.get('DPO_NUM_GPUS', self.config.dpo_num_gpus))
        batch_size = int(os.environ.get('DPO_BATCH_SIZE', self.config.dpo_batch_size))
        gradient_accumulation = int(os.environ.get('DPO_GRADIENT_ACCUMULATION', self.config.dpo_gradient_accumulation))
        num_epochs = int(os.environ.get('DPO_NUM_EPOCHS', self.config.dpo_epochs))
        learning_rate = float(os.environ.get('DPO_LEARNING_RATE', self.config.dpo_learning_rate))
        
        self.logger.info(f"DPO Training Config: GPUs={num_gpus}, batch_size={batch_size}, "
                        f"grad_accum={gradient_accumulation}, epochs={num_epochs}, lr={learning_rate}")
        
        training_args = {
            # === model ===
            "model_name_or_path": str(model_path),
            "image_max_pixels": 262144,
            "video_max_pixels": 16384,
            "trust_remote_code": True,
            "model_revision": "main",  
            
            # === method ===
            "stage": "dpo",
            "do_train": True,
            "finetuning_type": "full",  
            "freeze_vision_tower": self.config.dpo_freeze_vision_tower, 
            "freeze_multi_modal_projector": self.config.dpo_freeze_projector, 
            "freeze_language_model": self.config.dpo_freeze_llm, 
            "pref_beta": self.config.dpo_beta,
            "pref_loss": "sigmoid", 
            "deepspeed": str(self.llamafactory_path / "examples/deepspeed/ds_z3_config.json"),  
            
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
        try:
            self.logger.info(f"=" * 80)
            self.logger.info(f"Starting DPO training for Round {round_num}")
            self.logger.info(f"=" * 80)
            
            # 1. prepare dataset_info.json
            dataset_info_path = self.llamafactory_path / "data" / "dataset_info.json"
            self.prepare_dataset_info(dataset_name, dpo_data_file, dataset_info_path)
            
            # 2. create training config
            config_dir = output_dir / "dpo_configs"
            config_path = config_dir / f"dpo_config_round_{round_num}.yaml"
            self.create_training_config(dataset_name, model_path, output_dir, config_path)
            
            # 3. call LLaMA-Factory for training
            self.logger.info("Calling LLaMA-Factory for DPO training...")
            
            # build command - directly call local LLaMA-Factory's Python module, avoid using global llamafactory-cli
            # this ensures using the LLaMA-Factory in math_generation6 directory, not the system-wide installed version
            cmd = [
                "python3", "-m", "llamafactory.cli", "train",
                str(config_path)
            ]
            
            self.logger.info(f"Command: {' '.join(cmd)}")
            self.logger.info(f"Working directory: {self.llamafactory_path}")
            
            # set environment variables
            env = os.environ.copy()
            env['HF_HUB_OFFLINE'] = '1'  # fully offline mode
            env['TRANSFORMERS_OFFLINE'] = '1'  # Transformers offline mode
            env['HF_DATASETS_OFFLINE'] = '1'  # Datasets offline mode
            env['DATASETS_VERBOSITY'] = 'error'  # reduce datasets log
            # disable datasets cache to avoid multi-GPU race conditions
            env['HF_DATASETS_CACHE'] = '/tmp/hf_datasets_cache_dpo'  # use temporary cache directory
            os.makedirs('/tmp/hf_datasets_cache_dpo', exist_ok=True)
            
            # set PYTHONPATH to ensure using local LLaMA-Factory
            llamafactory_src = str(self.llamafactory_path / "src")
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{llamafactory_src}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = llamafactory_src
            self.logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")
            
            # execute training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.llamafactory_path),
                env=env
            )
            
            # real-time output log
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.logger.info(line)
            
            # wait for completion
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
    # test code
    from config import get_default_config
    
    config = get_default_config()
    trainer = DPOTrainer(config)
    
    # test dataset configuration
    dataset_name = "test_dpo_round_1"
    dataset_file = Path("./test_dpo_data.json")
    dataset_info_path = Path("./test_dataset_info.json")
    
    trainer.prepare_dataset_info(dataset_name, dataset_file, dataset_info_path)
    
    print("✅ DPO Trainer test completed")

