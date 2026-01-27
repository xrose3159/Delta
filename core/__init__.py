"""
Adaptive Training Core Modules
"""

# 使用try-except避免导入错误
try:
    from .dpo_trainer import DPOTrainer
    from .multi_round_pipeline_dpo import MultiRoundPipelineDPO
    
    __all__ = [
        "DPOTrainer",
        "MultiRoundPipelineDPO",
    ]
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from adaptive_training.core.dpo_trainer import DPOTrainer
    from adaptive_training.core.multi_round_pipeline_dpo import MultiRoundPipelineDPO
    
    __all__ = [
        "DPOTrainer",
        "MultiRoundPipelineDPO",
    ]
