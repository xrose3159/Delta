Adaptive Training Core Modules


try:
    from .dpo_trainer import DPOTrainer
    from .multi_round_pipeline_dpo import MultiRoundPipelineDPO

    __all__ = [
        "DPOTrainer",
        "MultiRoundPipelineDPO",
    ]
except ImportError:

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
