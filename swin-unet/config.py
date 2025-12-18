# swin-unet/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    # Temporal setup
    year_target: int = 5  # 4 input years + 1 target year (matches original thesis model)

    # Dataset paths
    dir_folders: str = "data/satellite/dataset_month3"
    collection: str = "JRC_GSW1_4_MonthlyHistory"

    # Filtering & values
    nonwater_threshold: int = 480_000
    nodata_value: int = -1
    nonwater_value: int = 0
    scaled_classes: bool = True

    # DataLoader - reduced batch size for larger model
    batch_size: int = 8
    num_workers: int = 12
    use_cache: bool = True
    cache_dir: Path = Path("swin-unet/cache")

    # For plotting / misclassification
    plots_dir: Path = Path("swin-unet/plots")


@dataclass
class ModelConfig:
    # Model variant: "tiny" or "small"
    variant: str = "tiny"  # Options: "tiny", "small"

    # Temporal aggregation: how to combine temporal frames
    temporal_aggregation: str = "concat_proj"  # Options: "concat_proj", "learned_weighted_sum", "mean"

    # Architecture hyperparameters (for custom configurations)
    # These override variant defaults if specified
    img_size: tuple = (1000, 500)
    patch_size: Optional[list] = None  # Will use variant default if None
    in_chans: int = 4  # Number of temporal frames (inferred from data)
    num_classes: int = 1  # Binary segmentation
    embed_dim: Optional[int] = None  # Will use variant default if None
    depths: Optional[list] = None  # Will use variant default if None
    num_heads: Optional[list] = None  # Will use variant default if None
    window_size: Optional[list] = None  # Will use variant default if None
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1


@dataclass
class TrainConfig:
    num_epochs: int = 50
    lr: float = 1e-4  # Conservative for transformers
    weight_decay: float = 0.05  # Standard for vision transformers

    nonwater_label: int = 0
    water_label: int = 1
    pixel_size: int = 60
    water_threshold: float = 0.5
    loss_f: str = "BCE"  # Options: "BCE", "BCE_Logits", "Focal"
    physics: bool = False  # Physics-based loss terms

    # Checkpoint management (will be updated based on variant)
    ckpt_dir: Path = Path("swin-unet/checkpoints_tiny")
    save_every_n_epochs: int = 1

    # Learning rate scheduling (optional)
    use_scheduler: bool = False
    scheduler_type: str = "cosine"  # Options: "cosine", "step"
    warmup_epochs: int = 5


@dataclass
class EvalConfig:
    # These will be updated based on model variant
    # Default to tiny, but should be updated when variant changes
    checkpoint_pattern: str = "stswin_tiny_epoch*.pt"
    checkpoint_dir: Path = Path("swin-unet/checkpoints_tiny")
    scores_csv: Path = Path("swin-unet/scores/test_metrics_all_epochs_stswin_tiny.csv")

    def update_for_variant(self, variant: str):
        """Update paths based on model variant."""
        self.checkpoint_pattern = f"stswin_{variant}_epoch*.pt"
        self.checkpoint_dir = Path(f"swin-unet/checkpoints_{variant}")
        self.scores_csv = Path(f"swin-unet/scores/test_metrics_all_epochs_stswin_{variant}.csv")


# Single global instances you can import everywhere
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
eval_cfg = EvalConfig()
