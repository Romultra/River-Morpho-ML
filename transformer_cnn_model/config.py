# transformer_cnn_model/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    # Temporal setup
    year_target: int = 10

    # Dataset paths
    dir_folders: str = "data/satellite/dataset_month3"
    collection: str = "JRC_GSW1_4_MonthlyHistory"

    # Filtering & values
    nonwater_threshold: int = 480_000
    nodata_value: int = -1
    nonwater_value: int = 0
    scaled_classes: bool = True

    # DataLoader
    batch_size: int = 16
    num_workers: int = 12
    use_cache: bool = True
    cache_dir: Path = Path("transformer_cnn_model/cache")

    # For plotting / misclassification
    plots_dir: Path = Path("transformer_cnn_model/plots")


@dataclass
class ModelConfig:
    # Which architecture: "transunet" or "unet3d"
    architecture: str = "transunet"

    # Shared UNet params
    init_hid_dim: int = 8
    kernel_size: int = 3
    pooling: str = "max"
    bilinear: bool = False
    drop_channels: bool = False
    p_drop: Optional[float] = None
    n_classes: int = 1

    # Transformer-specific
    use_temporal_transformer: bool = True
    d_model: int = 8
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 64
    dropout: float = 0.1


@dataclass
class TrainConfig:
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5

    nonwater_label: int = 0
    water_label: int = 1
    pixel_size: int = 60
    water_threshold: float = 0.5
    loss_f: str = "BCE"
    physics: bool = False

    # Where to save checkpoints
    ckpt_dir: Path = Path("transformer_cnn_model/checkpoints_transunet_9years")


@dataclass
class EvalConfig:
    # Pattern for evaluating transformer checkpoints
    checkpoint_pattern: str = "transunet_epoch*.pt"
    # Default directory for checkpoints (can still be overridden via CLI)
    checkpoint_dir: Path = Path("transformer_cnn_model/checkpoints_transunet_9years")
    # Default CSV for scores
    scores_csv: Path = Path(
        "transformer_cnn_model/scores/test_metrics_all_epochs_transunet_9years.csv"
    )


# Single global instances you can import everywhere
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
eval_cfg = EvalConfig()
