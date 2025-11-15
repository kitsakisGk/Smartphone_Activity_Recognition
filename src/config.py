"""Configuration management for the activity recognition system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Directory paths (relative to project root)
    data_dir: Path = Path("../Data")  # Points to existing Data folder
    models_dir: Path = Path("models")
    images_dir: Path = Path("outputs/images")

    # Dataset parameters
    num_users: int = 3
    num_days: int = 3
    num_sensors: int = 3  # Accelerometer, Gyroscope, Magnetometer
    num_features: int = 9  # 3 sensors × 3 axes (x, y, z)

    # Phone positions available in dataset
    phone_positions: List[str] = field(default_factory=lambda: ["Hand", "Torso", "Bag", "Hips"])
    selected_position: str = "Hand"

    # Preprocessing
    normalization_range: Tuple[float, float] = (-1.0, 1.0)

    # Train/validation split (standard split, no LOSO)
    train_split: float = 0.8
    validation_split: float = 0.2
    random_seed: int = 42

    # Time series windowing (for LSTM/GRU models)
    use_time_windows: bool = True
    window_size: int = 50  # Number of timesteps per window
    window_step: int = 25  # Sliding window step (50% overlap)


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""

    # Input shape
    input_shape: Tuple[int, int] = (9, 1)  # (features, channels)

    # Training parameters (REDUCED for slower laptops)
    learning_rate: float = 0.001
    batch_size: int = 5000   # Reduced from 10000
    epochs: int = 15         # Reduced from 25
    patience: int = 10       # Reduced from 20

    # Model architecture (CNN)
    cnn_filters: List[int] = field(default_factory=lambda: [64, 32, 64, 64])
    cnn_kernel_size: int = 3
    cnn_pool_size: int = 2
    dense_units: List[int] = field(default_factory=lambda: [512, 256])

    # CNN-LSTM specific
    lstm_units: int = 100
    lstm_dense_units: int = 20
    lstm_dropout: float = 0.5

    # CNN-GRU specific
    gru_units: int = 64
    gru_spatial_dropout: float = 0.2
    gru_dropout: float = 0.1

    # Regularization
    use_dropout: bool = True
    dropout_rate: float = 0.5

    # Callbacks
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"

    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "f1_score",
        "precision",
        "recall"
    ])


@dataclass
class Config:
    """Main configuration object combining all configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        """Create necessary directories."""
        self.data.models_dir.mkdir(parents=True, exist_ok=True)
        self.data.images_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
