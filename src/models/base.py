"""Base model class for activity recognition models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from ..config import ModelConfig
from ..utils.logger import get_logger


class BaseActivityModel(ABC):
    """
    Abstract base class for activity recognition models.

    All models should inherit from this class and implement the build() method.
    """

    def __init__(self, config: ModelConfig, num_classes: int, model_name: str):
        """
        Initialize the base model.

        Args:
            config: Model configuration
            num_classes: Number of activity classes to predict
            model_name: Name of the model (for logging and saving)
        """
        self.config = config
        self.num_classes = num_classes
        self.model_name = model_name
        self.model: Optional[Sequential] = None
        self.logger = get_logger(__name__)

    @abstractmethod
    def build(self) -> Sequential:
        """
        Build and return the model architecture.

        Returns:
            Compiled Keras Sequential model
        """
        pass

    def compile_model(self, model: Sequential) -> Sequential:
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            model: Uncompiled Keras model

        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Compiling {self.model_name} model")

        # Create optimizer
        optimizer = Adam(learning_rate=self.config.learning_rate)

        # Define metrics
        metrics = [
            "accuracy",
            keras.metrics.F1Score(average="macro", name="f1_score"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.categorical_crossentropy,
            metrics=metrics
        )

        return model

    def get_model(self) -> Sequential:
        """
        Get or build the model.

        Returns:
            Compiled Keras model
        """
        if self.model is None:
            self.model = self.build()

        return self.model

    def summary(self) -> None:
        """Print model summary."""
        model = self.get_model()
        self.logger.info(f"\n{self.model_name} Architecture:")
        model.summary()

    def get_callbacks(self, save_path: Path) -> list:
        """
        Get training callbacks.

        Args:
            save_path: Path to save model weights

        Returns:
            List of Keras callbacks
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                filepath=str(save_path),
                save_weights_only=True,
                monitor=self.config.monitor_metric,
                mode="max",
                save_best_only=self.config.save_best_only,
                verbose=1
            ),
            EarlyStopping(
                monitor=self.config.monitor_metric,
                min_delta=0,
                patience=self.config.patience,
                verbose=1,
                mode="auto"
            )
        ]

        return callbacks

    def save_weights(self, path: Path) -> None:
        """
        Save model weights.

        Args:
            path: Path to save weights
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(path))
        self.logger.info(f"Model weights saved to {path}")

    def load_weights(self, path: Path) -> None:
        """
        Load model weights.

        Args:
            path: Path to load weights from
        """
        if self.model is None:
            self.model = self.build()

        self.model.load_weights(str(path))
        self.logger.info(f"Model weights loaded from {path}")
