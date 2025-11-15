"""Standalone GRU model for activity recognition."""

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

from .base import BaseActivityModel
from ..config import ModelConfig


class GRUModel(BaseActivityModel):
    """
    Standalone GRU model for activity recognition.

    Pure GRU architecture - lighter and faster than LSTM.
    """

    def __init__(self, config: ModelConfig, num_classes: int):
        """
        Initialize GRU model.

        Args:
            config: Model configuration
            num_classes: Number of activity classes
        """
        super().__init__(config, num_classes, model_name="GRU")

    def build(self) -> Sequential:
        """
        Build the GRU architecture.

        Architecture:
            - GRU layer (128 units)
            - Dropout (0.5)
            - GRU layer (64 units)
            - Dropout (0.5)
            - Dense layer (32 units)
            - Output layer

        Returns:
            Compiled GRU model
        """
        self.logger.info("Building standalone GRU model")

        model = Sequential(name=self.model_name)

        # First GRU layer (return sequences for stacking)
        model.add(GRU(
            units=128,
            return_sequences=True,
            input_shape=self.config.input_shape,
            name="gru_1"
        ))
        model.add(Dropout(0.5, name="dropout_1"))

        # Second GRU layer
        model.add(GRU(
            units=64,
            return_sequences=False,
            name="gru_2"
        ))
        model.add(Dropout(0.5, name="dropout_2"))

        # Dense layer
        model.add(Dense(
            units=32,
            activation="relu",
            name="dense"
        ))

        # Output layer
        model.add(Dense(
            units=self.num_classes,
            activation="softmax",
            name="output"
        ))

        # Compile the model
        model = self.compile_model(model)

        self.logger.info(f"GRU model built with {model.count_params():,} parameters")

        return model
