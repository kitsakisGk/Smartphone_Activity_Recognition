"""Standalone LSTM model for activity recognition."""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from .base import BaseActivityModel
from ..config import ModelConfig


class LSTMModel(BaseActivityModel):
    """
    Standalone LSTM model for activity recognition.

    Pure LSTM architecture for temporal sequence modeling.
    """

    def __init__(self, config: ModelConfig, num_classes: int):
        """
        Initialize LSTM model.

        Args:
            config: Model configuration
            num_classes: Number of activity classes
        """
        super().__init__(config, num_classes, model_name="LSTM")

    def build(self) -> Sequential:
        """
        Build the LSTM architecture.

        Architecture:
            - LSTM layer (128 units)
            - Dropout (0.5)
            - LSTM layer (64 units)
            - Dropout (0.5)
            - Dense layer (32 units)
            - Output layer

        Returns:
            Compiled LSTM model
        """
        self.logger.info("Building standalone LSTM model")

        model = Sequential(name=self.model_name)

        # First LSTM layer (return sequences for stacking)
        model.add(LSTM(
            units=128,
            return_sequences=True,
            input_shape=self.config.input_shape,
            name="lstm_1"
        ))
        model.add(Dropout(0.5, name="dropout_1"))

        # Second LSTM layer
        model.add(LSTM(
            units=64,
            return_sequences=False,
            name="lstm_2"
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

        self.logger.info(f"LSTM model built with {model.count_params():,} parameters")

        return model
