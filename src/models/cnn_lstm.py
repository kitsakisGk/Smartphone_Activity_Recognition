"""CNN-LSTM hybrid model for activity recognition."""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Dropout

from .base import BaseActivityModel
from ..config import ModelConfig


class CNNLSTMModel(BaseActivityModel):
    """
    Hybrid CNN-LSTM model for activity recognition.

    Combines CNN for spatial feature extraction with LSTM for temporal modeling.
    """

    def __init__(self, config: ModelConfig, num_classes: int):
        """
        Initialize CNN-LSTM model.

        Args:
            config: Model configuration
            num_classes: Number of activity classes
        """
        super().__init__(config, num_classes, model_name="CNN_LSTM")

    def build(self) -> Sequential:
        """
        Build the CNN-LSTM hybrid architecture.

        Architecture:
            - Conv1D layers (spatial feature extraction)
            - LSTM layer (temporal modeling)
            - Dense layers (classification)

        Returns:
            Compiled CNN-LSTM model
        """
        self.logger.info("Building CNN-LSTM hybrid model")

        model = Sequential(name=self.model_name)

        # CNN layers for feature extraction
        model.add(Conv1D(
            filters=64,
            kernel_size=self.config.cnn_kernel_size,
            activation="relu",
            padding="same",
            input_shape=self.config.input_shape,
            name="conv1d_1"
        ))
        model.add(Dropout(0.3, name="dropout_1"))

        model.add(Conv1D(
            filters=64,
            kernel_size=self.config.cnn_kernel_size,
            activation="relu",
            padding="same",
            name="conv1d_2"
        ))
        model.add(Dropout(0.3, name="dropout_2"))

        model.add(MaxPooling1D(
            pool_size=self.config.cnn_pool_size,
            name="maxpool"
        ))

        # LSTM layer for temporal modeling
        model.add(LSTM(
            units=self.config.lstm_units,
            return_sequences=False,
            name="lstm"
        ))
        model.add(Dropout(self.config.lstm_dropout, name="lstm_dropout"))

        # Dense layers for classification
        model.add(Dense(
            units=self.config.lstm_dense_units,
            activation="relu",
            name="dense_1"
        ))
        model.add(Dense(
            units=self.num_classes,
            activation="softmax",
            name="output"
        ))

        # Compile the model
        model = self.compile_model(model)

        # Build the model explicitly before counting parameters
        model.build(input_shape=(None,) + self.config.input_shape)
        self.logger.info(f"CNN-LSTM model built with {model.count_params():,} parameters")

        return model
