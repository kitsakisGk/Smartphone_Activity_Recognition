"""CNN-GRU hybrid model for activity recognition."""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, GRU, Dropout, SpatialDropout1D

from .base import BaseActivityModel
from ..config import ModelConfig


class CNNGRUModel(BaseActivityModel):
    """
    Hybrid CNN-GRU model for activity recognition.

    Combines CNN for spatial feature extraction with GRU for temporal modeling.
    GRU is lighter and faster than LSTM while maintaining similar performance.
    """

    def __init__(self, config: ModelConfig, num_classes: int):
        """
        Initialize CNN-GRU model.

        Args:
            config: Model configuration
            num_classes: Number of activity classes
        """
        super().__init__(config, num_classes, model_name="CNN_GRU")

    def build(self) -> Sequential:
        """
        Build the CNN-GRU hybrid architecture.

        Architecture:
            - Conv1D (64 filters) with SpatialDropout
            - MaxPooling1D
            - GRU (64 units) with Dropout
            - Dense output layer

        Returns:
            Compiled CNN-GRU model
        """
        self.logger.info("Building CNN-GRU hybrid model")

        model = Sequential(name=self.model_name)

        # Convolutional layer
        model.add(Conv1D(
            filters=64,
            kernel_size=self.config.cnn_kernel_size,
            activation="relu",
            input_shape=self.config.input_shape,
            name="conv1d"
        ))

        # Spatial dropout
        model.add(SpatialDropout1D(
            rate=self.config.gru_spatial_dropout,
            name="spatial_dropout"
        ))

        # Max pooling
        model.add(MaxPooling1D(
            pool_size=self.config.cnn_pool_size,
            name="maxpool"
        ))

        # GRU layer
        model.add(GRU(
            units=self.config.gru_units,
            name="gru"
        ))

        # Dropout
        model.add(Dropout(
            rate=self.config.gru_dropout,
            name="dropout"
        ))

        # Output layer
        model.add(Dense(
            units=self.num_classes,
            activation="softmax",
            name="output"
        ))

        # Compile the model
        model = self.compile_model(model)

        # Build the model explicitly before counting parameters
        model.build(input_shape=(None,) + self.config.input_shape)
        self.logger.info(f"CNN-GRU model built with {model.count_params():,} parameters")

        return model
