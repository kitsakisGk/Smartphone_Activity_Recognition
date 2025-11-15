"""CNN model for activity recognition (VGG-16 inspired)."""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense

from .base import BaseActivityModel
from ..config import ModelConfig


class CNNModel(BaseActivityModel):
    """
    Convolutional Neural Network for activity recognition.

    Architecture inspired by VGG-16 with Conv1D layers adapted for sensor data.
    """

    def __init__(self, config: ModelConfig, num_classes: int):
        """
        Initialize CNN model.

        Args:
            config: Model configuration
            num_classes: Number of activity classes
        """
        super().__init__(config, num_classes, model_name="CNN")

    def build(self) -> Sequential:
        """
        Build the CNN architecture.

        Architecture:
            - Conv1D (64 filters) -> Conv1D (32 filters) -> MaxPool1D
            - Conv1D (64 filters) -> Conv1D (64 filters) -> MaxPool1D
            - Flatten -> Dense (512) -> Dense (256) -> Output

        Returns:
            Compiled CNN model
        """
        self.logger.info("Building CNN model (VGG-16 inspired)")

        model = Sequential(name=self.model_name)

        # First convolutional block
        model.add(Conv1D(
            filters=64,
            kernel_size=self.config.cnn_kernel_size,
            padding="same",
            activation="relu",
            input_shape=self.config.input_shape,
            name="conv1d_1"
        ))
        model.add(Conv1D(
            filters=32,
            kernel_size=self.config.cnn_kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_2"
        ))
        model.add(MaxPool1D(
            pool_size=self.config.cnn_pool_size,
            strides=2,
            name="maxpool_1"
        ))

        # Second convolutional block
        model.add(Conv1D(
            filters=64,
            kernel_size=self.config.cnn_kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_3"
        ))
        model.add(Conv1D(
            filters=64,
            kernel_size=self.config.cnn_kernel_size,
            padding="same",
            activation="relu",
            name="conv1d_4"
        ))
        model.add(MaxPool1D(
            pool_size=self.config.cnn_pool_size,
            strides=2,
            name="maxpool_2"
        ))

        # Fully connected layers
        model.add(Flatten(name="flatten"))
        model.add(Dense(
            units=512,
            activation="relu",
            name="dense_1"
        ))
        model.add(Dense(
            units=256,
            activation="relu",
            name="dense_2"
        ))

        # Output layer
        model.add(Dense(
            units=self.num_classes,
            activation="softmax",
            name="output"
        ))

        # Compile the model
        model = self.compile_model(model)

        self.logger.info(f"CNN model built with {model.count_params():,} parameters")

        return model
