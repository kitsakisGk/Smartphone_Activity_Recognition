"""Training script for activity recognition models."""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data import SHLDataLoader, DataPreprocessor
from src.models import CNNModel, LSTMModel, GRUModel, CNNLSTMModel, CNNGRUModel
from src.utils import setup_logging, get_logger, save_metrics_plot


def train_model(model_type: str = "cnn", config_path: Path = None):
    """
    Train a model for activity recognition.

    Args:
        model_type: Type of model ('cnn', 'cnn_lstm', or 'cnn_gru')
        config_path: Optional path to config file
    """
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info(f"Starting training for {model_type.upper()} model")

    # Load configuration
    config = get_config()

    # Initialize data loader and preprocessor
    data_loader = SHLDataLoader(config.data)
    preprocessor = DataPreprocessor(config.data)

    # Load all data
    logger.info("Loading dataset...")
    X_all, y_all = data_loader.load_all_data()

    # Get number of classes
    num_classes = data_loader.get_num_classes(y_all)

    # REDUCE DATASET SIZE for slower laptops (use 30% of data)
    import numpy as np
    sample_size = int(len(X_all) * 0.3)  # Use only 30% of data
    indices = np.random.RandomState(42).choice(len(X_all), sample_size, replace=False)
    X_all = X_all.iloc[indices].reset_index(drop=True)
    y_all = y_all.iloc[indices].reset_index(drop=True)
    logger.info(f"⚡ Using reduced dataset: {len(X_all):,} samples (30% of full data)")

    # Prepare data (normalize and split)
    logger.info("Preparing data...")
    X_train, X_val, y_train, y_val = preprocessor.prepare_data(X_all, y_all, num_classes)

    # Reshape data based on model type
    if model_type == "cnn":
        X_train = preprocessor.reshape_for_cnn(X_train)
        X_val = preprocessor.reshape_for_cnn(X_val)
        model_class = CNNModel
        save_name = "CNN.h5"
    elif model_type == "lstm":
        X_train = preprocessor.reshape_for_cnn(X_train)  # LSTM needs (samples, timesteps, features)
        X_val = preprocessor.reshape_for_cnn(X_val)
        model_class = LSTMModel
        save_name = "LSTM.h5"
    elif model_type == "gru":
        X_train = preprocessor.reshape_for_cnn(X_train)  # GRU needs (samples, timesteps, features)
        X_val = preprocessor.reshape_for_cnn(X_val)
        model_class = GRUModel
        save_name = "GRU.h5"
    elif model_type == "cnn_lstm":
        X_train = preprocessor.reshape_for_cnn(X_train)  # CNN-LSTM uses Conv1D, needs (samples, features, 1)
        X_val = preprocessor.reshape_for_cnn(X_val)
        model_class = CNNLSTMModel
        save_name = "CNN_LSTM.h5"
    elif model_type == "cnn_gru":
        X_train = preprocessor.reshape_for_cnn(X_train)
        X_val = preprocessor.reshape_for_cnn(X_val)
        model_class = CNNGRUModel
        save_name = "CNN_GRU.h5"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    # Build model
    logger.info(f"Building {model_type.upper()} model...")
    model = model_class(config.model, num_classes)
    model.summary()

    # Get model and callbacks
    keras_model = model.get_model()
    save_path = config.data.models_dir / save_name
    callbacks = model.get_callbacks(save_path)

    # Train model
    logger.info("Starting training...")
    history = keras_model.fit(
        X_train,
        y_train,
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save plots
    logger.info("Saving training plots...")
    save_metrics_plot(history.history, model_type.upper(), config.data.images_dir, "accuracy")
    save_metrics_plot(history.history, model_type.upper(), config.data.images_dir, "metrics")

    # Evaluate final model
    logger.info("Evaluating model on validation set...")
    results = keras_model.evaluate(X_val, y_val, verbose=0)

    logger.info("\n" + "=" * 50)
    logger.info("FINAL RESULTS")
    logger.info("=" * 50)
    logger.info(f"Validation Loss: {results[0]:.4f}")
    logger.info(f"Validation Accuracy: {results[1]:.4f}")
    logger.info(f"Validation F1-Score: {results[2]:.4f}")
    logger.info(f"Validation Precision: {results[3]:.4f}")
    logger.info(f"Validation Recall: {results[4]:.4f}")
    logger.info("=" * 50)

    logger.info(f"Model weights saved to: {save_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train activity recognition model")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "lstm", "gru", "cnn_lstm", "cnn_gru"],
        help="Model type to train (default: cnn)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (optional)"
    )

    args = parser.parse_args()

    train_model(args.model, args.config)
