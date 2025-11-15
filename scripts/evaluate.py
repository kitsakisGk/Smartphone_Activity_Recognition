"""Evaluation script for trained models."""

import sys
from pathlib import Path
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data import SHLDataLoader, DataPreprocessor
from src.models import CNNModel, CNNLSTMModel, CNNGRUModel
from src.utils import setup_logging, get_logger


def evaluate_model(model_type: str = "cnn", weights_path: Path = None):
    """
    Evaluate a trained model.

    Args:
        model_type: Type of model ('cnn', 'cnn_lstm', or 'cnn_gru')
        weights_path: Path to model weights file
    """
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info(f"Evaluating {model_type.upper()} model")

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

    # Prepare data
    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X_all, y_all, num_classes)

    # Reshape data based on model type
    if model_type == "cnn":
        X_test = preprocessor.reshape_for_cnn(X_test)
        model_class = CNNModel
        default_weights = config.data.models_dir / "CNN.h5"
    elif model_type == "cnn_lstm":
        X_test = preprocessor.reshape_for_lstm(X_test)
        model_class = CNNLSTMModel
        default_weights = config.data.models_dir / "CNN_LSTM.h5"
    elif model_type == "cnn_gru":
        X_test = preprocessor.reshape_for_cnn(X_test)
        model_class = CNNGRUModel
        default_weights = config.data.models_dir / "CNN_GRU.h5"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if weights_path is None:
        weights_path = default_weights

    # Build and load model
    logger.info(f"Loading {model_type.upper()} model from {weights_path}")
    model = model_class(config.model, num_classes)
    model.load_weights(weights_path)

    keras_model = model.get_model()

    # Evaluate
    logger.info("Evaluating model...")
    results = keras_model.evaluate(X_test, y_test, verbose=1)

    # Make predictions for detailed analysis
    predictions = keras_model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Calculate per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix

    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Test Loss: {results[0]:.4f}")
    logger.info(f"Test Accuracy: {results[1]:.4f}")
    logger.info(f"Test F1-Score: {results[2]:.4f}")
    logger.info(f"Test Precision: {results[3]:.4f}")
    logger.info(f"Test Recall: {results[4]:.4f}")
    logger.info("=" * 50)

    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(true_classes, predicted_classes))

    logger.info("\nConfusion Matrix:")
    logger.info("\n" + str(confusion_matrix(true_classes, predicted_classes)))

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "cnn_lstm", "cnn_gru"],
        help="Model type to evaluate (default: cnn)"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to model weights file (optional)"
    )

    args = parser.parse_args()

    evaluate_model(args.model, args.weights)
