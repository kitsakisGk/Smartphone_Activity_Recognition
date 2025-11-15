"""Train baseline models (Random Forest, Logistic Regression)."""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data import SHLDataLoader, DataPreprocessor
from src.models.baseline import BaselineModel
from src.utils import setup_logging, get_logger


def train_baseline(model_type: str = "random_forest"):
    """
    Train a baseline model.

    Args:
        model_type: 'random_forest' or 'logistic_regression'
    """
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info(f"=" * 60)
    logger.info(f"TRAINING BASELINE MODEL: {model_type.upper()}")
    logger.info(f"=" * 60)

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

    # Prepare data
    logger.info("Preparing data...")
    X_train, X_val, y_train, y_val = preprocessor.prepare_data(X_all, y_all, num_classes)

    # Baseline models don't need reshaping - they work on flat features
    logger.info(f"Train data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    # Create baseline model (REDUCED for slower laptops)
    if model_type == "random_forest":
        model = BaselineModel(
            model_type="random_forest",
            n_estimators=50,  # Reduced from 100
            max_depth=15,     # Reduced from 20
            random_state=42,
            n_jobs=2          # Use only 2 CPU cores instead of all
        )
        save_name = "baseline_rf.pkl"
    else:  # logistic_regression
        model = BaselineModel(
            model_type="logistic_regression",
            max_iter=1000,
            random_state=42
        )
        save_name = "baseline_lr.pkl"

    # Train model
    logger.info("Training model...")
    model.train(X_train, y_train)

    # Evaluate on validation set
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    val_metrics = model.evaluate(X_val, y_val)

    # Save model
    save_path = config.data.models_dir / save_name
    model.save(save_path)

    # Feature importance (Random Forest only)
    if model_type == "random_forest":
        importance = model.get_feature_importance()
        if importance is not None:
            logger.info("\nTop 5 Most Important Features:")
            indices = importance.argsort()[-5:][::-1]
            feature_names = [
                "Accel_X", "Accel_Y", "Accel_Z",
                "Gyro_X", "Gyro_Y", "Gyro_Z",
                "Mag_X", "Mag_Y", "Mag_Z"
            ]
            for idx in indices:
                logger.info(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Baseline model saved to: {save_path}")
    logger.info("Training complete!")
    logger.info("=" * 60)

    return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Baseline model type (default: random_forest)"
    )

    args = parser.parse_args()

    train_baseline(args.model)
