"""Compare all models (baseline + deep learning)."""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data import SHLDataLoader, DataPreprocessor
from src.models import CNNModel, CNNLSTMModel, CNNGRUModel
from src.models.baseline import BaselineModel
from src.utils import setup_logging, get_logger


def compare_all_models():
    """
    Compare all models on the same test set.
    """
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("MODEL COMPARISON - BASELINE VS DEEP LEARNING")
    logger.info("=" * 70)

    # Load configuration
    config = get_config()

    # Initialize data loader and preprocessor
    data_loader = SHLDataLoader(config.data)
    preprocessor = DataPreprocessor(config.data)

    # Load all data
    logger.info("\nLoading dataset...")
    X_all, y_all = data_loader.load_all_data()

    # Get number of classes
    num_classes = data_loader.get_num_classes(y_all)

    # Prepare data
    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(X_all, y_all, num_classes)

    results = []

    # ===== BASELINE MODELS =====
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING BASELINE MODELS")
    logger.info("=" * 70)

    # Random Forest
    try:
        logger.info("\n[1/5] Random Forest Baseline")
        rf_path = config.data.models_dir / "baseline_rf.pkl"
        if rf_path.exists():
            rf_model = BaselineModel(model_type="random_forest")
            rf_model.load(rf_path)
            rf_metrics = rf_model.evaluate(X_test, y_test)
            results.append({
                "Model": "Random Forest (Baseline)",
                "Type": "Traditional ML",
                **rf_metrics
            })
        else:
            logger.warning(f"Model not found: {rf_path}")
    except Exception as e:
        logger.error(f"Error evaluating Random Forest: {e}")

    # Logistic Regression
    try:
        logger.info("\n[2/5] Logistic Regression Baseline")
        lr_path = config.data.models_dir / "baseline_lr.pkl"
        if lr_path.exists():
            lr_model = BaselineModel(model_type="logistic_regression")
            lr_model.load(lr_path)
            lr_metrics = lr_model.evaluate(X_test, y_test)
            results.append({
                "Model": "Logistic Regression (Baseline)",
                "Type": "Traditional ML",
                **lr_metrics
            })
        else:
            logger.warning(f"Model not found: {lr_path}")
    except Exception as e:
        logger.error(f"Error evaluating Logistic Regression: {e}")

    # ===== DEEP LEARNING MODELS =====
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING DEEP LEARNING MODELS")
    logger.info("=" * 70)

    # CNN
    try:
        logger.info("\n[3/5] CNN Model")
        cnn_path = config.data.models_dir / "CNN.h5"
        if cnn_path.exists():
            X_test_cnn = preprocessor.reshape_for_cnn(X_test)
            cnn_model = CNNModel(config.model, num_classes)
            cnn_model.load_weights(cnn_path)
            keras_model = cnn_model.get_model()

            cnn_results = keras_model.evaluate(X_test_cnn, y_test, verbose=0)
            results.append({
                "Model": "CNN (VGG-inspired)",
                "Type": "Deep Learning",
                "accuracy": cnn_results[1],
                "f1_score": cnn_results[2],
                "precision": cnn_results[3],
                "recall": cnn_results[4]
            })
            logger.info(f"Accuracy: {cnn_results[1]:.4f}")
        else:
            logger.warning(f"Model not found: {cnn_path}")
    except Exception as e:
        logger.error(f"Error evaluating CNN: {e}")

    # CNN-LSTM
    try:
        logger.info("\n[4/5] CNN-LSTM Model")
        lstm_path = config.data.models_dir / "CNN_LSTM.h5"
        if lstm_path.exists():
            X_test_lstm = preprocessor.reshape_for_lstm(X_test)
            lstm_model = CNNLSTMModel(config.model, num_classes)
            lstm_model.load_weights(lstm_path)
            keras_model = lstm_model.get_model()

            lstm_results = keras_model.evaluate(X_test_lstm, y_test, verbose=0)
            results.append({
                "Model": "CNN-LSTM (Hybrid)",
                "Type": "Deep Learning",
                "accuracy": lstm_results[1],
                "f1_score": lstm_results[2],
                "precision": lstm_results[3],
                "recall": lstm_results[4]
            })
            logger.info(f"Accuracy: {lstm_results[1]:.4f}")
        else:
            logger.warning(f"Model not found: {lstm_path}")
    except Exception as e:
        logger.error(f"Error evaluating CNN-LSTM: {e}")

    # CNN-GRU
    try:
        logger.info("\n[5/5] CNN-GRU Model")
        gru_path = config.data.models_dir / "CNN_GRU.h5"
        if gru_path.exists():
            X_test_gru = preprocessor.reshape_for_cnn(X_test)
            gru_model = CNNGRUModel(config.model, num_classes)
            gru_model.load_weights(gru_path)
            keras_model = gru_model.get_model()

            gru_results = keras_model.evaluate(X_test_gru, y_test, verbose=0)
            results.append({
                "Model": "CNN-GRU (Hybrid)",
                "Type": "Deep Learning",
                "accuracy": gru_results[1],
                "f1_score": gru_results[2],
                "precision": gru_results[3],
                "recall": gru_results[4]
            })
            logger.info(f"Accuracy: {gru_results[1]:.4f}")
        else:
            logger.warning(f"Model not found: {gru_path}")
    except Exception as e:
        logger.error(f"Error evaluating CNN-GRU: {e}")

    # ===== RESULTS SUMMARY =====
    if not results:
        logger.error("\nNo models found! Train models first.")
        return

    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON RESULTS")
    logger.info("=" * 70)

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)

    # Print table
    print("\n" + df.to_string(index=False))

    # Save to CSV
    output_path = config.data.images_dir / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    metrics = ["accuracy", "f1_score", "precision", "recall"]
    titles = ["Accuracy", "F1-Score", "Precision", "Recall"]

    for ax, metric, title in zip(axes, metrics, titles):
        colors = ['#ff7f0e' if t == "Traditional ML" else '#1f77b4' for t in df["Type"]]
        ax.barh(df["Model"], df[metric], color=colors)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plot_path = config.data.images_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {plot_path}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    compare_all_models()
