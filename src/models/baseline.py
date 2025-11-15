"""Baseline models for comparison with deep learning approaches."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..utils.logger import get_logger


class BaselineModel:
    """
    Traditional ML baseline model for activity recognition.

    Provides Random Forest and Logistic Regression baselines to compare
    against deep learning approaches.
    """

    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        Initialize baseline model.

        Args:
            model_type: Type of model ('random_forest' or 'logistic_regression')
            **kwargs: Additional parameters for sklearn model
        """
        self.model_type = model_type
        self.logger = get_logger(__name__)

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 20),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
                verbose=kwargs.get("verbose", 1)
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                random_state=kwargs.get("random_state", 42),
                n_jobs=kwargs.get("n_jobs", -1),
                verbose=kwargs.get("verbose", 1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"Initialized {model_type} baseline model")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the baseline model.

        Args:
            X_train: Training features (samples, features)
            y_train: Training labels (samples,) - class indices, not one-hot
        """
        self.logger.info(f"Training {self.model_type} on {len(X_train)} samples")

        # If y_train is one-hot encoded, convert to class indices
        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)

        self.model.fit(X_train, y_train)
        self.logger.info("Training complete")

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels (can be one-hot or class indices)

        Returns:
            Dictionary of metrics
        """
        # Convert one-hot to class indices if needed
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)

        predictions = self.model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions, average="macro"),
            "precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "recall": recall_score(y_test, predictions, average="macro", zero_division=0)
        }

        self.logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted class indices
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (Random Forest only).

        Returns:
            Feature importance array or None
        """
        if self.model_type == "random_forest":
            return self.model.feature_importances_
        else:
            self.logger.warning("Feature importance only available for Random Forest")
            return None
