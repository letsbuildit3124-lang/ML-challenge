"""
ER-X GBDT Pairwise Matcher & Probability Calibration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.erx.config import ERXConfig
from src.erx.features import FEATURE_NAMES

logger = logging.getLogger("erx.model")


class ERXCalibrator:
    """Post-hoc probability calibrator supporting Platt scaling and Isotonic regression."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.calibrator: Optional[Any] = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """Fits calibration curve on validation raw probabilities."""
        if len(raw_probs) == 0:
            return
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_probs, y_true)
        else:  # Platt scaling
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(raw_probs.reshape(-1, 1), y_true)

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrates raw probabilities."""
        if self.calibrator is None or len(raw_probs) == 0:
            return raw_probs
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)
        else:
            return self.calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]


class ERXModelTrainer:
    """Trains and manages LightGBM pairwise matching models."""

    def __init__(self, config: ERXConfig):
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = FEATURE_NAMES
        self.calibrator = ERXCalibrator(method="platt")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Trains LightGBM model with validation monitoring."""
        logger.info(f"Training LightGBM on {X_train.shape[0]:,} pairs with {X_train.shape[1]} features...")
        logger.info(f"Positive labels: {int(np.sum(y_train)):,}, Negative labels: {int(len(y_train) - np.sum(y_train)):,}")

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names, free_raw_data=False)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=train_data, free_raw_data=False)
            valid_sets.append(val_data)
            valid_names.append("valid")

        params = dict(self.config.lgb_params)
        num_boost_round = params.pop("n_estimators", 600)

        callbacks = [
            lgb.log_evaluation(period=100),
        ]
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(stopping_rounds=40, verbose=True))

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Fit calibrator on validation predictions if validation set is supplied
        if X_val is not None and y_val is not None and len(X_val) > 0:
            raw_val_probs = self.model.predict(X_val)
            self.calibrator.fit(raw_val_probs, y_val)
            logger.info("Probability calibrator fitted on validation split.")

        # Extract top feature importances
        importance_vals = self.model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(self.feature_names, importance_vals), key=lambda x: x[1], reverse=True)

        logger.info("Top 15 most important features by Gain:")
        for feat, imp in feat_imp[:15]:
            logger.info(f"  {feat:30s}: {imp:,.2f}")

        return {
            "best_iteration": self.model.best_iteration,
            "feature_importances": {f: float(i) for f, i in feat_imp},
        }

    def predict_proba(self, X: np.ndarray, calibrate: bool = True) -> np.ndarray:
        """Predicts matching probabilities for pair feature matrix."""
        if self.model is None or len(X) == 0:
            return np.empty((0,), dtype=np.float32)
        raw_probs = self.model.predict(X)
        if calibrate:
            return self.calibrator.predict(raw_probs)
        return raw_probs

    def save(self, model_path: Path) -> None:
        """Saves LightGBM booster model to file."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save_model(str(model_path))
            logger.info(f"Saved LightGBM model to {model_path}")

    def load(self, model_path: Path) -> None:
        """Loads LightGBM booster model from file."""
        self.model = lgb.Booster(model_file=str(model_path))
        logger.info(f"Loaded LightGBM model from {model_path}")
