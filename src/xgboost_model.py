"""
V3 XGBoost Classifier Module for Business Entity Resolution.
Supports:
- CPU histogram tree building ('hist')
- Entity-level Macro F0.5 evaluation
- Model checkpoint serialization and inference.
"""

import os
import time
from typing import Dict, List, Optional, Any
import numpy as np
import xgboost as xgb

from src.config import Config

class XGBoostERModel:
    """XGBoost classifier wrapper for pairwise match probability scoring."""
    def __init__(self, config: Config, custom_params: Optional[Dict[str, Any]] = None):
        self.config = config
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "random_state": config.seed,
            "n_jobs": 2
        }
        if custom_params:
            self.params.update(custom_params)
        self.model: Optional[xgb.XGBClassifier] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Trains the XGBoost model on pair feature matrix."""
        print(f"[XGBoost] Training on {len(X_train):,} pairs (Pos: {int(y_train.sum()):,}, Neg: {int(len(y_train) - y_train.sum()):,})")
        t0 = time.time()

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=50
        )

        train_time = time.time() - t0
        best_iter = getattr(self.model, "best_iteration", self.params["n_estimators"])
        print(f"[XGBoost] Training finished in {train_time:.2f}s. Best iteration: {best_iter}")

        # Feature importances
        importances = self.model.feature_importances_
        feat_names = getattr(self.config, "feature_names", [f"f_{i}" for i in range(len(importances))])
        importance_dict = {f: float(imp) for f, imp in zip(feat_names, importances)}

        return {
            "train_time_seconds": train_time,
            "best_iteration": int(best_iter) if best_iter is not None else 0,
            "feature_importances": sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts positive class match probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded!")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        # Return probability of positive match (class 1)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, model_path: str):
        """Serializes model to JSON / UBJSON format."""
        if self.model is None:
            raise ValueError("No model to save!")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        print(f"[XGBoost] Saved model checkpoint to {model_path}")

    def load(self, model_path: str):
        """Loads model from saved checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"[XGBoost] Loaded model checkpoint from {model_path}")
