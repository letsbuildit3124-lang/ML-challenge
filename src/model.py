"""
LightGBM binary classification model training and inference.
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import lightgbm as lgb
from src.config import Config

class ERModel:
    """
    LightGBM classifier wrapper for entity resolution match scoring.
    """
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self.feature_names = config.feature_names

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Trains the LightGBM model with early stopping.
        """
        print(f"[Model Training] Training on {len(X_train):,} pairs (Positives: {int(y_train.sum()):,}, Negatives: {int(len(y_train) - y_train.sum()):,})")
        print(f"[Model Training] Validating on {len(X_val):,} pairs (Positives: {int(y_val.sum()):,}, Negatives: {int(len(y_val) - y_val.sum()):,})")

        t0 = time.time()
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50)
        ]

        params = self.config.lgb_params.copy()
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks
        )

        t1 = time.time()
        print(f"[Model Training] Training finished in {t1 - t0:.2f}s. Best iteration: {self.model.best_iteration}")

        # Feature importances
        importances = self.model.feature_importance(importance_type="gain")
        imp_dict = dict(sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True))

        print("\n[Feature Importances (Gain)]:")
        for fname, imp in list(imp_dict.items())[:10]:
            print(f"  {fname:<30}: {imp:,.1f}")

        # Save model
        os.makedirs(self.config.models_dir, exist_ok=True)
        self.model.save_model(self.config.model_save_path)
        print(f"[Model] Saved model to {self.config.model_save_path}")

        return {
            "best_iteration": self.model.best_iteration,
            "feature_importances": imp_dict,
            "training_time": t1 - t0
        }

    def load(self, model_path: Optional[str] = None):
        path = model_path or self.config.model_save_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = lgb.Booster(model_file=path)
        print(f"[Model] Loaded model from {path}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
