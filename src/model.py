"""
Unified model training and inference wrappers for LightGBM and XGBoost classifiers.
"""

import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import lightgbm as lgb
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.config import Config


class BaseERModel:
    """Base interface for Entity Resolution models."""
    def __init__(self, config: Config):
        self.config = config
        self.feature_names = config.feature_names
        self.best_iteration: int = 0
        self.training_time: float = 0.0

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, model_path: str):
        raise NotImplementedError

    def load(self, model_path: str):
        raise NotImplementedError


class LightGBMERModel(BaseERModel):
    """LightGBM classifier wrapper for entity resolution match scoring."""
    def __init__(self, config: Config):
        super().__init__(config)
        self.model: Optional[lgb.Booster] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        print(f"[LightGBM] Training on {len(X_train):,} pairs (Pos: {int(y_train.sum()):,}, Neg: {int(len(y_train) - y_train.sum()):,})")
        t0 = time.time()
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        
        valid_sets = [train_data]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=50)]

        if X_val is not None and y_val is not None and len(X_val) > 0:
            print(f"[LightGBM] Validating on {len(X_val):,} pairs (Pos: {int(y_val.sum()):,}, Neg: {int(len(y_val) - y_val.sum()):,})")
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False))

        params = self.config.lgb_params.copy()
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        t1 = time.time()
        self.training_time = t1 - t0
        self.best_iteration = self.model.best_iteration or self.model.current_iteration()
        print(f"[LightGBM] Training finished in {self.training_time:.2f}s. Best iteration: {self.best_iteration}")

        importances = self.model.feature_importance(importance_type="gain")
        imp_dict = dict(sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True))

        return {
            "best_iteration": self.best_iteration,
            "feature_importances": imp_dict,
            "training_time": self.training_time
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("LightGBM model is not loaded or trained.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return self.model.predict(X, num_iteration=self.best_iteration)

    def save(self, model_path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self.model.save_model(model_path)
        print(f"[LightGBM] Saved model to {model_path}")

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = lgb.Booster(model_file=model_path)
        self.best_iteration = self.model.best_iteration or self.model.current_iteration()
        print(f"[LightGBM] Loaded model from {model_path}")


class XGBoostERModel(BaseERModel):
    """XGBoost classifier wrapper for entity resolution match scoring."""
    def __init__(self, config: Config):
        super().__init__(config)
        if xgb is None:
            raise ImportError("XGBoost is not installed. Please install it using: pip install xgboost")
        self.model: Optional[xgb.XGBClassifier] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        print(f"[XGBoost] Training on {len(X_train):,} pairs (Pos: {int(y_train.sum()):,}, Neg: {int(len(y_train) - y_train.sum()):,})")
        t0 = time.time()

        eval_set = None
        early_stopping = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            print(f"[XGBoost] Validating on {len(X_val):,} pairs (Pos: {int(y_val.sum()):,}, Neg: {int(len(y_val) - y_val.sum()):,})")
            eval_set = [(X_train, y_train), (X_val, y_val)]
            early_stopping = self.config.early_stopping_rounds

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.config.seed,
            n_jobs=4,
            early_stopping_rounds=early_stopping
        )

        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=50
            )
            self.best_iteration = getattr(self.model, "best_iteration", 500)
        else:
            self.model.fit(
                X_train,
                y_train,
                verbose=50
            )
            self.best_iteration = 500

        t1 = time.time()
        self.training_time = t1 - t0
        print(f"[XGBoost] Training finished in {self.training_time:.2f}s. Best iteration: {self.best_iteration}")

        importances = self.model.feature_importances_
        imp_dict = dict(sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True))

        return {
            "best_iteration": self.best_iteration,
            "feature_importances": imp_dict,
            "training_time": self.training_time
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("XGBoost model is not loaded or trained.")
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        # Return probability of class 1
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def save(self, model_path: str):
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self.model.save_model(model_path)
        print(f"[XGBoost] Saved model to {model_path}")

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"[XGBoost] Loaded model from {model_path}")


def get_model(model_type: str, config: Config) -> BaseERModel:
    """Factory function to instantiate models."""
    m_type = model_type.lower().strip()
    if m_type == "lightgbm":
        return LightGBMERModel(config)
    elif m_type == "xgboost":
        return XGBoostERModel(config)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Expected 'lightgbm' or 'xgboost'.")


# Backwards compatibility alias
ERModel = LightGBMERModel
