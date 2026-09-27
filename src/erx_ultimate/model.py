"""
ER-X Ultimate: LightGBM GBDT Training Engine & Isotonic Probability Calibration
"""

from __future__ import annotations
import gc
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.features import NUM_FEATURES

logger = logging.getLogger("erx_ultimate.model")


class ERXModelEngine:
    """Production LightGBM engine with memmap streaming and probability calibration."""

    def __init__(self, config: UltimateConfig = CONFIG):
        self.config = config
        self.models_dir = Path(config.paths.artifacts_dir) / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.lgb_model: Optional[lgb.Booster] = None
        self.calibrator: Optional[IsotonicRegression] = None

    def train_from_memmap(
        self,
        mmap_x_path: Path,
        mmap_y_path: Path,
        num_samples: int,
        val_x: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> None:
        """Train LightGBM directly on disk-backed memory mapped feature arrays."""
        logger.info(f"Connecting to memory-mapped training data ({num_samples:,} samples, {NUM_FEATURES} features)...")
        
        X_train = np.memmap(mmap_x_path, dtype=np.float32, mode="r", shape=(num_samples, NUM_FEATURES))
        y_train = np.memmap(mmap_y_path, dtype=np.uint8, mode="r", shape=(num_samples,))

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            free_raw_data=self.config.model.free_raw_data,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if val_x is not None and val_y is not None:
            val_data = lgb.Dataset(val_x, label=val_y, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("val")

        params: Dict[str, Any] = {
            "objective": self.config.model.objective,
            "metric": self.config.model.metric,
            "boosting_type": self.config.model.boosting_type,
            "learning_rate": self.config.model.learning_rate,
            "num_leaves": self.config.model.num_leaves,
            "max_depth": self.config.model.max_depth,
            "min_child_samples": self.config.model.min_child_samples,
            "subsample": self.config.model.subsample,
            "colsample_bytree": self.config.model.colsample_bytree,
            "max_bin": self.config.model.max_bin,
            "n_jobs": self.config.model.n_jobs,
            "verbose": -1,
            "seed": self.config.system.random_seed,
        }

        logger.info("Training production LightGBM Booster...")
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.model.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.log_evaluation(period=100)],
        )
        logger.info("LightGBM training complete.")
        del train_data
        gc.collect()

    def fit_calibrator(self, raw_probs: np.ndarray, y_true: np.ndarray) -> None:
        """Fit Isotonic Regression probability calibrator on OOF predictions."""
        logger.info(f"Fitting Isotonic Calibrator on {len(raw_probs):,} samples...")
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw_probs, y_true)
        logger.info("Isotonic calibration fitted successfully.")

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate raw model probabilities and calibrated probabilities."""
        if self.lgb_model is None:
            raise ValueError("Model is not loaded or trained.")

        raw_probs = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)
        if self.calibrator is not None:
            calibrated_probs = self.calibrator.predict(raw_probs)
        else:
            calibrated_probs = raw_probs

        return raw_probs, calibrated_probs

    def save_artifacts(self) -> None:
        """Persist LightGBM booster and Isotonic calibrator to disk."""
        model_path = self.models_dir / "lgb_booster.txt"
        if self.lgb_model is not None:
            self.lgb_model.save_model(str(model_path))
            logger.info(f"Saved LightGBM model to {model_path}")

        calib_path = self.models_dir / "isotonic_calibrator.pkl"
        if self.calibrator is not None:
            with open(calib_path, "wb") as f:
                pickle.dump(self.calibrator, f)
            logger.info(f"Saved Isotonic calibrator to {calib_path}")

    def load_artifacts(self) -> None:
        """Load LightGBM model and Isotonic calibrator from disk."""
        model_path = self.models_dir / "lgb_booster.txt"
        if model_path.exists():
            self.lgb_model = lgb.Booster(model_file=str(model_path))
            logger.info(f"Loaded LightGBM model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        calib_path = self.models_dir / "isotonic_calibrator.pkl"
        if calib_path.exists():
            with open(calib_path, "rb") as f:
                self.calibrator = pickle.load(f)
            logger.info(f"Loaded Isotonic calibrator from {calib_path}")
