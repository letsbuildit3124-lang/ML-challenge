"""
Configuration dataclasses and settings for ER-X Entity Resolution Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ERXConfig:
    # Random seeds for 3-seed validation
    validation_seeds: List[int] = field(default_factory=lambda: [42, 123, 2026])
    val_ratio: float = 0.20

    # Base filesystem paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path("dataset"))
    cache_dir: Path = field(default_factory=lambda: Path("cache/erx"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # DuckDB settings
    duckdb_threads: int = 8
    duckdb_memory_limit: str = "24GB"
    duckdb_cache_db: Path = field(default_factory=lambda: Path("cache/entity_resolution.duckdb"))

    # Learned rule parameters
    min_alias_observations: int = 3
    min_alias_purity: float = 0.85
    min_char_confusion_obs: int = 4
    min_char_confusion_purity: float = 0.80

    # Retrieval channel hyperparameters
    tfidf_ngram_range: tuple = (3, 5)
    tfidf_sublinear_tf: bool = True
    tfidf_max_features: int = 150_000
    tfidf_min_df: int = 2
    tfidf_top_k: int = 15

    rare_token_max_posting_size: int = 25_000
    rare_token_min_idf: float = 2.0
    rare_token_top_k: int = 15

    address_top_k: int = 10
    phonetic_top_k: int = 10
    max_total_candidates_per_target: int = 35

    # Hard-negative mining parameters
    hard_negatives_per_positive: int = 8
    max_train_samples: Optional[int] = None

    # GBDT Model hyperparameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "n_estimators": 600,
        "n_jobs": 8,
        "random_state": 42,
        "verbose": -1,
    })

    # Decision thresholds and exclusivity
    target_exclusivity: bool = True
    match_threshold: float = 0.50
    margin_threshold: float = 0.10
    s2_match_threshold: float = 0.50
    s3_match_threshold: float = 0.50

    # Batching and workers
    target_chunk_size: int = 50_000
    rapidfuzz_workers: int = 8

    def ensure_directories(self) -> None:
        """Creates all required directories safely."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "indexes").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "candidates").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "models").mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
