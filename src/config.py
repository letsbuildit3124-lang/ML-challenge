"""
Configuration module for the Business Entity Resolution V1 Baseline.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Config:
    # Random seed for reproducibility
    seed: int = 42

    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "dataset")
    train_dir: str = os.path.join(data_dir, "train")
    test_dir: str = os.path.join(data_dir, "test")
    models_dir: str = os.path.join(base_dir, "models")
    output_dir: str = os.path.join(base_dir, "output")
    reports_dir: str = os.path.join(base_dir, "reports")
    experiments_dir: str = os.path.join(base_dir, "experiments")
    utils_dir: str = os.path.join(base_dir, "utils")

    # Specific file paths
    train_s1_path: str = os.path.join(train_dir, "train_source1.tsv")
    train_s2_path: str = os.path.join(train_dir, "train_source2.tsv")
    train_s3_path: str = os.path.join(train_dir, "train_source3.tsv")
    train_gt_path: str = os.path.join(train_dir, "train_ground_truth.tsv")

    test_s1_path: str = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path: str = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path: str = os.path.join(test_dir, "test_source3.tsv")

    model_save_path: str = os.path.join(models_dir, "lightgbm_baseline.txt")
    matching_results_path: str = os.path.join(output_dir, "matching_results.tsv")
    candidate_pairs_path: str = os.path.join(output_dir, "candidate_pairs.tsv")
    output_matching_path: str = os.path.join(output_dir, "matching_results.tsv")
    output_candidates_path: str = os.path.join(output_dir, "candidate_pairs.tsv")
    validator_path: str = os.path.join(utils_dir, "validate_submission.py")

    # Validation Split
    train_s1_ratio: float = 0.80
    val_s1_ratio: float = 0.20
    
    # Scale & sampling controls for train dataset construction
    train_sample_s1_count: int = 10000
    val_sample_s1_count: int = 2500

    # Blocking Configuration
    max_cands_per_key: int = 40
    max_total_cands_per_s1: int = 40

    # LightGBM Parameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_estimators": 400,
        "n_jobs": 4,
        "verbose": -1,
    })
    early_stopping_rounds: int = 30

    # Threshold Search Grid
    threshold_grid: List[float] = field(default_factory=lambda: [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ])
    
    # Selected best threshold (updated after validation evaluation)
    selected_threshold: float = 0.50

    # Feature List
    feature_names: List[str] = field(default_factory=lambda: [
        "name_exact_match",
        "name_compact_match",
        "name_levenshtein_sim",
        "name_jaro_winkler_sim",
        "name_token_jaccard",
        "name_token_overlap_count",
        "name_token_overlap_ratio",
        "name_char_3gram_sim",
        "name_len_diff",
        "name_rel_len_diff",
        "name_is_missing",
        "addr_exact_match",
        "addr_levenshtein_sim",
        "addr_jaro_winkler_sim",
        "addr_token_jaccard",
        "addr_token_overlap_count",
        "addr_token_overlap_ratio",
        "addr_numeric_overlap_count",
        "addr_numeric_jaccard",
        "addr_len_diff",
        "addr_is_missing",
        "country_match",
        "country_is_missing",
        "is_source_2",
        "is_source_3",
        "name_addr_sim_product",
        "name_addr_sim_max",
        "name_addr_sim_weighted",
        "strong_both_agreement"
    ])

def get_config() -> Config:
    return Config()
