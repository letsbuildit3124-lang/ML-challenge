"""
ER-X Ultimate: Centralized Configuration Management
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class SystemConfig:
    num_threads: int = 8
    max_memory_gb: float = 20.0
    soft_memory_gb: float = 12.0
    batch_size: int = 50000
    random_seed: int = 42


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    train_s1: str = "dataset/train/train_source1.tsv"
    train_s2: str = "dataset/train/train_source2.tsv"
    train_s3: str = "dataset/train/train_source3.tsv"
    train_ground_truth: str = "dataset/train/train_ground_truth.tsv"
    test_s1: str = "dataset/test/test_source1.tsv"
    test_s2: str = "dataset/test/test_source2.tsv"
    test_s3: str = "dataset/test/test_source3.tsv"
    duckdb_path: str = "cache/erx_ultimate/entity_resolution.duckdb"
    cache_dir: str = "cache/erx_ultimate"
    artifacts_dir: str = "artifacts/erx_ultimate"
    outputs_dir: str = "outputs/erx_ultimate"

    def resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.project_root / p


@dataclass
class RetrievalConfig:
    top_k_candidates: int = 30
    channels: List[str] = field(default_factory=lambda: [
        "exact_alias",
        "tfidf_char_ngram",
        "rare_token_idf",
        "phonetic_metaphone",
        "address_numeric",
        "learned_typo_ocr"
    ])
    rrf_k: int = 60
    min_candidate_score: float = 0.001


@dataclass
class ModelConfig:
    objective: str = "binary"
    metric: str = "binary_logloss"
    boosting_type: str = "gbdt"
    n_estimators: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 127
    max_depth: int = 8
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_bin: int = 255
    free_raw_data: bool = True
    n_jobs: int = 8


@dataclass
class DecisionConfig:
    f_beta: float = 0.5
    base_threshold: float = 0.78
    min_margin: float = 0.05
    enforce_target_exclusivity: bool = True
    allow_singletons: bool = True


@dataclass
class UltimateConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "UltimateConfig":
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls(
                system=SystemConfig(**data.get("system", {})),
                paths=PathConfig(**data.get("paths", {})),
                retrieval=RetrievalConfig(**data.get("retrieval", {})),
                model=ModelConfig(**data.get("model", {})),
                decision=DecisionConfig(**data.get("decision", {})),
            )
        return cls()


# Default singleton instance
CONFIG = UltimateConfig()
