"""
ER-X: Production-Grade High-Recall Entity Resolution System.
"""

from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, ProvenanceMask, CandidatePair
from src.erx.normalization import ERXNormalizer, MultiViewRecord
from src.erx.learned_rules import LearnedRuleEngine
from src.erx.retrieval import ERXRetrievalEngine
from src.erx.features import ERXFeatureExtractor
from src.erx.model import ERXModelTrainer, ERXCalibrator
from src.erx.pipeline import ERXPipeline

__version__ = "1.0.0"

__all__ = [
    "ERXConfig",
    "InternalIDMapper",
    "ProvenanceMask",
    "CandidatePair",
    "ERXNormalizer",
    "MultiViewRecord",
    "LearnedRuleEngine",
    "ERXRetrievalEngine",
    "ERXFeatureExtractor",
    "ERXModelTrainer",
    "ERXCalibrator",
    "ERXPipeline",
]
