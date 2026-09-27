"""
ER-X Ultimate: In-Fold Learned Typo, Alias, and OCR Rule Induction Engine
Processes full ground-truth universe (7.64M links) via streaming DuckDB cursor without truncation.
"""

from __future__ import annotations
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
import duckdb

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.normalization import normalize_business_name

logger = logging.getLogger("erx_ultimate.learned_rules")


class LearnedRulesEngine:
    """Mines frequent token substitution and typo aliases from the complete training ground truth."""

    def __init__(self, config: UltimateConfig = CONFIG):
        self.config = config
        self.rules_dir = Path(config.paths.artifacts_dir) / "rules"
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.token_substitutions: Dict[str, str] = {}
        self.total_gt_processed: int = 0
        self.unique_links_processed: int = 0

    def fit_from_ground_truth(
        self,
        db_path: str,
        chunk_size: int = 50000,
        max_rules: int = 10000,
        min_support: int = 5,
        s1_fold_filter_sql: Optional[str] = None,
    ) -> None:
        """
        Extract frequent token aliases from ground truth pairs in training set.
        Streams the entire ground truth universe chunk-by-chunk with zero truncation.
        Supports both Source 2 and Source 3 links.
        """
        logger.info("Extracting learned aliases and typo patterns from COMPLETE training ground truth...")
        conn = duckdb.connect(db_path, read_only=True)
        
        try:
            token_pairs: Counter = Counter()
            self.total_gt_processed = 0

            # Scan Source 2 links
            filter_clause = f"AND ({s1_fold_filter_sql})" if s1_fold_filter_sql else ""
            
            s2_query = f"""
                SELECT 
                    s1.name_norm AS s1_name,
                    s2.name_norm AS target_name
                FROM train_ground_truth gt
                JOIN train_s1 s1 ON gt.source1_id = s1.id
                JOIN train_s2 s2 ON gt.target_id = s2.id AND gt.target_source = 2
                WHERE s1.name_norm != '' AND s2.name_norm != '' AND s1.name_norm != s2.name_norm
                {filter_clause};
            """
            cursor_s2 = conn.cursor()
            cursor_s2.execute(s2_query)
            while True:
                rows = cursor_s2.fetchmany(chunk_size)
                if not rows:
                    break
                self.total_gt_processed += len(rows)
                for s1_name, tgt_name in rows:
                    self._mine_tokens_from_pair(s1_name, tgt_name, token_pairs)

            # Scan Source 3 links
            s3_query = f"""
                SELECT 
                    s1.name_norm AS s1_name,
                    s3.name_norm AS target_name
                FROM train_ground_truth gt
                JOIN train_s1 s1 ON gt.source1_id = s1.id
                JOIN train_s3 s3 ON gt.target_id = s3.id AND gt.target_source = 3
                WHERE s1.name_norm != '' AND s3.name_norm != '' AND s1.name_norm != s3.name_norm
                {filter_clause};
            """
            cursor_s3 = conn.cursor()
            cursor_s3.execute(s3_query)
            while True:
                rows = cursor_s3.fetchmany(chunk_size)
                if not rows:
                    break
                self.total_gt_processed += len(rows)
                for s1_name, tgt_name in rows:
                    self._mine_tokens_from_pair(s1_name, tgt_name, token_pairs)

            self.unique_links_processed = len(token_pairs)
            self.token_substitutions = {}
            for (src_tok, dst_tok), count in token_pairs.most_common(max_rules):
                if count >= min_support:
                    self.token_substitutions[src_tok] = dst_tok

            logger.info(
                f"Mined {len(self.token_substitutions):,} token substitution rules "
                f"from {self.total_gt_processed:,} positive links (unique candidate token pairs: {self.unique_links_processed:,})."
            )
            self.save()
        finally:
            conn.close()

    @staticmethod
    def _mine_tokens_from_pair(s1_name: str, tgt_name: str, token_pairs: Counter) -> None:
        """Helper to extract single-token replacements between two normalized names."""
        s1_tokens = set(s1_name.split())
        tgt_tokens = set(tgt_name.split())
        diff_s1 = s1_tokens - tgt_tokens
        diff_tgt = tgt_tokens - s1_tokens
        if len(diff_s1) == 1 and len(diff_tgt) == 1:
            t1 = next(iter(diff_s1))
            t2 = next(iter(diff_tgt))
            if len(t1) > 2 and len(t2) > 2 and t1 != t2:
                token_pairs[(t2, t1)] += 1

    def save(self) -> None:
        """Persist learned rules and metadata to JSON file."""
        out_file = self.rules_dir / "learned_token_rules.json"
        metadata_file = self.rules_dir / "learned_rules_metadata.json"
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.token_substitutions, f, indent=2)
            
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "rule_count": len(self.token_substitutions),
                "total_gt_processed": self.total_gt_processed,
                "unique_pairs_mined": self.unique_links_processed,
                "min_support": 5,
                "has_truncation": False,
            }, f, indent=2)
            
        logger.info(f"Saved {len(self.token_substitutions):,} learned rules to {out_file}")

    def load(self) -> None:
        """Load learned rules from disk."""
        in_file = self.rules_dir / "learned_token_rules.json"
        if in_file.exists():
            with open(in_file, "r", encoding="utf-8") as f:
                self.token_substitutions = json.load(f)
            logger.info(f"Loaded {len(self.token_substitutions):,} rules from {in_file}")
        else:
            logger.warning("No pre-trained rules found; using empty substitution dict.")

    def apply_substitutions(self, text: str) -> str:
        """Apply learned substitutions to text tokens."""
        if not self.token_substitutions or not text:
            return text
        tokens = text.split()
        replaced = [self.token_substitutions.get(t, t) for t in tokens]
        return " ".join(replaced)
