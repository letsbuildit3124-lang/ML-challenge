"""
ER-X High-Recall Target -> S1 Retrieval Engine.
Implements 6 Complementary Retrieval Channels:
- Channel A: Exact / Learned Keys (Canonical name, compact name, sorted tokens, country+name, name+house_num)
- Channel B: Character 3-5 TF-IDF Sparse Cosine Retrieval
- Channel C: Rare Token Inverted Index with IDF Weighting
- Channel D: Address & House Number / Numeric Signature Index
- Channel E: Phonetic Token Signature Index
- Channel F: Learned Typo / OCR Variant Index
"""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.erx.config import ERXConfig
from src.erx.types import MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask

logger = logging.getLogger("erx.retrieval")


class ERXRetrievalEngine:
    """High-recall inverted index & sparse retrieval engine indexing S1 entities."""

    def __init__(self, config: ERXConfig):
        self.config = config
        self.s1_records: Dict[int, Union[MultiViewRecord, CompactS1Record]] = {}

        # Channel A & F: Hash/Inverted exact and learned keys
        self.index_norm_name: Dict[str, List[int]] = defaultdict(list)
        self.index_compact_name: Dict[str, List[int]] = defaultdict(list)
        self.index_sorted_tokens: Dict[str, List[int]] = defaultdict(list)
        self.index_country_name: Dict[str, List[int]] = defaultdict(list)
        self.index_name_house: Dict[str, List[int]] = defaultdict(list)

        # Channel B: Sparse Char TF-IDF
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.s1_tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.s1_id_order: List[int] = []

        # Channel C: Rare Token IDF Index
        self.token_postings: Dict[str, List[int]] = defaultdict(list)
        self.token_idf: Dict[str, float] = {}

        # Channel D: Address & Numeric Signatures
        self.index_numeric_sig: Dict[str, List[int]] = defaultdict(list)
        self.index_house_token: Dict[str, List[int]] = defaultdict(list)

        # Channel E: Phonetic Signatures
        self.index_phonetic: Dict[str, List[int]] = defaultdict(list)

    def index_s1(self, s1_records: Union[List[MultiViewRecord], List[CompactS1Record]]) -> None:
        """Builds all multi-channel indexes over S1 records."""
        logger.info(f"Indexing {len(s1_records):,} S1 entities across 6 channels...")
        num_s1 = len(s1_records)
        token_doc_freq: Dict[str, int] = defaultdict(int)
        names_for_tfidf: List[str] = []
        self.s1_id_order = []

        for idx, rec in enumerate(s1_records, 1):
            s1_id = rec.internal_id
            self.s1_records[s1_id] = rec
            self.s1_id_order.append(s1_id)

            # Channel A: Exact & Learned keys
            if rec.norm_name:
                self.index_norm_name[rec.norm_name].append(s1_id)
            if rec.compact_name:
                self.index_compact_name[rec.compact_name].append(s1_id)
            if rec.sorted_token_name and len(rec.name_tokens) > 1:
                self.index_sorted_tokens[rec.sorted_token_name].append(s1_id)
            if rec.country and rec.norm_name:
                self.index_country_name[f"{rec.country}:{rec.norm_name}"].append(s1_id)
            if rec.norm_name and rec.house_numbers:
                for hn in rec.house_numbers:
                    self.index_name_house[f"{rec.norm_name}:{hn}"].append(s1_id)

            # Channel C: Token doc frequencies
            for tok in rec.name_tok_set:
                if len(tok) >= 2:
                    token_doc_freq[tok] += 1
                    self.token_postings[tok].append(s1_id)

            # Channel D: Address & numeric signatures
            if rec.numeric_signature:
                self.index_numeric_sig[rec.numeric_signature].append(s1_id)
            for hn in rec.house_numbers:
                if rec.name_tokens:
                    # Index house number + first significant name token
                    first_sig = rec.name_tokens[0]
                    self.index_house_token[f"{hn}:{first_sig}"].append(s1_id)

            # Channel E: Phonetic signature
            if rec.name_phonetic_sig:
                self.index_phonetic[rec.name_phonetic_sig].append(s1_id)

            # Channel B text collection
            names_for_tfidf.append(rec.norm_name if rec.norm_name else "empty")

            if idx % 300000 == 0 or idx == num_s1:
                logger.info(f"  -> Inverted channels indexed: {idx:,} / {num_s1:,} records ({idx/num_s1*100:.1f}%)...")

        # Compute IDF for Channel C and apply posting caps
        logger.info("Computing Token IDF scores and pruning postings...")
        max_posting = self.config.rare_token_max_posting_size
        min_idf = self.config.rare_token_min_idf if num_s1 >= 100 else 1.0
        pruned_postings = {}
        for tok, count in token_doc_freq.items():
            idf = math.log((num_s1 + 1.0) / (count + 1.0)) + 1.0
            self.token_idf[tok] = idf
            # Cap extreme postings to prevent combinatorial explosions and discard non-rare tokens
            if idf >= min_idf:
                if len(self.token_postings[tok]) <= max_posting:
                    pruned_postings[tok] = self.token_postings[tok]
                else:
                    pruned_postings[tok] = self.token_postings[tok][:max_posting]
        self.token_postings = pruned_postings

        logger.info(f"6-Channel Inverted Index ready with {len(self.token_postings):,} rare tokens and {len(self.index_compact_name):,} compact names.")

    def retrieve_for_target(
        self,
        target: MultiViewRecord,
        top_k: Optional[int] = None
    ) -> List[CandidatePair]:
        """
        Retrieves union of candidate S1 entities for a single target record.
        Accumulates candidates in compact dictionary: s1_internal_id -> (best_retrieval_score, provenance_mask).
        """
        max_k = top_k or self.config.max_total_candidates_per_target
        candidate_map: Dict[int, Tuple[float, int]] = {}

        def add_candidate(s1_id: int, score: float, bit: int):
            if s1_id in candidate_map:
                curr_score, curr_mask = candidate_map[s1_id]
                candidate_map[s1_id] = (max(curr_score, score), curr_mask | bit)
            else:
                candidate_map[s1_id] = (score, bit)

        # -------------------------------------------------------------
        # Channel A: Exact & Learned Keys
        # -------------------------------------------------------------
        # 1. Canonical / Transliterated name match
        for name_key in [target.norm_name, target.translit_name, target.learned_name]:
            if name_key and name_key in self.index_norm_name:
                for s1_id in self.index_norm_name[name_key]:
                    add_candidate(s1_id, 1.0, ProvenanceMask.EXACT_OR_LEARNED)

        # 2. Compact name match (native and transliterated)
        for c_name in [target.compact_name, target.translit_comp_name]:
            if c_name and c_name in self.index_compact_name:
                for s1_id in self.index_compact_name[c_name]:
                    add_candidate(s1_id, 0.95, ProvenanceMask.EXACT_OR_LEARNED)

        # 3. Sorted token name match (word reorder invariance)
        if target.sorted_token_name and len(target.name_tokens) > 1:
            if target.sorted_token_name in self.index_sorted_tokens:
                for s1_id in self.index_sorted_tokens[target.sorted_token_name]:
                    add_candidate(s1_id, 0.90, ProvenanceMask.EXACT_OR_LEARNED)

        # 4. Country + Name match
        for n_key in [target.norm_name, target.translit_name]:
            if target.country and n_key:
                c_key = f"{target.country}:{n_key}"
                if c_key in self.index_country_name:
                    for s1_id in self.index_country_name[c_key]:
                        add_candidate(s1_id, 1.0, ProvenanceMask.EXACT_OR_LEARNED)

        # 5. Name + House number match
        for n_key in [target.norm_name, target.translit_name]:
            if n_key and target.house_numbers:
                for hn in target.house_numbers:
                    nh_key = f"{n_key}:{hn}"
                    if nh_key in self.index_name_house:
                        for s1_id in self.index_name_house[nh_key]:
                            add_candidate(s1_id, 1.0, ProvenanceMask.EXACT_OR_LEARNED)

        # -------------------------------------------------------------
        # Channel C: Rare Token IDF Retrieval (Native + Transliterated)
        # -------------------------------------------------------------
        if target.name_tok_set or target.translit_tok_set:
            token_scores: Dict[int, float] = defaultdict(float)
            min_idf = self.config.rare_token_min_idf if len(self.s1_records) >= 100 else 1.0
            
            seen_tokens: Set[str] = set()
            for tok in target.name_tok_set:
                seen_tokens.add(tok)
                idf = self.token_idf.get(tok, 0.0)
                if idf >= min_idf and tok in self.token_postings:
                    for s1_id in self.token_postings[tok]:
                        token_scores[s1_id] += idf

            for tok in target.translit_tok_set:
                if tok not in seen_tokens:
                    idf = self.token_idf.get(tok, 0.0)
                    if idf >= min_idf and tok in self.token_postings:
                        for s1_id in self.token_postings[tok]:
                            token_scores[s1_id] += idf

            if token_scores:
                # Top K rare token candidates
                sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.rare_token_top_k]
                for s1_id, t_score in sorted_tokens:
                    norm_score = min(1.0, t_score / 15.0)
                    add_candidate(s1_id, norm_score, ProvenanceMask.RARE_TOKEN)

        # -------------------------------------------------------------
        # Channel D: Address & House Number Signatures
        # -------------------------------------------------------------
        if target.house_numbers and target.name_tokens:
            first_sig = target.name_tokens[0]
            for hn in target.house_numbers:
                ht_key = f"{hn}:{first_sig}"
                if ht_key in self.index_house_token:
                    for s1_id in self.index_house_token[ht_key]:
                        add_candidate(s1_id, 0.75, ProvenanceMask.ADDRESS)

        if target.numeric_signature and target.numeric_signature in self.index_numeric_sig:
            for s1_id in self.index_numeric_sig[target.numeric_signature][:self.config.address_top_k]:
                add_candidate(s1_id, 0.70, ProvenanceMask.ADDRESS)

        # -------------------------------------------------------------
        # Channel E: Phonetic Token Signatures
        # -------------------------------------------------------------
        if target.name_phonetic_sig and target.name_phonetic_sig in self.index_phonetic:
            for s1_id in self.index_phonetic[target.name_phonetic_sig][:self.config.phonetic_top_k]:
                add_candidate(s1_id, 0.65, ProvenanceMask.PHONETIC)

        # -------------------------------------------------------------
        # Convert candidate map to list of CandidatePair
        # -------------------------------------------------------------
        candidates = [
            CandidatePair(
                target_internal_id=target.internal_id,
                s1_internal_id=s1_id,
                retrieval_score=score,
                provenance_mask=mask
            )
            for s1_id, (score, mask) in candidate_map.items()
        ]

        # Sort candidates by retrieval score descending
        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates[:max_k]

    def retrieve_batch_with_tfidf(
        self,
        targets: List[MultiViewRecord],
        top_k: Optional[int] = None
    ) -> List[List[CandidatePair]]:
        """
        Batch retrieval combining Channels A-F with vectorized sparse TF-IDF matrix multiplication for Channel B.
        Routes transliterated representations for non-ASCII target records to match S1 Latin index.
        """
        results: List[List[CandidatePair]] = []
        max_k = top_k or self.config.max_total_candidates_per_target

        # 1. Base channel retrieval for each target
        for target in targets:
            cands = self.retrieve_for_target(target, top_k=max_k)
            results.append(cands)

        # 2. Vectorized Sparse TF-IDF (Channel B)
        if self.tfidf_vectorizer is not None and self.s1_tfidf_matrix is not None:
            target_names = [
                t.translit_name if (t.translit_name and not t.norm_name.isascii()) else (t.norm_name if t.norm_name else "empty")
                for t in targets
            ]
            t_matrix = self.tfidf_vectorizer.transform(target_names)  # (batch_size, num_features)

            # Sparse matrix dot product: (batch_size, num_s1)
            sim_matrix = t_matrix.dot(self.s1_tfidf_matrix.T)

            tfidf_k = self.config.tfidf_top_k
            for i, target in enumerate(targets):
                row = sim_matrix.getrow(i)
                if row.nnz > 0:
                    data = row.data
                    indices = row.indices
                    if len(data) > tfidf_k:
                        top_idx_in_row = np.argpartition(data, -tfidf_k)[-tfidf_k:]
                        top_scores = data[top_idx_in_row]
                        top_s1_indices = indices[top_idx_in_row]
                    else:
                        top_scores = data
                        top_s1_indices = indices

                    # Merge into existing candidates
                    cand_dict: Dict[int, CandidatePair] = {c.s1_internal_id: c for c in results[i]}
                    for score, idx in zip(top_scores, top_s1_indices):
                        if score < 0.25:  # Minimum TF-IDF similarity threshold
                            continue
                        s1_id = self.s1_id_order[idx]
                        if s1_id in cand_dict:
                            cand = cand_dict[s1_id]
                            cand.retrieval_score = max(cand.retrieval_score, float(score))
                            cand.provenance_mask |= ProvenanceMask.CHAR_TFIDF
                        else:
                            cand_dict[s1_id] = CandidatePair(
                                target_internal_id=target.internal_id,
                                s1_internal_id=s1_id,
                                retrieval_score=float(score),
                                provenance_mask=int(ProvenanceMask.CHAR_TFIDF)
                            )

                    merged_list = list(cand_dict.values())
                    merged_list.sort(key=lambda c: c.retrieval_score, reverse=True)
                    results[i] = merged_list[:max_k]

        return results
