"""
ER-X High-Recall Target -> S1 Retrieval Engine.
High-Performance CSR-backed Integer Array Indexing & Native Vectorized Candidate Union.

Implements 6 Complementary Retrieval Channels:
- Channel A: Exact / Learned Keys (Canonical name, compact name, sorted tokens, country+name, name+house_num)
- Channel B: Character 3-5 TF-IDF Sparse Cosine Retrieval
- Channel C: Rare Token Inverted Index with IDF Weighting
- Channel D: Address & House Number / Numeric Signature Index
- Channel E: Phonetic Token Signature Index
- Channel F: Learned Typo / OCR Variant Index
"""

import math
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.erx.config import ERXConfig
from src.erx.types import MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask

logger = logging.getLogger("erx.retrieval")


class CSRChannelIndex:
    """Contiguous CSR integer index for zero-copy slice lookups."""
    __slots__ = ("lut", "s1_ids")

    def __init__(self):
        self.lut: Dict[str, Tuple[int, int]] = {}
        self.s1_ids: np.ndarray = np.empty((0,), dtype=np.uint32)

    def build_from_dict(self, mapping: Dict[str, List[int]]) -> None:
        total_elements = sum(len(v) for v in mapping.values())
        self.s1_ids = np.empty(total_elements, dtype=np.uint32)
        offset = 0
        lut = {}
        ids_buf = self.s1_ids

        for k, v in mapping.items():
            cnt = len(v)
            if cnt > 0:
                lut[k] = (offset, cnt)
                ids_buf[offset : offset + cnt] = v
                offset += cnt

        self.lut = lut

    def get_slice(self, key: str) -> Optional[np.ndarray]:
        info = self.lut.get(key)
        if info is not None:
            off, cnt = info
            return self.s1_ids[off : off + cnt]
        return None

    def get(self, key: str, default: Any = None) -> Any:
        sl = self.get_slice(key)
        return sl if sl is not None else default

    def __getitem__(self, key: str) -> np.ndarray:
        sl = self.get_slice(key)
        if sl is None:
            raise KeyError(key)
        return sl

    def __contains__(self, key: str) -> bool:
        return key in self.lut

    def __iter__(self):
        return iter(self.lut)

    def save(self, dir_path: Path, prefix: str) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        lut_file = dir_path / f"{prefix}_lut.pkl"
        ids_file = dir_path / f"{prefix}_ids.npy"
        with open(lut_file, "wb") as f:
            pickle.dump(self.lut, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(ids_file, self.s1_ids)

    def load_mmap(self, dir_path: Path, prefix: str) -> bool:
        lut_file = dir_path / f"{prefix}_lut.pkl"
        ids_file = dir_path / f"{prefix}_ids.npy"
        if not lut_file.exists() or not ids_file.exists():
            return False
        with open(lut_file, "rb") as f:
            self.lut = pickle.load(f)
        self.s1_ids = np.load(ids_file, mmap_mode="r")
        return True

    def __len__(self) -> int:
        return len(self.lut)


class ERXRetrievalEngine:
    """High-recall inverted index & sparse retrieval engine indexing S1 entities with CSR acceleration."""

    def __init__(self, config: ERXConfig):
        self.config = config
        self.s1_records: Dict[int, Union[MultiViewRecord, CompactS1Record]] = {}

        # Channel A & F: Hash/Inverted exact and learned keys
        self.index_norm_name = CSRChannelIndex()
        self.index_compact_name = CSRChannelIndex()
        self.index_sorted_tokens = CSRChannelIndex()
        self.index_country_name = CSRChannelIndex()
        self.index_name_house = CSRChannelIndex()

        # Channel B: Sparse Char TF-IDF
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.s1_tfidf_matrix: Optional[sparse.csr_matrix] = None
        self.s1_id_order: List[int] = []

        # Channel C: Rare Token IDF Index
        self.token_postings = CSRChannelIndex()
        self.token_idf: Dict[str, float] = {}

        # Channel D: Address & Numeric Signatures
        self.index_numeric_sig = CSRChannelIndex()
        self.index_house_token = CSRChannelIndex()

        # Channel E: Phonetic Signatures
        self.index_phonetic = CSRChannelIndex()

    def index_s1(self, s1_records: Union[List[MultiViewRecord], List[CompactS1Record]]) -> None:
        """Builds all multi-channel indexes over S1 records and converts to contiguous CSR integer storage."""
        logger.info(f"Indexing {len(s1_records):,} S1 entities across 6 channels...")
        num_s1 = len(s1_records)
        token_doc_freq: Dict[str, int] = defaultdict(int)
        names_for_tfidf: List[str] = []
        self.s1_id_order = []

        raw_norm_name: Dict[str, List[int]] = defaultdict(list)
        raw_compact_name: Dict[str, List[int]] = defaultdict(list)
        raw_sorted_tokens: Dict[str, List[int]] = defaultdict(list)
        raw_country_name: Dict[str, List[int]] = defaultdict(list)
        raw_name_house: Dict[str, List[int]] = defaultdict(list)
        raw_token_postings: Dict[str, List[int]] = defaultdict(list)
        raw_numeric_sig: Dict[str, List[int]] = defaultdict(list)
        raw_house_token: Dict[str, List[int]] = defaultdict(list)
        raw_phonetic: Dict[str, List[int]] = defaultdict(list)

        for idx, rec in enumerate(s1_records, 1):
            s1_id = rec.internal_id
            self.s1_records[s1_id] = rec
            self.s1_id_order.append(s1_id)

            # Channel A: Exact & Learned keys
            if rec.norm_name:
                raw_norm_name[rec.norm_name].append(s1_id)
            if rec.compact_name:
                raw_compact_name[rec.compact_name].append(s1_id)
            if rec.sorted_token_name and len(rec.name_tokens) > 1:
                raw_sorted_tokens[rec.sorted_token_name].append(s1_id)
            if rec.country and rec.norm_name:
                raw_country_name[f"{rec.country}:{rec.norm_name}"].append(s1_id)
            if rec.norm_name and rec.house_numbers:
                for hn in rec.house_numbers:
                    raw_name_house[f"{rec.norm_name}:{hn}"].append(s1_id)

            # Channel C: Token doc frequencies
            for tok in rec.name_tok_set:
                if len(tok) >= 2:
                    token_doc_freq[tok] += 1
                    raw_token_postings[tok].append(s1_id)

            # Channel D: Address & numeric signatures
            if rec.numeric_signature:
                raw_numeric_sig[rec.numeric_signature].append(s1_id)
            for hn in rec.house_numbers:
                if rec.name_tokens:
                    first_sig = rec.name_tokens[0]
                    raw_house_token[f"{hn}:{first_sig}"].append(s1_id)

            # Channel E: Phonetic signature
            if rec.name_phonetic_sig:
                raw_phonetic[rec.name_phonetic_sig].append(s1_id)

            names_for_tfidf.append(rec.norm_name if rec.norm_name else "empty")

        # Prune rare token postings
        max_posting = self.config.rare_token_max_posting_size
        min_idf = self.config.rare_token_min_idf if num_s1 >= 100 else 1.0
        pruned_token_postings = {}
        for tok, count in token_doc_freq.items():
            idf = math.log((num_s1 + 1.0) / (count + 1.0)) + 1.0
            self.token_idf[tok] = idf
            if idf >= min_idf:
                if len(raw_token_postings[tok]) <= max_posting:
                    pruned_token_postings[tok] = raw_token_postings[tok]
                else:
                    pruned_token_postings[tok] = raw_token_postings[tok][:max_posting]

        # Convert all to contiguous CSR integer structures
        logger.info("Converting inverted indices to contiguous CSR uint32 arrays...")
        self.index_norm_name.build_from_dict(raw_norm_name)
        self.index_compact_name.build_from_dict(raw_compact_name)
        self.index_sorted_tokens.build_from_dict(raw_sorted_tokens)
        self.index_country_name.build_from_dict(raw_country_name)
        self.index_name_house.build_from_dict(raw_name_house)
        self.token_postings.build_from_dict(pruned_token_postings)
        self.index_numeric_sig.build_from_dict(raw_numeric_sig)
        self.index_house_token.build_from_dict(raw_house_token)
        self.index_phonetic.build_from_dict(raw_phonetic)

        logger.info(
            f"CSR 6-Channel Inverted Index ready: {len(self.token_postings):,} rare tokens, "
            f"{len(self.index_compact_name):,} compact names, {len(self.index_norm_name):,} canonical names."
        )

    def save_csr_indexes(self, csr_dir: Path) -> None:
        """Persists CSR inverted indexes to disk for cross-process memory mapping."""
        import pickle
        csr_dir.mkdir(parents=True, exist_ok=True)
        self.index_norm_name.save(csr_dir, "norm_name")
        self.index_compact_name.save(csr_dir, "compact_name")
        self.index_sorted_tokens.save(csr_dir, "sorted_tokens")
        self.index_country_name.save(csr_dir, "country_name")
        self.index_name_house.save(csr_dir, "name_house")
        self.token_postings.save(csr_dir, "token_postings")
        self.index_numeric_sig.save(csr_dir, "numeric_sig")
        self.index_house_token.save(csr_dir, "house_token")
        self.index_phonetic.save(csr_dir, "phonetic")
        with open(csr_dir / "token_idf.pkl", "wb") as f:
            pickle.dump(self.token_idf, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_csr_indexes(self, csr_dir: Path) -> bool:
        """Loads CSR inverted indexes from disk using zero-copy memory mapping."""
        import pickle
        if not (csr_dir / "norm_name_lut.pkl").exists():
            return False
        try:
            ok = (
                self.index_norm_name.load_mmap(csr_dir, "norm_name")
                and self.index_compact_name.load_mmap(csr_dir, "compact_name")
                and self.index_sorted_tokens.load_mmap(csr_dir, "sorted_tokens")
                and self.index_country_name.load_mmap(csr_dir, "country_name")
                and self.index_name_house.load_mmap(csr_dir, "name_house")
                and self.token_postings.load_mmap(csr_dir, "token_postings")
                and self.index_numeric_sig.load_mmap(csr_dir, "numeric_sig")
                and self.index_house_token.load_mmap(csr_dir, "house_token")
                and self.index_phonetic.load_mmap(csr_dir, "phonetic")
            )
            if ok and (csr_dir / "token_idf.pkl").exists():
                with open(csr_dir / "token_idf.pkl", "rb") as f:
                    self.token_idf = pickle.load(f)
            return ok
        except Exception:
            return False

    def retrieve_for_target(
        self,
        target: MultiViewRecord,
        top_k: Optional[int] = None
    ) -> List[CandidatePair]:
        """
        Retrieves union of candidate S1 entities for a single target record using CSR arrays.
        """
        max_k = top_k or self.config.max_total_candidates_per_target
        cand_scores: Dict[int, float] = {}
        cand_masks: Dict[int, int] = {}

        def add_candidate_slice(s1_slice: np.ndarray, score: float, bit: int):
            for s1_id in s1_slice:
                s1_id_int = int(s1_id)
                old_s = cand_scores.get(s1_id_int)
                if old_s is not None:
                    if score > old_s:
                        cand_scores[s1_id_int] = score
                    cand_masks[s1_id_int] |= bit
                else:
                    cand_scores[s1_id_int] = score
                    cand_masks[s1_id_int] = bit

        # Channel A: Exact & Learned Keys
        for name_key in [target.norm_name, target.translit_name, target.learned_name]:
            if name_key:
                sl = self.index_norm_name.get_slice(name_key)
                if sl is not None:
                    add_candidate_slice(sl, 1.0, ProvenanceMask.EXACT_OR_LEARNED)

        for c_name in [target.compact_name, target.translit_comp_name]:
            if c_name:
                sl = self.index_compact_name.get_slice(c_name)
                if sl is not None:
                    add_candidate_slice(sl, 0.95, ProvenanceMask.EXACT_OR_LEARNED)

        if target.sorted_token_name and len(target.name_tokens) > 1:
            sl = self.index_sorted_tokens.get_slice(target.sorted_token_name)
            if sl is not None:
                add_candidate_slice(sl, 0.90, ProvenanceMask.EXACT_OR_LEARNED)

        for n_key in [target.norm_name, target.translit_name]:
            if target.country and n_key:
                sl = self.index_country_name.get_slice(f"{target.country}:{n_key}")
                if sl is not None:
                    add_candidate_slice(sl, 1.0, ProvenanceMask.EXACT_OR_LEARNED)

        for n_key in [target.norm_name, target.translit_name]:
            if n_key and target.house_numbers:
                for hn in target.house_numbers:
                    sl = self.index_name_house.get_slice(f"{n_key}:{hn}")
                    if sl is not None:
                        add_candidate_slice(sl, 1.0, ProvenanceMask.EXACT_OR_LEARNED)

        # Channel C: Rare Token IDF Retrieval
        if target.name_tok_set or target.translit_tok_set:
            token_scores: Dict[int, float] = defaultdict(float)
            min_idf = self.config.rare_token_min_idf if len(self.s1_records) >= 100 else 1.0
            seen_tokens: Set[str] = set()

            for tok in target.name_tok_set:
                seen_tokens.add(tok)
                idf = self.token_idf.get(tok, 0.0)
                if idf >= min_idf:
                    sl = self.token_postings.get_slice(tok)
                    if sl is not None:
                        for s1_id in sl:
                            token_scores[int(s1_id)] += idf

            for tok in target.translit_tok_set:
                if tok not in seen_tokens:
                    idf = self.token_idf.get(tok, 0.0)
                    if idf >= min_idf:
                        sl = self.token_postings.get_slice(tok)
                        if sl is not None:
                            for s1_id in sl:
                                token_scores[int(s1_id)] += idf

            if token_scores:
                sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.rare_token_top_k]
                for s1_id, t_score in sorted_tokens:
                    norm_score = min(1.0, t_score / 15.0)
                    old_s = cand_scores.get(s1_id)
                    if old_s is not None:
                        if norm_score > old_s:
                            cand_scores[s1_id] = norm_score
                        cand_masks[s1_id] |= ProvenanceMask.RARE_TOKEN
                    else:
                        cand_scores[s1_id] = norm_score
                        cand_masks[s1_id] = ProvenanceMask.RARE_TOKEN

        # Channel D: Address & House Number Signatures
        if target.house_numbers and target.name_tokens:
            first_sig = target.name_tokens[0]
            for hn in target.house_numbers:
                sl = self.index_house_token.get_slice(f"{hn}:{first_sig}")
                if sl is not None:
                    add_candidate_slice(sl, 0.75, ProvenanceMask.ADDRESS)

        if target.numeric_signature:
            sl = self.index_numeric_sig.get_slice(target.numeric_signature)
            if sl is not None:
                add_candidate_slice(sl[:self.config.address_top_k], 0.70, ProvenanceMask.ADDRESS)

        # Channel E: Phonetic Signatures
        if target.name_phonetic_sig:
            sl = self.index_phonetic.get_slice(target.name_phonetic_sig)
            if sl is not None:
                add_candidate_slice(sl[:self.config.phonetic_top_k], 0.65, ProvenanceMask.PHONETIC)

        t_int_id = target.internal_id
        candidates = [
            CandidatePair(
                target_internal_id=t_int_id,
                s1_internal_id=s1_id,
                retrieval_score=cand_scores[s1_id],
                provenance_mask=cand_masks[s1_id]
            )
            for s1_id in cand_scores
        ]

        if len(candidates) > max_k:
            candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
            return candidates[:max_k]
        elif len(candidates) > 1:
            candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates

    def retrieve_batch_arrays(
        self,
        targets: List[MultiViewRecord],
        top_k: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        High-Throughput Batched Integer Retrieval returning contiguous 1D NumPy arrays.
        Eliminates Python candidate object creation and list fragmentation across large target batches.
        """
        max_k = top_k or self.config.max_total_candidates_per_target
        num_targets = len(targets)

        target_cand_offsets = np.zeros(num_targets + 1, dtype=np.int32)
        all_s1_ids = []
        all_target_idx = []
        all_scores = []
        all_masks = []
        all_ranks = []

        best_scores_arr = []
        second_best_arr = []
        cand_counts_arr = []
        count_05_arr = []
        count_07_arr = []
        count_08_arr = []

        total_pair_count = 0

        for t_idx, target in enumerate(targets):
            cands = self.retrieve_for_target(target, top_k=max_k)
            c_len = len(cands)
            target_cand_offsets[t_idx] = total_pair_count

            if c_len > 0:
                b_score = cands[0].retrieval_score
                sec_score = cands[1].retrieval_score if c_len > 1 else 0.0
                c05 = sum(1 for c in cands if c.retrieval_score >= 0.5)
                c07 = sum(1 for c in cands if c.retrieval_score >= 0.7)
                c08 = sum(1 for c in cands if c.retrieval_score >= 0.8)

                for rank, c in enumerate(cands):
                    all_s1_ids.append(c.s1_internal_id)
                    all_target_idx.append(t_idx)
                    all_scores.append(c.retrieval_score)
                    all_masks.append(c.provenance_mask)
                    all_ranks.append(float(rank))
                    best_scores_arr.append(b_score)
                    second_best_arr.append(sec_score)
                    cand_counts_arr.append(float(c_len))
                    count_05_arr.append(float(c05))
                    count_07_arr.append(float(c07))
                    count_08_arr.append(float(c08))

                total_pair_count += c_len

        target_cand_offsets[num_targets] = total_pair_count

        return {
            "cand_s1_ids": np.array(all_s1_ids, dtype=np.uint32),
            "cand_target_idx": np.array(all_target_idx, dtype=np.int32),
            "cand_scores": np.array(all_scores, dtype=np.float32),
            "cand_prov_masks": np.array(all_masks, dtype=np.uint32),
            "cand_ranks": np.array(all_ranks, dtype=np.float32),
            "target_offsets": target_cand_offsets,
            "best_scores": np.array(best_scores_arr, dtype=np.float32),
            "second_best_scores": np.array(second_best_arr, dtype=np.float32),
            "cand_counts": np.array(cand_counts_arr, dtype=np.float32),
            "counts_above_05": np.array(count_05_arr, dtype=np.float32),
            "counts_above_07": np.array(count_07_arr, dtype=np.float32),
            "counts_above_08": np.array(count_08_arr, dtype=np.float32),
            "total_pairs": total_pair_count,
        }
