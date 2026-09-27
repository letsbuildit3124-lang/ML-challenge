"""
ER-X Ultimate: Multi-Channel CSR Inverted Index & Reciprocal Rank Fusion (RRF) Retrieval Engine
"""

from __future__ import annotations
import math
import os
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import pyarrow.parquet as pq

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.types import EntityRecord, CandidateMatch, SourceType
from src.erx_ultimate.normalization import extract_ngrams, compute_soundex

logger = logging.getLogger("erx_ultimate.retrieval")


class CSRChannelIndex:
    """Zero-copy memory-mapped Compressed Sparse Row inverted index."""

    def __init__(self, name: str):
        self.name = name
        self.vocabulary: Dict[str, int] = {}
        self.indptr: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def build_from_postings(self, postings: Dict[str, List[int]], weights: Optional[Dict[str, List[float]]] = None) -> None:
        """Construct CSR arrays from Python postings dict."""
        self.vocabulary = {term: idx for idx, term in enumerate(postings.keys())}
        num_terms = len(self.vocabulary)
        
        counts = [len(postings[term]) for term in self.vocabulary]
        self.indptr = np.zeros(num_terms + 1, dtype=np.uint32)
        self.indptr[1:] = np.cumsum(counts, dtype=np.uint32)
        
        total_entries = self.indptr[-1]
        self.indices = np.empty(total_entries, dtype=np.uint32)
        has_weights = weights is not None
        if has_weights:
            self.weights = np.empty(total_entries, dtype=np.float32)

        for term, term_idx in self.vocabulary.items():
            start = self.indptr[term_idx]
            end = self.indptr[term_idx + 1]
            self.indices[start:end] = postings[term]
            if has_weights and weights:
                self.weights[start:end] = weights[term]

    def save(self, directory: Path) -> None:
        """Save vocabulary and NumPy arrays to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        import json
        vocab_path = directory / f"{self.name}_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f)
            
        np.save(directory / f"{self.name}_indptr.npy", self.indptr)
        np.save(directory / f"{self.name}_indices.npy", self.indices)
        if self.weights is not None:
            np.save(directory / f"{self.name}_weights.npy", self.weights)
        logger.info(f"Saved CSR index [{self.name}] to {directory} ({len(self.vocabulary):,} terms).")

    def load(self, directory: Path, mmap_mode: Optional[str] = "r") -> None:
        """Load vocabulary and memory-map NumPy arrays."""
        import json
        vocab_path = directory / f"{self.name}_vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocabulary = json.load(f)
            
        self.indptr = np.load(directory / f"{self.name}_indptr.npy", mmap_mode=mmap_mode)
        self.indices = np.load(directory / f"{self.name}_indices.npy", mmap_mode=mmap_mode)
        weight_file = directory / f"{self.name}_weights.npy"
        if weight_file.exists():
            self.weights = np.load(weight_file, mmap_mode=mmap_mode)
        logger.info(f"Loaded CSR index [{self.name}] ({len(self.vocabulary):,} terms).")

    def query(self, term: str, max_postings: int = 5000) -> np.ndarray:
        """Retrieve matching document IDs for a single term."""
        term_idx = self.vocabulary.get(term)
        if term_idx is None or self.indptr is None or self.indices is None:
            return np.empty(0, dtype=np.uint32)
        start = self.indptr[term_idx]
        end = self.indptr[term_idx + 1]
        if end - start > max_postings:
            return self.indices[start:start + max_postings]
        return self.indices[start:end]


def _top_k_from_posting_lists(posting_lists: List[np.ndarray], top_n: int = 100) -> List[int]:
    """Fast C/NumPy aggregation of candidate document indices across posting lists."""
    if not posting_lists:
        return []
    valid = [p for p in posting_lists if len(p) > 0]
    if not valid:
        return []
    if len(valid) == 1:
        arr = valid[0]
        return arr[:top_n].tolist() if len(arr) > top_n else arr.tolist()
    
    flat = np.concatenate(valid)
    if len(flat) == 0:
        return []
    if len(flat) < 300:
        c = Counter(flat)
        return [doc_idx for doc_idx, _ in c.most_common(top_n)]
    
    vals, counts = np.unique(flat, return_counts=True)
    if len(vals) <= top_n:
        order = np.argsort(-counts)
        return vals[order].tolist()
    else:
        top_idx = np.argpartition(counts, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(-counts[top_idx])]
        return vals[top_idx].tolist()


class ERXRetrievalEngine:
    """Multi-channel candidate retriever utilizing Reciprocal Rank Fusion."""

    def __init__(self, config: UltimateConfig = CONFIG):
        self.config = config
        self.indices_dir = Path(config.paths.cache_dir) / "indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.channels: Dict[str, CSRChannelIndex] = {}
        self.s1_id_map: Optional[np.ndarray] = None  # Array mapping internal doc_idx -> s1_id

    def build_s1_indexes(self, s1_parquet_path: Path) -> None:
        """Scan S1 Parquet file and build all 6 CSR inverted index channels."""
        logger.info(f"Building S1 CSR Inverted Indexes from {s1_parquet_path}...")
        table = pq.read_table(s1_parquet_path)
        
        ids = table["id"].to_numpy()
        names = table["name_norm"].to_pylist()
        addrs = table["address_norm"].to_pylist()
        cities = table["city_norm"].to_pylist()
        phones = table["phone_norm"].to_pylist()
        webs = table["website_norm"].to_pylist()
        
        num_docs = len(ids)
        self.s1_id_map = ids.astype(np.int64)
        np.save(self.indices_dir / "s1_id_map.npy", self.s1_id_map)

        # 1. Exact & Key Identifiers
        exact_postings: Dict[str, List[int]] = defaultdict(list)
        # 2. Char 3-grams
        ngram_postings: Dict[str, List[int]] = defaultdict(list)
        # 3. Rare Tokens
        token_postings: Dict[str, List[int]] = defaultdict(list)
        # 4. Phonetic Tokens
        phonetic_postings: Dict[str, List[int]] = defaultdict(list)
        # 5. Address / Numeric
        address_postings: Dict[str, List[int]] = defaultdict(list)

        token_counts: Counter = Counter()

        for doc_idx in range(num_docs):
            name = names[doc_idx]
            addr = addrs[doc_idx]
            city = cities[doc_idx]
            phone = phones[doc_idx]
            web = webs[doc_idx]

            # Exact keys
            if name:
                exact_postings[f"name:{name}"].append(doc_idx)
            if phone:
                exact_postings[f"phone:{phone}"].append(doc_idx)
            if web:
                exact_postings[f"web:{web}"].append(doc_idx)

            # N-grams (3-grams)
            if name:
                for ng in extract_ngrams(name, 3):
                    ngram_postings[ng].append(doc_idx)

            # Tokens
            if name:
                for tok in name.split():
                    token_postings[tok].append(doc_idx)
                    token_counts[tok] += 1
                    # Phonetic
                    ph = compute_soundex(tok)
                    if ph:
                        phonetic_postings[ph].append(doc_idx)

            # Address numeric
            if addr:
                for tok in addr.split():
                    if tok.isdigit() and len(tok) >= 2:
                        address_postings[f"num:{tok}"].append(doc_idx)
            if city:
                address_postings[f"city:{city}"].append(doc_idx)

        # Build CSR indices
        logger.info("Constructing and saving CSR Channel structures...")
        idx_configs = [
            ("exact", exact_postings),
            ("ngram", ngram_postings),
            ("token", token_postings),
            ("phonetic", phonetic_postings),
            ("address", address_postings),
        ]

        for name, postings in idx_configs:
            csr = CSRChannelIndex(name)
            csr.build_from_postings(postings)
            csr.save(self.indices_dir)
            self.channels[name] = csr

        logger.info("S1 CSR Inverted Indexes successfully built.")

    def load_indexes(self, mmap_mode: str = "r") -> None:
        """Load all memory-mapped CSR inverted indices."""
        self.s1_id_map = np.load(self.indices_dir / "s1_id_map.npy", mmap_mode=mmap_mode)
        for name in ["exact", "ngram", "token", "phonetic", "address"]:
            csr = CSRChannelIndex(name)
            csr.load(self.indices_dir, mmap_mode=mmap_mode)
            self.channels[name] = csr
        logger.info("All CSR Inverted Indexes loaded into memory map.")

    def retrieve_candidates_for_record(
        self,
        record: EntityRecord,
        top_k: int = 30,
        rrf_k: int = 60,
    ) -> List[CandidateMatch]:
        """Retrieve top candidate S1 entities using Reciprocal Rank Fusion across all channels."""
        channel_scores: Dict[int, float] = defaultdict(float)
        channel_presence: Dict[int, int] = defaultdict(int)

        # Channel 1: Exact / Key
        if "exact" in self.channels:
            exact_hits: List[int] = []
            if record.name_norm:
                exact_hits.extend(self.channels["exact"].query(f"name:{record.name_norm}"))
            if record.phone_norm:
                exact_hits.extend(self.channels["exact"].query(f"phone:{record.phone_norm}"))
            if record.website_norm:
                exact_hits.extend(self.channels["exact"].query(f"web:{record.website_norm}"))
            for r, doc_idx in enumerate(exact_hits[:100]):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 1

        # Channel 2: N-gram
        if "ngram" in self.channels and record.name_norm:
            ngrams = extract_ngrams(record.name_norm, 3)
            posting_lists = [self.channels["ngram"].query(ng) for ng in ngrams]
            top_ng_docs = _top_k_from_posting_lists(posting_lists, top_n=100)
            for r, doc_idx in enumerate(top_ng_docs):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 2

        # Channel 3: Token
        if "token" in self.channels and record.name_norm:
            tokens = record.name_norm.split()
            posting_lists = [self.channels["token"].query(tok) for tok in tokens]
            top_tok_docs = _top_k_from_posting_lists(posting_lists, top_n=100)
            for r, doc_idx in enumerate(top_tok_docs):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 4

        # Channel 4: Phonetic
        if "phonetic" in self.channels and record.name_norm:
            tokens = record.name_norm.split()
            posting_lists = [self.channels["phonetic"].query(compute_soundex(tok)) for tok in tokens if tok]
            top_ph_docs = _top_k_from_posting_lists(posting_lists, top_n=50)
            for r, doc_idx in enumerate(top_ph_docs):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 8

        # Channel 5: Address / Numeric
        if "address" in self.channels:
            posting_lists = []
            if record.address_norm:
                for tok in record.address_norm.split():
                    if tok.isdigit() and len(tok) >= 2:
                        posting_lists.append(self.channels["address"].query(f"num:{tok}"))
            if record.city_norm:
                posting_lists.append(self.channels["address"].query(f"city:{record.city_norm}"))
            top_addr_docs = _top_k_from_posting_lists(posting_lists, top_n=50)
            for r, doc_idx in enumerate(top_addr_docs):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 16

        if not channel_scores:
            return []

        # Sort top-K by composite RRF score
        ranked_candidates = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        matches = []
        for doc_idx, score in ranked_candidates:
            s1_id = int(self.s1_id_map[doc_idx])
            matches.append(CandidateMatch(
                target_id=record.id,
                target_source=record.source,
                s1_id=s1_id,
                rrf_score=score,
                channel_mask=channel_presence[doc_idx]
            ))
        return matches
