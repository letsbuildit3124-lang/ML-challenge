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

    def query(self, term: str) -> np.ndarray:
        """Retrieve matching document IDs for a single term."""
        term_idx = self.vocabulary.get(term)
        if term_idx is None or self.indptr is None or self.indices is None:
            return np.empty(0, dtype=np.uint32)
        start = self.indptr[term_idx]
        end = self.indptr[term_idx + 1]
        return self.indices[start:end]


class ERXRetrievalEngine:
    """Multi-channel candidate retriever utilizing Reciprocal Rank Fusion."""

    def __init__(self, config: UltimateConfig = CONFIG):
        self.config = config
        self.indices_dir = Path(config.paths.cache_dir) / "indices"
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
            ng_hits: Counter = Counter()
            for ng in extract_ngrams(record.name_norm, 3):
                for doc_idx in self.channels["ngram"].query(ng):
                    ng_hits[doc_idx] += 1
            for r, (doc_idx, _) in enumerate(ng_hits.most_common(100)):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 2

        # Channel 3: Token
        if "token" in self.channels and record.name_norm:
            tok_hits: Counter = Counter()
            for tok in record.name_norm.split():
                for doc_idx in self.channels["token"].query(tok):
                    tok_hits[doc_idx] += 1
            for r, (doc_idx, _) in enumerate(tok_hits.most_common(100)):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 4

        # Channel 4: Phonetic
        if "phonetic" in self.channels and record.name_norm:
            ph_hits: Counter = Counter()
            for tok in record.name_norm.split():
                ph = compute_soundex(tok)
                if ph:
                    for doc_idx in self.channels["phonetic"].query(ph):
                        ph_hits[doc_idx] += 1
            for r, (doc_idx, _) in enumerate(ph_hits.most_common(50)):
                channel_scores[doc_idx] += 1.0 / (rrf_k + r + 1)
                channel_presence[doc_idx] |= 8

        # Channel 5: Address / Numeric
        if "address" in self.channels:
            addr_hits: Counter = Counter()
            if record.address_norm:
                for tok in record.address_norm.split():
                    if tok.isdigit() and len(tok) >= 2:
                        for doc_idx in self.channels["address"].query(f"num:{tok}"):
                            addr_hits[doc_idx] += 1
            if record.city_norm:
                for doc_idx in self.channels["address"].query(f"city:{record.city_norm}"):
                    addr_hits[doc_idx] += 1
            for r, (doc_idx, _) in enumerate(addr_hits.most_common(50)):
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
