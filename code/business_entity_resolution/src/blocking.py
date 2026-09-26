"""
Stage 3: Multi-Pass Blocking / Candidate Generation.
Maximizes candidate recall (target >= 0.99) while maintaining high reduction ratio.
Runs multiple complementary passes:
1. Exact sorted-core name + country hashing.
2. Character n-gram TF-IDF cosine similarity (NearestNeighbors).
3. Dense semantic embedding ANN search (FAISS IndexFlatIP).
"""

from typing import List, Tuple, Dict, Set, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .phonetic import phonetic_key
from .config import BLOCKING_TOP_K_TFIDF, BLOCKING_TOP_K_EMB, TFIDF_NGRAM_RANGE




def exact_key_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame, max_bucket_size: int = 30) -> List[Tuple[str, str]]:
    """
    Pass 1: Exact sorted-core-name + country hash bucketing.
    Memory-safe: vectorized array iteration with max_bucket_size cap to prevent combinatorial explosion.
    """
    pairs = []
    buckets = {}

    other_ids = other_df["entity_id"].astype(str).str.strip().values
    other_cores = other_df["sorted_core"].astype(str).str.strip().values
    other_countries = other_df["country"].astype(str).str.strip().values

    for cid, core, country in zip(other_ids, other_cores, other_countries):
        if len(core) >= 3:
            b = buckets.setdefault((core, country), [])
            if len(b) < max_bucket_size:
                b.append(cid)

    s1_ids = s1_df["entity_id"].astype(str).str.strip().values
    s1_cores = s1_df["sorted_core"].astype(str).str.strip().values
    s1_countries = s1_df["country"].astype(str).str.strip().values

    for sid, core, country in zip(s1_ids, s1_cores, s1_countries):
        if len(core) >= 3:
            matched = buckets.get((core, country), [])
            for cid in matched:
                pairs.append((sid, cid))

    return pairs


def phonetic_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame, max_bucket_size: int = 30) -> List[Tuple[str, str]]:
    """
    Pass 2: Phonetic NYSIIS key of core name + country.
    Memory-safe: vectorized array iteration with max_bucket_size cap.
    """
    pairs = []
    buckets = {}

    other_ids = other_df["entity_id"].astype(str).str.strip().values
    other_cores = other_df["core_name"].astype(str).str.strip().values
    other_countries = other_df["country"].astype(str).str.strip().values

    for cid, core, country in zip(other_ids, other_cores, other_countries):
        if len(core) >= 3:
            pkey = phonetic_key(core)
            if pkey:
                b = buckets.setdefault((pkey, country), [])
                if len(b) < max_bucket_size:
                    b.append(cid)

    s1_ids = s1_df["entity_id"].astype(str).str.strip().values
    s1_cores = s1_df["core_name"].astype(str).str.strip().values
    s1_countries = s1_df["country"].astype(str).str.strip().values

    for sid, core, country in zip(s1_ids, s1_cores, s1_countries):
        if len(core) >= 3:
            pkey = phonetic_key(core)
            if pkey:
                matched = buckets.get((pkey, country), [])
                for cid in matched:
                    pairs.append((sid, cid))

    return pairs


def tfidf_topk_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                        k: int = BLOCKING_TOP_K_TFIDF,
                        char_ngram: Tuple[int, int] = TFIDF_NGRAM_RANGE) -> List[Tuple[str, str]]:
    """
    Pass 2: Character n-gram TF-IDF cosine similarity, top-k per S1 entity.
    Operates on combined normalized name + address strings.
    """
    if len(s1_df) == 0 or len(other_df) == 0:
        return []

    s1_text = (s1_df["normalized_name"].fillna("") + " " + s1_df["normalized_addr"].fillna("")).tolist()
    other_text = (other_df["normalized_name"].fillna("") + " " + other_df["normalized_addr"].fillna("")).tolist()

    all_text = s1_text + other_text
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, min_df=1)
    vec.fit(all_text)

    s1_vecs = vec.transform(s1_text)
    other_vecs = vec.transform(other_text)

    n_neighbors = min(k, len(other_text))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(other_vecs)

    _, idx = nn.kneighbors(s1_vecs)

    pairs = []
    other_ids = other_df["entity_id"].values
    s1_ids = s1_df["entity_id"].values

    for i, neighbors in enumerate(idx):
        sid = str(s1_ids[i]).strip()
        for j in neighbors:
            pairs.append((sid, str(other_ids[j]).strip()))

    return pairs


def embedding_topk_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                            model: Any,
                            k: int = BLOCKING_TOP_K_EMB) -> List[Tuple[str, str]]:
    """
    Pass 3: Semantic embedding ANN (FAISS) with inner product (cosine on L2-normalized vectors).
    Catches transliteration and severe typo cases that n-gram TF-IDF might miss.
    """
    import faiss

    if len(s1_df) == 0 or len(other_df) == 0:
        return []

    s1_text = (s1_df["normalized_name"].fillna("") + " " + s1_df["normalized_addr"].fillna("")).tolist()
    other_text = (other_df["normalized_name"].fillna("") + " " + other_df["normalized_addr"].fillna("")).tolist()

    s1_emb = model.encode(s1_text, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    other_emb = model.encode(other_text, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    dim = other_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(other_emb.astype("float32"))

    n_neighbors = min(k, len(other_text))
    _, idx = index.search(s1_emb.astype("float32"), n_neighbors)

    pairs = []
    other_ids = other_df["entity_id"].values
    s1_ids = s1_df["entity_id"].values

    for i, neighbors in enumerate(idx):
        sid = str(s1_ids[i]).strip()
        for j in neighbors:
            if j >= 0:  # FAISS returns -1 for empty slots
                pairs.append((sid, str(other_ids[j]).strip()))

    return pairs


def union_candidates(*pair_lists: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Merges multiple candidate pair lists while preserving ordering and removing duplicates.
    """
    seen = set()
    merged = []
    for pairs in pair_lists:
        for p in pairs:
            if p not in seen:
                seen.add(p)
                merged.append(p)
    return merged


def candidate_list_to_dict(pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Converts a list of (s1_id, candidate_id) tuples to a dictionary mapping s1_id -> [candidate_ids]."""
    d = {}
    for sid, cid in pairs:
        d.setdefault(sid, []).append(cid)
    return d
