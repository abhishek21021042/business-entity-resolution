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

import re

def first_word_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame, max_bucket_size: int = 30) -> List[Tuple[str, str]]:
    """
    Pass 4: Distinctive first word of core business name + country.
    Catches multi-word company names where the tail had severe OCR corruption/typos.
    """
    stopwords = {"the", "and", "for", "with", "all", "new", "top", "pro", "best", "inc", "ltd", "pvt", "corp", "llc", "co"}
    pairs = []
    buckets = {}

    other_ids = other_df["entity_id"].astype(str).str.strip().values
    other_cores = other_df["core_name"].fillna("").astype(str).str.strip().values
    other_countries = other_df["country"].fillna("").astype(str).str.strip().values

    for cid, core, ctry in zip(other_ids, other_cores, other_countries):
        toks = [t for t in core.split() if t not in {"m/s", "dr", "mr", "ms", "sri", "shri"}]
        if toks and len(toks[0]) >= 3 and toks[0] not in stopwords:
            key = (toks[0], ctry)
            b = buckets.setdefault(key, [])
            if len(b) < max_bucket_size:
                b.append(cid)

    s1_ids = s1_df["entity_id"].astype(str).str.strip().values
    s1_cores = s1_df["core_name"].fillna("").astype(str).str.strip().values
    s1_countries = s1_df["country"].fillna("").astype(str).str.strip().values

    for sid, core, ctry in zip(s1_ids, s1_cores, s1_countries):
        toks = [t for t in core.split() if t not in {"m/s", "dr", "mr", "ms", "sri", "shri"}]
        if toks and len(toks[0]) >= 3 and toks[0] not in stopwords:
            key = (toks[0], ctry)
            for cid in buckets.get(key, []):
                pairs.append((sid, cid))

    return pairs


def address_key_blocking(s1_df: pd.DataFrame, other_df: pd.DataFrame, max_bucket_size: int = 30) -> List[Tuple[str, str]]:
    """
    Pass 5: Exact house/flat/plot number + locality or postal code hashing.
    Catches DBA / Brand / Indic script name differences where address is identical.
    """
    pairs = []
    buckets = {}

    def get_keys(addr_str: str, country: str, postal: Any):
        if not addr_str:
            return []
        keys = []
        raw_nums = re.findall(r"\b[a-zA-Z]?[-#]?\d+[/a-zA-Z\-_]*\d*[a-zA-Z]?\b", addr_str.lower())
        nums = set()
        for n in raw_nums:
            clean_n = n.strip("-# ").replace(" ", "")
            if len(clean_n) >= 2 and any(c.isdigit() for c in clean_n):
                nums.add(clean_n)
                digits = re.sub(r"\D", "", clean_n)
                if len(digits) >= 2:
                    nums.add(digits)
                    nums.add(digits.lstrip("0"))
                clean_no_zero = clean_n.lstrip("0")
                if len(clean_no_zero) >= 2:
                    nums.add(clean_no_zero)

        clean_addr = re.sub(r"[^\w\s]", " ", addr_str.lower())
        tokens = [t for t in clean_addr.split() if len(t) >= 3 and not t.isdigit()]

        if postal and postal != "None" and len(str(postal)) >= 4:
            keys.append((f"post_{postal}", country))
            for n in list(nums)[:3]:
                keys.append((f"{postal}_{n}", country))

        for n in list(nums)[:4]:
            for t in tokens:
                if t not in {"near", "opp", "opposite", "behind", "floor", "road", "street", "lane", "plot", "flat", "unit", "building", "complex", "house", "phase"}:
                    keys.append((f"{n}_{t}", country))
        return keys

    other_ids = other_df["entity_id"].astype(str).str.strip().values
    other_addrs = other_df.get("raw_addr", other_df.get("address", pd.Series([""]*len(other_df)))).fillna("").astype(str).values
    other_countries = other_df["country"].fillna("").astype(str).str.strip().values
    other_postals = other_df["postal_code"].fillna("").values if "postal_code" in other_df else [None]*len(other_df)

    for cid, addr, ctry, post in zip(other_ids, other_addrs, other_countries, other_postals):
        for k in get_keys(addr, ctry, post):
            b = buckets.setdefault(k, [])
            if len(b) < max_bucket_size:
                b.append(cid)

    s1_ids = s1_df["entity_id"].astype(str).str.strip().values
    s1_addrs = s1_df.get("raw_addr", s1_df.get("address", pd.Series([""]*len(s1_df)))).fillna("").astype(str).values
    s1_countries = s1_df["country"].fillna("").astype(str).str.strip().values
    s1_postals = s1_df["postal_code"].fillna("").values if "postal_code" in s1_df else [None]*len(s1_df)

    for sid, addr, ctry, post in zip(s1_ids, s1_addrs, s1_countries, s1_postals):
        matched = set()
        for k in get_keys(addr, ctry, post):
            for cid in buckets.get(k, []):
                if cid not in matched:
                    matched.add(cid)
                    pairs.append((sid, cid))

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
