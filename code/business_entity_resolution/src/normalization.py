"""
Stage 2: Normalization Engine for Business Entity Resolution.
Provides robust text, legal entity suffix, and address normalization.
Fully country-agnostic with graceful fallbacks (e.g., France test set).
"""

import re
import unicodedata
from typing import Dict, Any, Optional

# Comprehensive canonical legal entity suffix mapping
LEGAL_SUFFIXES = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt",
    "llc": "llc", "llp": "llp", "gmbh": "gmbh", "sarl": "sarl", "sa": "sa",
    "company": "co", "co": "co", "plc": "plc", "corp.": "corp", "inc.": "inc",
    "ltd.": "ltd", "pvt.": "pvt", "co.": "co", "nv": "nv", "bv": "bv",
    "spa": "spa", "srl": "srl"
}

# Multi-word compound suffixes (evaluated first)
COMPOUND_SUFFIXES = [
    ("private limited", "pvt ltd"),
    ("pvt limited", "pvt ltd"),
    ("private ltd", "pvt ltd"),
    ("pvt ltd", "pvt ltd"),
    ("co ltd", "co ltd"),
    ("company limited", "co ltd"),
]

LANDMARK_PATTERN = re.compile(
    r"\b(near|opp\.?|opposite|behind|next to|beside|adj\.?|adjacent to)\b.*",
    re.IGNORECASE
)

PIN_PATTERNS = {
    "india": re.compile(r"\b\d{6}\b"),
    "us": re.compile(r"\b\d{5}(-\d{4})?\b"),
    "france": re.compile(r"\b\d{5}\b"),
    "generic": re.compile(r"\b\d{4,6}\b"),
}


def normalize_name(name: Any) -> Dict[str, Any]:
    """
    Normalizes a business name:
    - NFKC Unicode normalization
    - Symbol expansion ('&' -> 'and')
    - Punctuation removal and whitespace collapsing
    - Legal entity suffix extraction & stripping
    - Token sorting for word-order invariance
    """
    if not isinstance(name, str):
        name = "" if name is None else str(name)

    raw_val = name
    n = unicodedata.normalize("NFKC", name).lower().strip()
    # Strip domain extensions (e.g. .com, .in, .org) discovered in EDA
    n = re.sub(r"\.(com|org|net|in|co|io|gov|biz|edu|co\.in)\b", "", n)

    # Check for compound suffixes before stripping punctuation
    suffix = None
    for compound, canonical in COMPOUND_SUFFIXES:
        pattern = rf"\b{re.escape(compound)}\b"
        if re.search(pattern, n):
            suffix = canonical
            n = re.sub(pattern, " ", n)
            break

    # Remove non-alphanumeric characters (except whitespace)
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()

    tokens = n.split()
    core_tokens = []

    # If compound suffix wasn't found, check individual tokens from right to left
    for t in tokens:
        if t in LEGAL_SUFFIXES:
            if suffix is None:
                suffix = LEGAL_SUFFIXES[t]
        else:
            core_tokens.append(t)

    core_name = " ".join(core_tokens)
    sorted_core = " ".join(sorted(core_tokens))

    return {
        "raw": raw_val,
        "normalized": n,
        "core": core_name,
        "sorted_core": sorted_core,
        "suffix": suffix,
    }


def normalize_address(addr: Any, country: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalizes an address:
    - NFKC Unicode normalization
    - Landmark extraction (e.g. 'near metro', 'opp bank')
    - Postal code extraction (country-specific with generic fallback)
    - Punctuation removal and token sorting for permutation invariance
    """
    if not isinstance(addr, str):
        addr = "" if addr is None else str(addr)

    raw_val = addr
    a = unicodedata.normalize("NFKC", addr).lower().strip()

    landmark_match = LANDMARK_PATTERN.search(a)
    landmark = landmark_match.group(0).strip() if landmark_match else None
    a_no_landmark = LANDMARK_PATTERN.sub("", a).strip()

    # Extract postal code
    postal = None
    c_lower = str(country).lower().strip() if country else ""
    if c_lower in PIN_PATTERNS:
        m = PIN_PATTERNS[c_lower].search(a)
        if m:
            postal = m.group(0)

    if not postal:
        # Fallback to checking all country patterns then generic
        for pat in PIN_PATTERNS.values():
            m = pat.search(a)
            if m:
                postal = m.group(0)
                break

    # Clean punctuation
    a_cleaned = re.sub(r"[^\w\s]", " ", a_no_landmark)
    a_cleaned = re.sub(r"\s+", " ", a_cleaned).strip()
    tokens = a_cleaned.split()

    return {
        "raw": raw_val,
        "normalized": a_cleaned,
        "sorted_tokens": " ".join(sorted(tokens)),
        "postal_code": postal,
        "landmark": landmark,
    }


def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single entity row containing name, address, country, entity_id.
    """
    name_norm = normalize_name(row.get("name", ""))
    addr_norm = normalize_address(row.get("address", ""), row.get("country", ""))

    return {
        "entity_id": str(row.get("entity_id", "")).strip(),
        "country": str(row.get("country", "")).lower().strip(),
        "raw_name": name_norm["raw"],
        "normalized_name": name_norm["normalized"],
        "core_name": name_norm["core"],
        "sorted_core": name_norm["sorted_core"],
        "suffix": name_norm["suffix"],
        "raw_addr": addr_norm["raw"],
        "normalized_addr": addr_norm["normalized"],
        "sorted_addr_tokens": addr_norm["sorted_tokens"],
        "postal_code": addr_norm["postal_code"],
        "landmark": addr_norm["landmark"],
    }
