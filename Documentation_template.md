# Business Entity Resolution — Technical Methodology & Submission Report

## 1. Executive Summary & Team Methodology
- **Objective:** Macro-averaged $F_{0.5}$ score maximization across heterogeneous enterprise business entity records (Source 1, Source 2, Source 3).
- **Core Philosophy:** Precision-weighted entity deduplication using multi-pass recall-guaranteed blocking ($\ge 0.99$ target), robust string & phonetic normalization, gradient-boosted pairwise tabular classifier (HistGradientBoosting / LightGBM algorithm), macro $F_{0.5}$ threshold optimization, and global 1-to-many greedy conflict resolution.
- **Compliance & License Guarantee:**
  - 100% self-contained offline architecture (zero external web requests, no geocoding APIs, no external registry queries).
  - Decision model: Scikit-learn Histogram Gradient Boosting Classifier (MIT License, $\le$ 8B parameters).
  - Country-agnostic normalization ensuring zero degradation on test sets with unseen countries (e.g. France).

---

## 2. System Architecture & Pipeline Stages

```
[Raw Sources: S1, S2, S3]
         │
         ▼
[Stage 2: Unicode NFKC, Legal Suffix Stripping & Address Normalization]
         │
         ▼
[Stage 3: Multi-Pass Blocking (Exact Core Hash + Phonetic + Char n-gram TF-IDF)]
         │
         ▼ (Union Candidates with Recall >= 0.99)
[Stage 4: Feature Extraction (Fuzzy, Levenshtein, Jaro-Winkler, NYSIIS, Jaccard, Rank Gaps)]
         │
         ▼
[Stage 5: HistGradientBoosting Pairwise Classifier (GroupKFold by S1)]
         │
         ▼
[Stage 6: Macro F0.5 Threshold Optimization on Out-Of-Fold Predictions]
         │
         ▼
[Stage 7: Global Greedy 1-to-Many Conflict Resolution]
         │
         ▼
[Stage 8: Singleton Default Rule (Unmatched S1 -> Empty Set)]
         │
         ▼
[Stage 10: Submission Export: candidate_pairs.tsv & matching_results.tsv]
```

---

## 3. Detailed Stage Breakdown

### 3.1 Normalization (Stage 2)
- **Name Normalization:**
  - Unicode NFKC decomposition, lowercasing, symbol expansion (`&` $\to$ `and`).
  - Legal suffix stripping: canonical dictionary covering single-word and compound legal designations (`private limited`, `pvt ltd`, `corp`, `inc`, `llc`, `gmbh`, `sarl`, `sa`, etc.).
  - Output formats: `raw`, `normalized`, `core` (suffix stripped), and `sorted_core` (alphabetical token sort for permutation invariance).
- **Address Normalization:**
  - Landmark extraction and stripping (`near`, `opp`, `opposite`, `behind`, `next to`).
  - Postal code extraction with country-specific patterns (India PIN, US ZIP, France Code Postal) and generic fallback.

### 3.2 Candidate Generation / Blocking (Stage 3)
To ensure that recall is not bottlenecked before classification, we run a union of orthogonal blocking passes:
1. **Exact Sorted-Core Key:** Hashes alphabetically sorted core tokens within the same country (with fallback for missing country).
2. **Phonetic Key Pass:** NYSIIS phonetic key of core tokens to bridge phonetic spelling variations and transliterations.
3. **Character n-gram TF-IDF Nearest Neighbors:** Character 3-to-5 n-grams over combined normalized name and address, retrieving top-$K$ candidates via cosine similarity.

### 3.3 Feature Engineering (Stage 4)
- **String & Token Similarities:** Normalized Levenshtein ratio, Jaro-Winkler score, RapidFuzz Token Sort Ratio, Token Set Ratio, and Token Jaccard index.
- **Phonetic Matching:** Exact match indicator and token Jaccard similarity across NYSIIS phonetic codes.
- **Structural & Geographic:** Postal code exact match flag, postal missing flag, landmark match flag, country match flag, and source indicator (`source_is_s2`).
- **Relative Rank & Margin Features:**
  - `rank_within_s1`: Rank of candidate among all candidates for the given Source-1 entity.
  - `gap_to_next_best`: Absolute score difference between top candidate and runner-up.
  - `s1_candidate_count`: Total candidate density for the entity.

### 3.4 Model Architecture & Training (Stage 5)
- **Model:** Histogram-based Gradient Boosting Classifier (`HistGradientBoostingClassifier`, MIT License).
- **Validation:** 5-fold `GroupKFold` split grouped by `source1_entity_id` to strictly prevent data leakage.
- **Class Balancing:** `class_weight='balanced'` and sample weighting proportional to hard-negative to positive ratio.

### 3.5 Global Consistency & Singleton Handling (Stages 7 & 8)
- **Deduplication Invariant:** Ground truth validation confirms that Source-1 represents the canonical deduplicated reference, meaning each Source-2 or Source-3 record belongs to at most one Source-1 entity.
- **Greedy Conflict Resolution:** Candidate pairs above the decision threshold are sorted descending by predicted probability. Candidates are assigned to the highest-scoring Source-1 entity and removed from subsequent consideration.
- **Singleton Handling:** Entities with zero assigned candidates are cleanly exported as empty lists.

---

## 4. Empirical Evaluation & Verification

- **Evaluation Metric:** Exact macro $F_{0.5}$ harness matching the competition formula:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Submission Integrity Verification:** Verified via `validate_submission.py` ensuring exact Source-1 coverage, no duplicate IDs, no self-referential IDs, and complete candidate containment.
