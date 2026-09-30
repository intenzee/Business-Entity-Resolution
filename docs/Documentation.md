# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Bugwiserz
**Team Members:** varnit, Tanmay Roy, Dev Sharma, Divyanshi Rawat
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We treat the task as **per-entity set prediction under a macro-F₀.₅ metric**, not as pair classification.

The pipeline runs in five stages:

1. **Normalisation.** Country-agnostic and Unicode-safe, extended with a **rewrite and transliteration dictionary mined from the training labels**.
2. **Blocking.** A two-view sparse TF-IDF search (word/bigram view and char 4-gram view) with a multithreaded top-k sparse matmul.
3. **Feature scoring.** 48 pairwise, list-wise and name-uniqueness features (61 in stage 2, 74 in stage 3) feed a **cross-fitted two-stage LightGBM stack**. The second stage reasons collectively, in probability space, about competition for each record.
4. **Structural decoding.** An exclusivity constraint (each S2/S3 record joins at most one S1 entity) and an **exact expected-F₀.₅ set decoder**.
5. **Distribution matching.** The training data is re-sampled to reproduce the test set's higher distractor density (all never-matched decoy records kept), and a label-free **label-shift correction** rescales probabilities for nearby-house-number "sibling" decoys, which are about 4× denser in test than in validation.

Held-out validation macro F₀.₅ is **0.9763** (US 0.9785, India 0.9731) on a deliberately harder, decoy-dense validation split, against a blocking recall ceiling of 0.9815. Public leaderboard: **0.955**. Everything runs on an 8-core, 8 GB laptop on CPU, with no external data and no pretrained neural models.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on 2.2M train S1 entities, 5.0M S2 and 5.3M S3 records:

- **Match structure.**
  - 5.6 % of S1 entities are singletons; the mean is 3.46 matches (≈1.7 per source) and the maximum 11.
  - **7,638,365 ground-truth pairs contain 7,638,365 distinct S2/S3 ids.** Every S2/S3 record belongs to at most one S1 entity, which we exploit as a hard exclusivity constraint.
  - Matched pairs share the country label 100 % of the time, so country is a safe partition key. It is treated as an open set, which is how France is handled.
- **Distractors.** About 26 % of train S2/S3 records match nothing. The test set has **5.76 target records per S1 against 4.67 in train**, so it contains more "orphan" records whose S1 anchor is absent.
- **Name noise.**
  - Legal suffixes are moved, dropped or duplicated ("LLC HALL AND LYONS", "Roldan,-Nugent and Gould LLC LLC").
  - Names appear in **Devanagari, Gujarati and other Indic scripts** ("ग्रेट फाउंडेशन प्राइवेट लिमिटेड" = Great Foundation Private Limited).
  - Some names are websites or hashtags (`juarezswaim.com`, `#shahdaralearning`).
  - There are OCR digit swaps (`c0mmunity`, `techno1ogies`), letter scrambling (`Constbruciton`), junk prefixes (`--`, `<<`) and ID tags.
- **Address noise.**
  - S2 is upper-case and abbreviated, and mangles house numbers (2808 → 808, 13110 → 0013000).
  - S3 spells out US states (TX → Texas) but abbreviates Indian states (Maharashtra → MH), sometimes in native script (महाराष्ट्र).
  - Other variations: missing components, reordered components, `<NULL>` placeholders, "Near …" landmarks, and about 4.5 % empty addresses.
- **Ambiguity.** Many distinct Indian businesses share an identical name (e.g. "Silver International"). Name-only evidence is only safe when the name is rare.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-view blocking, then a stacked GBDT classifier, then structured, metric-aware decoding.

**Core Innovations:**

1. **Label-mined canonicalisation.**
   - A rewrite dictionary of about 800 rules (typos, abbreviations, state codes) from one-token-difference alignments of true pairs, filtered by support, confidence and coverage.
   - A **536-entry positional transliteration dictionary** for non-Latin-script names (`prphekt → perfect`, `lojistiks → logistics`).
2. **Test-distribution matching.** Training on a subsample that reproduces the test's orphan/distractor density.
3. **Collective stacking (three stages).** Cross-fitted out-of-fold stage-1 probabilities become list-wise features for stage 2. The top stage-2 features are the probability gap to the best competing S1 for the same record, the reverse rank, and the best competitor's probability.
4. **S2↔S3 anchor features (stage 3).** Each candidate is compared record-to-record with the entity's two most confident _other_ candidates: name, address and number agreement, plus the anchors' probabilities. All S2/S3 records of one business are noisy copies of each other, so agreement with a near-certain match is strong evidence.
5. **Self-training for the unseen country.** Confident, exclusivity-consistent France predictions (744,624 pseudo-pairs) are used as pseudo-labels to mine France-only rewrite rules. Examples: `gironde → nouvelle aquitaine`, `nord → hauts de france`, `farmacie → pharmacie`, `5arl → sarl`. France is then re-blocked and re-scored with them. Only the provided test records are used.
6. **Metric-aware decoding.** Exclusivity projection plus the exact expected-F₀.₅ top-k decoder (Poisson-binomial, numba).

---

## 3. Candidate Generation (Blocking)

Blocking runs separately for each (country label present in S1) × (target source S2, S3). There are no hard-coded countries.

- **Blocking keys used.** Two IDF-weighted TF-IDF views, each searched by cosine top-k with `sparse_dot_topn`, a multithreaded sparse matmul that keeps only the top-k per row:
  1. **Word view:**
     - name-core tokens and address tokens (after the mined rewrite dictionary);
     - **name bigrams and address bigrams** (e.g. `2808_red`, `red_hawk`);
     - an order-invariant sorted-token name signature.
       Features in more than 0.2 % of documents are dropped. The bigrams keep retrieval selective, which cut search time about 15× with no recall loss.
  2. **Char view:** character 4-grams of the space-free core name. This catches typos, concatenated website names and reordering when the address is missing or garbled.
  3. **Union** of the top-24 word and top-12 char neighbours per target source. Each pair keeps both cosine scores and both ranks as features.
- **Candidate pairs generated (test):** **113,729,744** (65.6 per S1 entity).

  The reduction ratio is above 99.9999 % against the full cross product.

- **How we ensured true matches were not lost.**
  - Recall was measured on train against the **full** S2/S3 pool, so distractor density is realistic. Pair completeness is **0.9815** on the held-out split (24+12 budget). The 16+8 budget reached 0.9789 and 10+6 reached 0.9727.
  - The two views are complementary: the word view recovers typo'd names through intact addresses, and the char view recovers empty or garbled addresses through names.
  - Remaining misses are mostly duplicate-name businesses whose address components differ entirely, e.g. a plot number in S2 but not in S1.

---

## 4. Matching Model

**Features used (48 in stage 1, 61 in stage 2, 74 in stage 3):**

- **Name features:**
  - rapidfuzz `ratio`, `token_sort_ratio`, `token_set_ratio` and `partial_ratio`, and Jaro-Winkler on the core name;
  - space-free ratio and partial ratio (website and concatenated names);
  - token_set on the full name, core-token Jaccard and intersection size, first-token equality, token counts;
  - legal-form state (agree / conflict / missing).
- **Address features:**
  - token_set, token_sort, partial and normalised Levenshtein on the canonicalised address;
  - numeric-token token_set, Jaccard, intersection and union sizes, first-number equality;
  - **first-number containment** (808 ⊂ 2808, which handles digit-drop noise);
  - empty-address pattern and landmark flags.
- **Other features:**
  - Retrieval: word-view and char-view cosine.
  - Within-entity context: the candidate's rank in the S1 entity's list, and gaps to that entity's best (overall, word, char).
  - **Reverse rank:** the rank of this S1 among all S1s competing for the same target, with gaps to the best and second-best competitor. Competitor count and list sizes.
  - **Name uniqueness:** the number of S1 entities and target records sharing the same core name in the country.
  - Source indicator and non-Latin-script flag.
  - **Stage 2 only:** the out-of-fold stage-1 probability p1, plus p1-space context:
    - p1 rank and gap within the entity; the entity's best p1, sum of p1 and count of p1 > 0.5, both per source and across sources;
    - p1 gap to the best competing entity for the target, the second-best competitor's p1, and the sum of p1 over competitors.

**Model type:** LightGBM (binary log-loss; 127 leaves; learning rate 0.08; early stopping on an entity-disjoint validation set).

- **Stage 1:** two models cross-fitted on disjoint 12 % entity folds, so every train candidate gets an out-of-fold probability.
- **Stage 2:** a stacker trained on 10.5M candidate rows.
- **Stage 3:** stage 2 plus 13 anchor features (S2↔S3 record agreement), trained on 8.5M rows.
- **Validation:** all splits are by hashed S1 id, so clusters never leak across folds.

**Threshold selection method:** direct macro-F₀.₅ optimisation on the held-out entity split.

- We compare global thresholds with the exact **expected-F₀.₅ decoder**. For each entity it scores every top-k prefix of its sorted candidates by E[F₀.₅] under a Poisson-binomial model, where k = 0 means "predict empty".
- Both are applied after the **exclusivity projection**. The decoder won (0.97710 vs 0.97672 for the best threshold, 0.7) and is used for the final submission.

---

## 5. Results & Error Analysis

| System (held-out split: 132,003 S1 entities, 456,907 true pairs)      | Macro F₀.₅        |
| --------------------------------------------------------------------- | ----------------- |
| Stage 1, threshold 0.5                                                | 0.96812           |
| Stage 1, best threshold (0.7)                                         | 0.97180           |
| Stage 1, expected-F decoder                                           | 0.97029           |
| Stage 2 stack + name-uniqueness + exclusivity + expected-F decoder    | 0.97450           |
| + stage 3 S2↔S3 anchor features + France self-training (v3)           | 0.97582           |
| + larger blocking budget 16+8 (ceiling 0.9727 → 0.9789), v4           | 0.97710           |
| – of which US / India (v4)                                            | 0.97903 / 0.97420 |
| **v8: 100 % decoys in training, list-wise address/name-agreement features, budget 24+12 (ceiling 0.9815)** — harder validation split, not comparable to rows above | **0.97633** |
| – of which US / India (v8)                                            | 0.97847 / 0.97313 |
| v8 + label-shift correction (final submission)                        | public LB **0.955** (v4: 0.953) |

Rows 1–4 are measured on the 6 % entity hold-out, row 5 on a 4 % hold-out (memory limit). From v8 on, validation keeps every never-matched decoy record, so it is harder than the v4 split and the two scores are not directly comparable.

**Label-shift correction (step 11, `src/shift.py`).** Checking predictions without labels showed the model predicting more matches per entity on test (US 3.49) than validation (3.31), while the train ground truth has the same distribution in every country. The excess came from sibling decoys: same street, house number within ±50. For each country we bucket pairs by house-number offset (≤10, ≤50, ≤500, larger) × whether the entity already has a confident match at its own number, and multiply p by min(1, validation rate ÷ test rate) of decoded pairs in that bucket (3 fixed-point iterations), then re-decode. After correction, US / India / France average 3.29 / 3.32 / 3.21 matches per entity, against 3.31 / 3.29 in validation. No test labels are used.

- **F₀.₅ score (macro):** **0.9763** on the decoy-dense validation split; **0.955** on the public leaderboard (final submission).
- **Common false positives (wrong merges):**
  - Same-name businesses at the same building but a different unit ("Eco India Pvt Ltd" vs "Eco Structures Pvt Ltd", U166 vs U169).
  - Near-identical house numbers (13504 vs 13505b).
  - Matches where the target address is empty and the name is common.
- **Common false negatives (missed matches):**
  - Records with an empty address and a name shared by several S1 entities (correctly uncertain).
  - Letter-scrambled or replaced names ("ARIACIRAECTO", "Kelovantage") with partially mangled addresses.
  - S2/S3 addresses carrying components absent from S1 (plot numbers, a different locality).

  About 1.8 % of true pairs are never retrieved by blocking, so they bound the achievable recall. Many of these have a randomly replaced name ("Polaris" → "Veraxyloquo") and an empty address, so no string or semantic similarity can recover them.

---

## 6. Conclusion

Optimising for the leaderboard's actual structure gave consistent gains over a plain classifier and threshold: exclusivity, competition for each record, name uniqueness, a test-like distractor density and the macro set-level metric. Mining normalisation rules and transliterations _from the labels themselves_ handled Indic scripts and state-code noise with no hand-built lexicons or external resources.

The remaining headroom is in blocking recall and in unresolvable same-name ambiguity. Next steps would be adding S2↔S3 record-to-record agreement as a third-order feature, and pseudo-label self-training on high-confidence France matches to adapt the model to the unseen country.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── run_all.sh          # one-command end-to-end reproduction
├── requirements.txt    # pinned versions (Python 3.12)
├── README.md
└── src/
    ├── norm.py         # Unicode folding, legal-form split, street canon, numbers, landmarks
    ├── prep.py         # parallel normalisation of all source files -> parquet
    ├── mine.py         # label-mined rewrite dictionary + positional transliteration dictionary
    ├── block.py        # two-view TF-IDF + sparse top-k blocking
    ├── features.py     # pairwise + list-wise features
    ├── decode.py       # exclusivity, expected-F0.5 decoder (numba), exact macro-F0.5 scorer
    ├── anchor.py       # stage-3 S2<->S3 anchor features
    ├── selftrain.py    # France self-training rules from confident stage-2 matches
    ├── pipeline.py     # candidates / train / train2 / train3 / predict2 / predict3 / output writers
    └── shift.py        # step 11: label-shift correction for sibling decoys + final decode
```

Entry point: `./run_all.sh`, which writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results

- **Blocking recall** (20k S1 sample per country vs the full target pool, top-12 word / top-8 char):

  | Source | India | US    |
  | ------ | ----- | ----- |
  | S2     | 0.977 | 0.974 |
  | S3     | 0.952 | 0.977 |

- **Blocking speed-up:** replacing unigram TF-IDF (df cap 2 %) with bigram features (df cap 0.2 %) and char 3-grams with char 4-grams cut top-k time from 251 s to 25 s per 20k queries, with recall unchanged or better.
- **Stage-2 feature importance (gain):** p1 ≫ p1 gap to the best competitor for the target > reverse rank > best competitor's p1 > sum of competitor p1.
- **Compliance:** no external data, lookups or APIs. All learned artefacts come from the provided training files. No pretrained or neural models are used; LightGBM is MIT-licensed.
- A full concept and tech-stack reference is in [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md).
