# Business Entity Resolution: Tech Stack & Knowledge Base

This document is the reference for every concept, tool and design decision used in the solution.

---

## 1. The problem, restated the way the metric reads it

- **Task type:** *clean–dirty record linkage*. Source 1 is deduplicated, so it is the "clean" side. Sources 2 and 3 are noisy and contain intra-source duplicates of the same entity. We link each S1 entity to a set of S2/S3 records.
- **Metric:** per-S1 F₀.₅, macro-averaged, with singletons included. For one entity with T true matches, k predicted and TP correct:

  **F₀.₅ = 1.25·TP / (0.25·T + k)**

  - Empty prediction on a true singleton scores 1. Any prediction on it scores 0.
  - A false positive raises the denominator by 1. A miss only lowers TP. Precision is worth about 2× recall.
  - Every entity weighs the same. Doing a 10-record cluster perfectly is worth the same as one singleton.

## 2. EDA facts that shaped the design (all measured on train)

| Fact | Value | Consequence |
|---|---|---|
| S1 entities | 2,206,821 | Scale: needs linear-time blocking |
| Singletons | 5.6 % | Recall matters as much as abstaining |
| Mean matches per S1 | 3.46 (≈1.7 S2 + 1.7 S3) | Set prediction, not pair classification |
| GT pairs / distinct S2-S3 ids | 7,638,365 / 7,638,365 | **Exclusivity:** every S2/S3 record belongs to ≤1 S1 entity |
| Country agreement in matched pairs | 100 % | Country is a safe blocking partition (open set) |
| S2/S3 records matching nothing | ≈26 % (train), ≈40 % inferred (test) | Distractors; test is harder, so we reproduce that in training |
| Records per S1: train vs test | 4.67 vs 5.76 | Test-like subsampling (section 6) |
| Empty target addresses | ≈4.5 % | Name-only matches need name-uniqueness features |

**Noise profile:**
- **S2:** UPPER-CASE addresses, abbreviated street types, house numbers mangled (2808 → 808, 13110 → 0013000). Some names are in **Devanagari or Gujarati** script.
- **S3:** expands US state codes (TX → Texas) but abbreviates Indian states (Maharashtra → MH). Some state names are in native script (महाराष्ट्र, കേരളം).
- **Both:** legal suffixes moved, added or dropped; `DBA` prefixes; website or hashtag names (`juarezswaim.com`, `#shahdaralearning`); OCR-style digit swaps (`c0mmunity`, `techno1ogies`); letter-scrambled names (`ARIACIRAECTO`); junk prefixes (`--`, `<<`); ID tags (`(ID: 42179)`).

## 3. Pipeline architecture

```
TSV ─► prep.py      Unicode-safe normalisation (anyascii), legal-form split, street canon, numbers
    ─► mine.py      data-mined rewrite dictionary: typo/abbrev/state-code rules + positional
                    transliteration dictionary for non-Latin-script names   (train labels only)
    ─► block.py     per (country, target source): two TF-IDF views searched with a
                    multithreaded sparse top-k matmul; union of top-24 word + top-12 char
                    ──► candidate_pairs.tsv  (≈43 candidates per S1, pair recall ≈97.9 %)
    ─► features.py  47 pairwise + list-wise features (rapidfuzz cpdist, polars window fns)
    ─► pipeline.py  name-uniqueness features ─► stage-1 LightGBM (2 cross-fitted models, OOF)
                    ─► probability-space context features ─► stage-2 LightGBM (stacker)
    ─► anchor.py    S2<->S3 record agreement with the entity's most confident matches ─► stage 3
    ─► selftrain.py France: pseudo-labels -> country-specific rewrite rules -> re-block
    ─► decode.py    exclusivity (each target keeps its best S1) + threshold / expected-F decoder
                    tuned on a held-out entity split ──► matching_results.tsv
```

## 4. Tech stack (all licences are permissive; there are no LLMs or external services)

| Layer | Tool | Licence | Why |
|---|---|---|---|
| Dataframes | polars 1.44 | MIT | Multithreaded, lazy, low-memory joins and window functions on 50M-row frames |
| Unicode folding | anyascii | ISC | Transliterates every script (Devanagari, Gujarati, Kannada, Malayalam, Bengali, accents). GPL-free, unlike unidecode |
| Sparse vectors | scikit-learn `HashingVectorizer`, scipy.sparse | BSD | Vocabulary-free (constant memory) TF-IDF over 10M documents |
| ANN / top-k | sparse_dot_topn | Apache-2.0 | Multithreaded C++ sparse matmul that keeps only the top-k per row, so blocking never materialises the full N×M |
| String similarity | rapidfuzz `process.cpdist` | MIT | Vectorised, multithreaded element-wise ratio / token_set / token_sort / partial / Jaro-Winkler / Levenshtein |
| Model | LightGBM 4.7 | MIT | Gradient-boosted trees on 7.5M rows in minutes on CPU |
| Decoder | numba | BSD | JIT-compiled Poisson-binomial expected-F search over 1.7M entities |
| Storage | pyarrow / parquet | Apache-2.0 | Columnar cache between stages |

Hardware used: a MacBook with 8 CPU cores and 8 GB RAM, CPU only. The full test run (11.7M records) takes under two hours end to end. The design is deliberately **no-GPU, no-LLM**: the licence rule (MIT/Apache, ≤8B parameters) is satisfied trivially, and the approach generalises through character-level and structural signals rather than language knowledge.

## 5. Concepts used, and where

### 5.1 Entity-resolution foundations
- **Record linkage vs deduplication; clean–dirty ER.** S1 is clean and S2/S3 are dirty. S2/S3 duplicates of one entity must *all* be returned.
- **Blocking and indexing.** Metrics: **pair completeness** (recall ceiling), **reduction ratio** (1 − candidates / all pairs) and **pair quality** (precision of candidates). Ours: PC ≈ 97.9 %; RR > 99.9999 %, since there are 43 candidates out of about 3M per country.
- **Exclusivity / assignment constraint.** This is a 1-to-many linkage with a unique parent, so each S2/S3 record keeps only its highest-scoring S1 entity.
- **Collective ER.** One pair's decision depends on its neighbours' decisions (competition for a target; other matches of the same entity). This is implemented as reverse-rank features and stage-2 stacking.

### 5.2 Normalisation
- **Unicode:** NFKC normalisation, then transliteration to ASCII, then lower-casing.
- **Legal form as a separate field.** Suffixes like Inc/LLC/Pvt/Ltd/SARL/SAS/EURL are removed from the *core* name and kept as a `legal` attribute. Agree, conflict or missing becomes a feature.
- **Street-type canonicalisation.** Rd→road, St→street, Ave/Av→avenue, Bd/Blvd→boulevard, R→rue, Imp→impasse, and so on. `N°`/`Nº`/`#`/`No` prefixes are stripped, and ordinals (1st, 5ème) collapse to numbers.
- **Numeric tokens.** Leading zeros are removed (0013000 → 13000). Numbers are compared as sets and by first-number equality, plus **substring containment** (808 ⊂ 2808), because the generator drops digits.
- **Landmark references** (near / opp / behind / à côté) are flagged.

### 5.3 Data-mined rewrite dictionary (the novel normalisation step)
- **Unexplained-token alignment.** For every ground-truth pair, if exactly one token is unexplained on each side, count (variant → canonical). A rule is kept if it has **support ≥ 30**, **confidence ≥ 0.6**, and **coverage ≥ 30 % of all occurrences** of the variant. The coverage filter stops `north → nc` from rewriting every "North Street".
- **What it learns** (about 800 rules): typos (`housotn → houston`), OCR digit swaps (`c0nsulting → consulting`), state codes (`texas → tx`, `mh → maharashtra`), and plural forms.
- **Positional transliteration dictionary** (536 rules). For names written in a non-Latin script, anyascii output is aligned token-by-token with the S1 name: `prphekt → perfect`, `sliusns → solutions`, `lojistiks → logistics`, `baumbe → bombay`. This solves the dropped-schwa problem (anyascii gives `mharastr` for महाराष्ट्र) without any external transliteration model.

### 5.4 Blocking: multi-view sparse retrieval
- **Word view.** Features are name-core tokens (`n_`), address tokens (`a_`), **name and address bigrams** (`nb_`, `ab_`), and a **sorted-token name signature** (`ns_`, which is order-invariant).
- **Char view.** Character 4-grams of the space-free core name. This catches typos, concatenated website names and word reordering.
- **Weighting.** IDF weighting with an **aggressive document-frequency cap** (feature dropped if df > 0.2 %). Bigrams make the retained features highly selective, which cut the matmul time about 15×, from 178s to 5s per 20k queries, with *no* recall loss.
- **Retrieval.** Cosine top-k via `sparse_dot_topn`. The union of top-24 (word) and top-12 (char) per target source gives about 33 candidates per source.
- **Partitioning.** Blocking runs per country label as an open set iterated from the data, so France runs through the identical code path.

### 5.5 Pairwise features (47)
- **Name:** `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, Jaro-Winkler, and space-free ratio / partial (for `wilfordhancock.com` versus "Wilford Hancock"). Also full-name token_set, core-token Jaccard, first-token equality and legal-form state.
- **Address:** token_set, token_sort, partial and normalised Levenshtein. Numeric token_set, Jaccard, first-number equality and containment. Address-empty pattern and landmark flags.
- **Retrieval scores:** word cosine and char cosine.
- **List-wise / contextual:**
  - rank of the target within the S1 entity's list, and the gap to the entity's best;
  - **reverse rank:** the rank of this S1 among all S1s competing for the same target, with gaps to the best and second-best competitor;
  - list sizes, and count of strong candidates.
- **Name uniqueness:** how many S1 entities and target records in the country share this core name. A unique name with an empty address is safe; "Silver International" (dozens of distinct businesses) is not.

### 5.6 Learning
- **Model:** gradient-boosted decision trees (LightGBM, binary log-loss, early stopping on an entity-disjoint validation split).
- **Entity-level splits** use a hash of the S1 id, so no cluster leaks across train and validation.
- **Test-distribution matching.** Test has 5.76 records per S1 against 4.67 in train. We train on 50 % of S1 entities, keep another 16 % of entities' records as *orphans* (their S1 is dropped) and all of the never-matched (decoy) records. Keeping only 50 % made the model over-predict on test. This reproduces the test's distractor density.
- **Stacking with cross-fitting.** Two stage-1 models on disjoint entity folds give **out-of-fold** probabilities for every candidate. Stage-2 features are computed in probability space:
  - the best p for this entity in each source, and the sum and count of p > 0.5;
  - p rank and gap within the entity;
  - p gap to the best competing entity for the target, and the second-best competitor.

  Stage 2 learns collective consistency, for example "this entity already has three confident matches, so this weaker one is probably also right".

### 5.7 Decision theory for the metric
- **Exclusivity projection.** Each target keeps only its highest-probability S1 entity.
- **Expected-F₀.₅ set decoder.** Under the probability-ranking principle (Lewis 1995; Jansche 2007; Nan et al. 2012), the F-optimal set is a top-k prefix of the sorted candidates. Expected F is computed exactly with Poisson-binomial convolutions (numba), and k = 0 gives E[F] = P(no true match). This compares against a tuned global threshold, and the better of the two on validation is used.
- The macro-F₀.₅ scorer is re-implemented exactly as specified, including the empty-vs-empty = 1 rule.

### 5.8 Generalising to an unseen country (France)
- No rule branches on a country value. Country is only a partition key and is never a model feature.
- All features are **relative similarities** in [0, 1] or ranks, so they transfer across languages.
- **Accent folding** and French street-type canonicalisation (R/Av/Bd/All/Imp/Chem/Rte) come from generic postal-abbreviation knowledge. This is not external data.
- French legal forms (SARL, SAS, SASU, EURL, SCI, SNC, EI) are in the legal-form lexicon.

### 5.9 Collective S2↔S3 evidence and self-training (final additions)
- **Anchor features (stage 3):** compare each candidate record-to-record with the entity's two most confident *other* candidates (name token_set, space-free ratio, address token_set, numbers), together with the anchors' probabilities. This is third-order evidence: it asks whether this record agrees with the records we are already sure about. Validation: 0.9745 → 0.9758.
- **Self-training (France):** high-confidence, exclusivity-consistent predictions serve as pseudo-labels. The same miner used on real labels then learns France-only rules (department ↔ region, frs → freres, farmacie → pharmacie, OCR legal forms). France is re-blocked and re-scored with them.
- **Larger blocking budget:** 16+8 instead of 10+6 raised the recall ceiling from 0.9727 to 0.9789 and final validation F₀.₅ to **0.9771**.
- **v8:** 24+12 budget (ceiling 0.9815), list-wise address/name-agreement features, all decoys kept in training: **0.9763** on the harder decoy-dense validation split.
- **Label-shift correction (`shift.py`):** test holds about 4× more sibling decoys (house number within ±50 on the same street) than validation. Probabilities are rescaled per house-number-offset bucket so each bucket's predicted-pair rate matches validation, then re-decoded. No test labels are used. Public leaderboard: 0.953 (v4) → **0.955**.

## 6. Compliance with the rules
- **No external data, APIs, geocoders, scraping or lookups of any kind.** Every learned artefact (rewrite dictionary, transliteration map, models) is derived *only* from the provided training files.
- **No pretrained neural model is used**, so the "MIT/Apache, ≤ 8B parameters" rule is satisfied trivially. The largest model is a LightGBM ensemble of about 1,000 trees.
- The exact candidate set scored by the model is written to `candidate_pairs.tsv`, and every match is a subset of it.

## 7. Further reading
- Fellegi & Sunter (1969), *A Theory for Record Linkage*
- Christen (2012), *Data Matching* (blocking, PC/RR/PQ)
- Papadakis et al. (2020), *Blocking and Filtering Techniques for Entity Resolution: A Survey*
- Mudgal et al. (2018), *DeepMatcher*; Li et al. (2020), *Ditto*
- Jansche (2007), *A Maximum Expected Utility Framework for Binary Sequence Labeling*; Nan, Chai, Lee & Chieu (2012), *Optimizing F-measure: A Tale of Two Approaches*
- Ke et al. (2017), *LightGBM*
- Wolpert (1992), *Stacked Generalization*
