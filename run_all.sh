#!/usr/bin/env bash
# End-to-end reproduction of the final submission (v8 + label-shift correction) on an 8 GB / 8-core laptop.
# data -> normalisation -> label-mined dictionaries -> blocking -> features ->
# 3-stage cross-fitted LightGBM (incl. S2<->S3 anchor features) -> France self-training
# rules -> metric-aware decoding -> label-shift correction -> output/*.tsv
set -euo pipefail
cd "$(dirname "$0")"
export BER_DATA="${BER_DATA:-$(pwd)/dataset}"
export BER_WORK="${BER_WORK:-$(pwd)/work}"
export BER_OUT="${BER_OUT:-$(pwd)/output}"
export BER_GT="$BER_DATA/train/train_ground_truth.tsv"
export BER_KW=24 BER_KC=12                # blocking budget per target source (word / char view)
export BER_TRPCT=6 BER_TR2=6 BER_TR3=5 BER_VPCT=3    # row budgets sized for 8 GB RAM
export BER_NMPCT=100                      # keep every never-matched decoy in training
python src/prep.py                        # 1. normalise all source files            (~25 min)
python src/mine.py                        # 2. label-mined rewrite + transliteration dictionaries
python src/pipeline.py candidates train   # 3. blocking + features, test-like train subsample
python src/pipeline.py candidates test    # 4. blocking + features, all test S1       (~20 min)
python src/pipeline.py train              # 5. base stage-1 model (fixes the feature list)
python src/pipeline.py train2             # 6. cross-fitted stage 1 + stage-2 stacker (~60 min)
python src/pipeline.py predict2           # 7a. stage-2 test scores (input to self-training)
python src/selftrain.py                   # 7b. France self-training: mine country rules from
                                          #     confident stage-2 test matches

rm -f "$BER_WORK"/cand_test_France_*.parquet
python src/pipeline.py candidates test    # 8. re-block France with its self-trained rules
python src/pipeline.py train3             # 9. stage 3: + S2<->S3 anchor features   (~30 min)
python src/pipeline.py predict3           # 10. score test, exclusivity + expected-F0.5 decoder
python src/shift.py                       # 11. label-shift correction for sibling decoys, re-decode
                                          #     (overwrites output/matching_results.tsv)
