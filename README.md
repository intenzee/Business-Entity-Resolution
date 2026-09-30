# Business Entity Resolution: reproduction guide

Clean-to-dirty business record linkage for ML Challenge 2026: two-view sparse blocking, a three-stage LightGBM stack, an exact expected-F0.5 set decoder, and a label-shift correction for nearby-house-number decoys. Validation macro F0.5 is 0.9763; public leaderboard score is 0.955. The method is written up in [`docs/Documentation.md`](docs/Documentation.md).

The challenge data and generated outputs (`candidate_pairs.tsv` is 1.5 GB) are not in this repo.

## Environment
- Python 3.12. CPU only; tested on macOS arm64 with 8 cores and 8 GB RAM.
- `pip install -r requirements.txt`. All dependencies are MIT/BSD/Apache/ISC licensed.
- No internet access is needed at run time, and no external data, APIs or pretrained models are used.

## Data layout
Point `BER_DATA` at the folder containing `train/` and `test/` from the challenge
(`train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `test_source{1,2,3}.tsv`).

## Run end-to-end
```bash
export BER_DATA=/path/to/student_resource/dataset
export BER_OUT=/path/to/output          # where the two TSVs are written
./run_all.sh
```

`run_all.sh` runs these steps in order:

1. `prep.py`: normalise all source files and cache them as parquet (~25 min).
2. `mine.py`: mine the rewrite and transliteration dictionaries from train labels.
3. `pipeline.py candidates train|test`: two-view sparse blocking (24 word + 12 char per source) plus pairwise and list-wise features.
4. `pipeline.py train`, then `train2`: stage-1 base model, cross-fitted stage 1, out-of-fold probabilities, stage-2 stacker.
5. `pipeline.py predict2`, then `selftrain.py`: France self-training, which mines country rules from confident stage-2 test matches.
6. Re-block France, then `pipeline.py train3` (stage 3 with S2<->S3 anchor features), then `predict3` (exclusivity + expected-F0.5 decoder).
7. `shift.py`: label-shift correction for nearby-house-number decoys (per-bucket rescale of p using validation vs test predicted-pair rates, no test labels), then the final decode.

Total time is about 4 hours on an 8-core / 8 GB laptop.

Outputs:
- `$BER_OUT/matching_results.tsv`: the final matches.
- `$BER_OUT/candidate_pairs.tsv`: exactly the pairs the model scored. The matches are a subset of these.

Validate with:
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source files
- `src/norm.py`: normalisation (Unicode to ASCII, legal forms, street types, numbers, landmarks)
- `src/prep.py`: parallel preprocessing
- `src/mine.py`: data-mined token dictionary and transliteration dictionary
- `src/block.py`: two-view sparse TF-IDF top-k blocking
- `src/features.py`: pairwise and list-wise features
- `src/anchor.py`: collective S2<->S3 anchor features (stage 3)
- `src/selftrain.py`: self-training rule mining for an unseen country
- `src/decode.py`: exclusivity, expected-F0.5 decoder, macro-F0.5 scorer
- `src/pipeline.py`: stage orchestration (candidates / train / train2 / train3 / predict2 / predict3)
- `src/shift.py`: label-shift correction for sibling decoys and the final decode

See [`docs/KNOWLEDGE_BASE.md`](docs/KNOWLEDGE_BASE.md) for the concepts and design rationale.
