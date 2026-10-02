"""Data-mined token canonicalisation dictionary (learned from training pairs only).

For each ground-truth pair (S1, S2/S3) we compute, per field, the tokens present only on
the S2/S3 side and only on the S1 side. When exactly one token is unexplained on each
side, (variant -> canonical) is counted. High-support, high-confidence rules become a
rewrite map applied to every record. This learns, without hand-written lists:
  * transliterations of Indic-script words  (praivet -> private, gret -> great)
  * state / region codes vs full names       (texas -> tx, mh -> maharashtra)
  * abbreviation variants                    (ln -> lane, pvt -> private, ...)
"""
import json
import os
import sys

import polars as pl

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/business_entity_resolution
WORK = os.environ.get("BER_WORK", os.path.join(_ROOT, "work"))
GT = os.environ.get("BER_GT", os.path.join(os.environ.get("BER_DATA", os.path.join(_ROOT, "..", "..", "dataset")), "train", "train_ground_truth.tsv"))


def load_gt_pairs():
    gt = pl.read_csv(GT, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .rename({"source1_entity_id": "q", "matched_entity_ids": "t"}))


def mine(field, pairs, s1, tg, min_count=30, min_conf=0.6):
    d = (pairs.join(s1.select(pl.col("id").alias("q"), pl.col(field).alias("a")), on="q")
         .join(tg.select(pl.col("id").alias("t"), pl.col(field).alias("b")), on="t")
         .select(a=pl.col("a").str.split(" "), b=pl.col("b").str.split(" ")))
    occ = d.select(pl.col("b").list.unique()).explode("b").group_by("b").len().rename({"b": "x", "len": "occ"})
    d = d.select(x=pl.col("b").list.set_difference("a"), y=pl.col("a").list.set_difference("b"))
    d = d.filter((pl.col("x").list.len() == 1) & (pl.col("y").list.len() == 1))
    d = d.select(x=pl.col("x").list.first(), y=pl.col("y").list.first()).filter(
        (pl.col("x") != "") & (pl.col("y") != ""))
    c = d.group_by("x", "y").len()
    tot = c.group_by("x").agg(pl.col("len").sum().alias("tot"))
    c = c.join(tot, on="x").join(occ, on="x").with_columns(conf=pl.col("len") / pl.col("tot"),
                                                          support=pl.col("len") / pl.col("occ"))
    c = c.filter((pl.col("len") >= min_count) & (pl.col("conf") >= min_conf) & (pl.col("support") >= 0.3)
                 & ~pl.col("x").str.contains(r"^\d+$") & ~pl.col("y").str.contains(r"^\d+$"))
    return dict(zip(c["x"].to_list(), c["y"].to_list()))


def mine_translit(pairs, s1, tg, min_count=3, min_conf=0.5):
    """Positional alignment for names written in a non-Latin script (indic=1): after
    anyascii folding each token is a phonetic transliteration of the S1 token at the
    same position (e.g. 'gret phaumdesn praivet limited' ~ 'great foundation private limited')."""
    d = (pairs.join(tg.filter(pl.col("indic") == 1).select(pl.col("id").alias("t"), pl.col("nm").alias("b")), on="t")
         .join(s1.select(pl.col("id").alias("q"), pl.col("nm").alias("a")), on="q")
         .select(a=pl.col("a").str.split(" "), b=pl.col("b").str.split(" ")))
    d = d.filter(pl.col("a").list.len() == pl.col("b").list.len()).explode("a", "b")
    c = d.rename({"b": "x", "a": "y"}).group_by("x", "y").len()
    tot = c.group_by("x").agg(pl.col("len").sum().alias("tot"))
    c = c.join(tot, on="x").filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_conf)
                                   & (pl.col("x") != pl.col("y")))
    return dict(zip(c["x"].to_list(), c["y"].to_list()))


if __name__ == "__main__":
    pairs = load_gt_pairs()
    s1 = pl.read_parquet(f"{WORK}/train_s1.parquet", columns=["id", "nm", "addr"])
    maps = {"nm": {}, "addr": {}}
    translit = {}
    for src in ["2", "3"]:
        tg = pl.read_parquet(f"{WORK}/train_s{src}.parquet", columns=["id", "nm", "addr"])
        for f in maps:
            m = mine(f, pairs, s1, tg)
            print(src, f, len(m), list(m.items())[:40], flush=True)
            for k, v in m.items():
                maps[f].setdefault(k, v)
        tr = mine_translit(pairs, s1, pl.read_parquet(f"{WORK}/train_s{src}.parquet", columns=["id", "nm", "indic"]))
        print(src, "translit", len(tr), list(tr.items())[:30], flush=True)
        for k, v in tr.items():
            translit.setdefault(k, v)
        del tg
    # never rewrite a token that is itself a canonical target (avoid chains/cycles)
    for f in maps:
        tgts = set(maps[f].values())
        maps[f] = {k: v for k, v in maps[f].items() if k not in tgts}
    maps["translit"] = translit
    json.dump(maps, open(f"{WORK}/token_map.json", "w"))
    print({f: len(m) for f, m in maps.items()})
