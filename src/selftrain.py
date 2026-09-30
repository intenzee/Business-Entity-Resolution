"""Self-training for an unseen country (France): no labels exist, so we treat the
current model's high-confidence, exclusivity-consistent test matches as pseudo-labels and
mine *country-specific* rewrite rules from them, exactly like mine.py does from the real
train labels. Only test data already provided by the challenge is used.

Rules allow a single variant token to map to a short canonical phrase, which captures
French department <-> region substitutions (e.g. 'gironde' vs 'nouvelle aquitaine') and
street-type abbreviations the generic seed list does not cover.
The rules are applied only to records carrying that country label.
"""
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import WORK, load  # noqa: E402

TARGET = os.environ.get("BER_SELFTRAIN_COUNTRIES", "France").split(",")


def pseudo_pairs(country, p_min=0.98):
    sc = pl.read_parquet(f"{WORK}/score2_{country}.parquet")
    sc = sc.filter(pl.col("p") == pl.col("p").max().over("t"))
    return sc.filter(pl.col("p") >= p_min).select("q", "t")


def mine_multi(field, pairs, s1, tg, min_count=20, min_conf=0.6, min_support=0.3):
    d = (pairs.join(s1.select(pl.col("id").alias("q"), pl.col(field).alias("a")), on="q")
         .join(tg.select(pl.col("id").alias("t"), pl.col(field).alias("b")), on="t")
         .select(a=pl.col("a").str.split(" "), b=pl.col("b").str.split(" ")))
    occ = d.select(pl.col("b").list.unique()).explode("b").group_by("b").len().rename({"b": "x", "len": "occ"})
    d = d.select(x=pl.col("b").list.set_difference("a"), y=pl.col("a").list.set_difference("b"))
    d = d.filter((pl.col("x").list.len() == 1) & (pl.col("y").list.len().is_between(1, 3)))
    d = d.select(x=pl.col("x").list.first(), y=pl.col("y").list.sort().list.join(" "))
    c = d.group_by("x", "y").len()
    tot = c.group_by("x").agg(pl.col("len").sum().alias("tot"))
    c = c.join(tot, on="x").join(occ, on="x").with_columns(conf=pl.col("len") / pl.col("tot"),
                                                          support=pl.col("len") / pl.col("occ"))
    c = c.filter((pl.col("len") >= min_count) & (pl.col("conf") >= min_conf) & (pl.col("support") >= min_support)
                 & ~pl.col("x").str.contains(r"^\d+$") & (pl.col("x") != "") & (pl.col("y") != ""))
    return dict(zip(c["x"].to_list(), c["y"].to_list())), c.sort("len", descending=True)


if __name__ == "__main__":
    maps = json.load(open(f"{WORK}/token_map.json"))
    base = {k: v for k, v in maps.items() if k != "by_country"}
    by_country = {}
    for c in TARGET:
        pairs = pseudo_pairs(c)
        s1 = load("test", "1", base).filter(pl.col("country") == c)
        rules = {"nm": {}, "addr": {}}
        for src in "23":
            tg = load("test", src, base).filter(pl.col("country") == c)
            for f in rules:
                m, tab = mine_multi(f, pairs, s1, tg)
                print(c, src, f, len(m), tab.head(25).select("x", "y", "len", "conf", "support").rows(), flush=True)
                for k, v in m.items():
                    rules[f].setdefault(k, v)
        for f in rules:  # avoid chains/cycles
            tgts = {t for v in rules[f].values() for t in v.split()}
            rules[f] = {k: v for k, v in rules[f].items() if k not in tgts}
        by_country[c] = rules
        print(c, "pseudo pairs", pairs.height, {f: len(r) for f, r in rules.items()})
    maps["by_country"] = by_country
    json.dump(maps, open(f"{WORK}/token_map.json", "w"))
