"""Collective S2<->S3 'anchor' features (third-order evidence).

For a candidate pair (S1 q, target t) we look at q's most confident *other* candidates
(by out-of-fold stage-1 probability p1), called anchors, and measure how strongly t agrees
with them record-to-record (S2<->S3 or S2<->S2). Intuition: all S2/S3 records of one
entity are noisy copies of the same business, so a weak candidate that closely matches
a near-certain match of the same entity is probably right, and one that disagrees with
all confident matches is probably a look-alike.

Features for the top-2 anchors (k = 1, 2):
  anc{k}_p      anchor's stage-1 probability
  anc{k}_same   anchor comes from the same source as t
  anc{k}_ntset  core-name token_set_ratio(t, anchor)
  anc{k}_nns    space-free name ratio(t, anchor)
  anc{k}_atset  address token_set_ratio(t, anchor)
  anc{k}_num    numeric-token token_set_ratio(t, anchor)
plus t_is_top (t is q's most confident candidate).
"""
import gc

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

ANCHOR_FEATS = ["t_is_top"] + [f"anc{k}_{n}" for k in (1, 2) for n in ("p", "same", "ntset", "nns", "atset", "num")]


def _tid(col):
    # S2-123 -> 123, S3-123 -> 10^10 + 123 (compact integer ids keep joins cheap)
    return (pl.col(col).str.slice(3).cast(pl.Int64)
            + pl.when(pl.col(col).str.starts_with("S3")).then(10_000_000_000).otherwise(0))


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def anchor_features(d: pl.DataFrame, recs: pl.DataFrame, n_buckets=8) -> pl.DataFrame:
    """d: q, t, src, p1 (all candidates of the involved S1 entities).
    recs: id, core, addr, nums for every target record that can appear in d.
    Returns q, t + ANCHOR_FEATS."""
    recs = recs.select(tid=_tid("id"), core="core", addr="addr", nums="nums",
                       ns=pl.col("core").str.replace_all(" ", ""))
    d = d.select("q", "t", "src", "p1").with_columns(qid=pl.col("q").str.slice(3).cast(pl.Int64), tid=_tid("t"))
    out = []
    for b in range(n_buckets):
        x = d.filter(pl.col("qid") % n_buckets == b)
        top = (x.with_columns(r=pl.col("p1").rank("ordinal", descending=True).over("qid"))
               .filter(pl.col("r") <= 3).select("qid", aid="tid", ap="p1", asrc="src", ar="r"))
        j = (x.select("qid", "tid", "src").join(top, on="qid").filter(pl.col("aid") != pl.col("tid"))
             .with_columns(k=pl.col("ar").rank("ordinal").over("qid", "tid")).filter(pl.col("k") <= 2))
        j = j.join(recs.rename({"tid": "aid", "core": "a_core", "addr": "a_addr", "nums": "a_nums", "ns": "a_ns"}),
                   on="aid", how="left").join(recs, on="tid", how="left").fill_null("")
        feats = j.select("qid", "tid", "k", ap="ap", same=(pl.col("asrc") == pl.col("src")).cast(pl.Int8)).with_columns(
            ntset=pl.Series(_cp(j["core"].to_list(), j["a_core"].to_list(), fuzz.token_set_ratio)),
            nns=pl.Series(_cp(j["ns"].to_list(), j["a_ns"].to_list(), fuzz.ratio)),
            atset=pl.Series(_cp(j["addr"].to_list(), j["a_addr"].to_list(), fuzz.token_set_ratio)),
            num=pl.Series(_cp(j["nums"].to_list(), j["a_nums"].to_list(), fuzz.token_set_ratio)),
        )
        del j
        wide = None
        for k in (1, 2):
            fk = feats.filter(pl.col("k") == k).drop("k").rename(
                {c: f"anc{k}_{c}" for c in ("ap", "same", "ntset", "nns", "atset", "num")})
            fk = fk.rename({"anc%d_ap" % k: "anc%d_p" % k})
            wide = fk if wide is None else wide.join(fk, on=["qid", "tid"], how="full", coalesce=True)
        topone = top.filter(pl.col("ar") == 1).select("qid", tid="aid").with_columns(t_is_top=pl.lit(1, pl.Int8))
        res = x.select("q", "t", "qid", "tid").join(wide, on=["qid", "tid"], how="left").join(
            topone, on=["qid", "tid"], how="left")
        out.append(res.drop("qid", "tid").with_columns(
            pl.col("t_is_top").fill_null(0),
            *[pl.col(f"anc{k}_p").fill_null(-1.0) for k in (1, 2)]).fill_null(-1))
        del x, top, feats, wide, res
        gc.collect()
    return pl.concat(out)
