"""Stage 2: pairwise + list-wise features for candidate pairs.

Pair-level string similarities are computed with rapidfuzz.process.cpdist (vectorised,
multithreaded element-wise scoring), numeric-token agreement with polars list ops, and
list-wise/contextual features (rank within the S1 entity's list, reverse rank among the
S1 entities competing for the same S2/S3 record, gaps to the best score) with window
functions.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

FEATS = []


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def pair_features(p: pl.DataFrame) -> pl.DataFrame:
    """p has columns q_* and t_* (core, nm, legal, addr, nums, lm, indic) + w, c."""
    qa, ta = p["q_core"].to_list(), p["t_core"].to_list()
    f = {}
    f["n_ratio"] = _cp(qa, ta, fuzz.ratio)
    f["n_tsort"] = _cp(qa, ta, fuzz.token_sort_ratio)
    f["n_tset"] = _cp(qa, ta, fuzz.token_set_ratio)
    f["n_partial"] = _cp(qa, ta, fuzz.partial_ratio)
    f["n_jw"] = _cp(qa, ta, JaroWinkler.normalized_similarity)
    qs = [s.replace(" ", "") for s in qa]
    ts = [s.replace(" ", "") for s in ta]
    f["n_nospace"] = _cp(qs, ts, fuzz.ratio)
    f["n_nospace_partial"] = _cp(qs, ts, fuzz.partial_ratio)
    qn, tn = p["q_nm"].to_list(), p["t_nm"].to_list()
    f["nm_tset"] = _cp(qn, tn, fuzz.token_set_ratio)
    qa2, ta2 = p["q_addr"].to_list(), p["t_addr"].to_list()
    f["a_tset"] = _cp(qa2, ta2, fuzz.token_set_ratio)
    f["a_tsort"] = _cp(qa2, ta2, fuzz.token_sort_ratio)
    f["a_partial"] = _cp(qa2, ta2, fuzz.partial_ratio)
    f["a_lev"] = _cp(qa2, ta2, Levenshtein.normalized_similarity)
    qn_, tn_ = p["q_nums"].to_list(), p["t_nums"].to_list()
    f["num_tset"] = _cp(qn_, tn_, fuzz.token_set_ratio)
    out = p.select("qi", "ti").with_columns([pl.Series(k, v) for k, v in f.items()])

    e = p.select(
        qn=pl.col("q_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        tn=pl.col("t_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        qc=pl.col("q_core").str.split(" "), tc=pl.col("t_core").str.split(" "),
        ql=pl.col("q_legal"), tl=pl.col("t_legal"),
        qaddr=pl.col("q_addr"), taddr=pl.col("t_addr"),
    )
    e = e.select(
        num_q=pl.col("qn").list.len(), num_t=pl.col("tn").list.len(),
        num_inter=pl.col("qn").list.set_intersection("tn").list.len(),
        num_union=pl.col("qn").list.set_union("tn").list.len(),
        num_first_eq=(pl.col("qn").list.first() == pl.col("tn").list.first()).fill_null(False),
        # house-number mangling (2808 vs 808, 13110 vs 13000): suffix containment
        num_first_sub=(pl.col("qn").list.first().str.contains(pl.col("tn").list.first(), literal=True)
                       | pl.col("tn").list.first().str.contains(pl.col("qn").list.first(), literal=True)).fill_null(False),
        core_q=pl.col("qc").list.len(), core_t=pl.col("tc").list.len(),
        core_inter=pl.col("qc").list.set_intersection("tc").list.len(),
        core_first_eq=(pl.col("qc").list.first() == pl.col("tc").list.first()),
        legal_state=pl.when((pl.col("ql") == "") | (pl.col("tl") == "")).then(0)
        .when(pl.col("ql") == pl.col("tl")).then(1).otherwise(2),
        a_empty=((pl.col("qaddr") == "").cast(pl.Int8) + (pl.col("taddr") == "").cast(pl.Int8) * 2),
    ).with_columns(num_jac=pl.col("num_inter") / pl.col("num_union").clip(1, None),
                   core_jac=pl.col("core_inter") / (pl.col("core_q") + pl.col("core_t") - pl.col("core_inter")).clip(1, None))
    out = pl.concat([out, e.with_columns(pl.col(pl.Boolean).cast(pl.Int8))], how="horizontal")
    out = out.with_columns(p.select("w", "c", "src", t_indic="t_indic", t_lm="t_lm", q_lm="q_lm"))
    return out


def context_features(f: pl.DataFrame, score="w") -> pl.DataFrame:
    """List-wise features. f has qi (S1 row), ti (target row), src and pair scores."""
    f = f.with_columns(sc=pl.col("w") + pl.col("c"))
    return f.with_columns(
        r_q=pl.col("sc").rank("ordinal", descending=True).over("qi", "src").cast(pl.Int16),
        r_t=pl.col("sc").rank("ordinal", descending=True).over("ti", "src").cast(pl.Int16),
        n_t=pl.len().over("ti", "src").cast(pl.Int16),
        gap_q=pl.col("sc") - pl.col("sc").max().over("qi", "src"),
        gap_t=pl.col("sc") - pl.col("sc").max().over("ti", "src"),
        gap2_t=pl.col("sc") - pl.col("sc").top_k(2).min().over("ti", "src"),
        w_gap_q=pl.col("w") - pl.col("w").max().over("qi", "src"),
        c_gap_q=pl.col("c") - pl.col("c").max().over("qi", "src"),
        w_gap_t=pl.col("w") - pl.col("w").max().over("ti", "src"),
        n_q=pl.len().over("qi", "src").cast(pl.Int16),
        hi_q=(pl.col("sc") > 1.0).sum().over("qi", "src").cast(pl.Int16),
    ).with_columns(
        _ae=((pl.col("a_tset") >= 90) & (pl.col("num_first_eq") == 1)),
        _ne=(pl.col("n_tset") >= 90),
    ).with_columns(
        # address density seen through the candidate lists (robust to address reformatting):
        # several different businesses at one address => a same-address match needs name agreement
        ae_q=pl.col("_ae").sum().over("qi", "src").cast(pl.Int16),
        ae_lown_q=(pl.col("_ae") & (pl.col("n_tset") < 70)).sum().over("qi", "src").cast(pl.Int16),
        ne_q=pl.col("_ne").sum().over("qi", "src").cast(pl.Int16),
        ne_diffaddr_q=(pl.col("_ne") & ~pl.col("_ae")).sum().over("qi", "src").cast(pl.Int16),
        ae_t=pl.col("_ae").sum().over("ti", "src").cast(pl.Int16),
        ae_hin_t=(pl.col("_ae") & pl.col("_ne")).sum().over("ti", "src").cast(pl.Int16),
        ne_t=pl.col("_ne").sum().over("ti", "src").cast(pl.Int16),
    ).drop("_ae", "_ne")
