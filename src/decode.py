"""Stage 4: metric-aware decoding.

1. Exclusivity: every S2/S3 record belongs to at most one S1 entity (verified on train:
   7.64M ground-truth pairs, 7.64M distinct S2/S3 ids). Each target keeps only its
   best-scoring S1 entity.
2. Expected-F0.5 set decoder: for each S1 entity, candidates sorted by calibrated
   probability p1>=p2>=...; under independence the F-optimal set is a top-k prefix.
   We compute the exact E[F0.5] for every k (k=0 means "predict empty", worth
   P(no true match)) with Poisson-binomial convolutions and output the argmax.
"""
import numpy as np
import polars as pl
from numba import njit


@njit(cache=True)
def _pb(ps):
    d = np.zeros(len(ps) + 1)
    d[0] = 1.0
    for j in range(len(ps)):
        p = ps[j]
        for m in range(j + 1, 0, -1):
            d[m] = d[m] * (1 - p) + d[m - 1] * p
        d[0] *= (1 - p)
    return d


@njit(cache=True)
def _best_k(p, beta2):
    n = len(p)
    best_k = 0
    best = 1.0
    for i in range(n):
        best *= (1 - p[i])  # E[F | predict empty] = P(T = 0)
    rest = _pb(p)  # placeholder so numba types are stable
    for k in range(1, n + 1):
        a = _pb(p[:k])
        b = _pb(p[k:])
        ef = 0.0
        for x in range(1, k + 1):
            if a[x] == 0.0:
                continue
            for y in range(len(b)):
                ef += a[x] * b[y] * (1 + beta2) * x / (beta2 * (x + y) + k)
        if ef > best:
            best = ef
            best_k = k
    return best_k, best


@njit(cache=True)
def decode_all(indptr, probs, beta2):
    """probs sorted descending inside each group [indptr[g], indptr[g+1])."""
    G = len(indptr) - 1
    ks = np.zeros(G, dtype=np.int32)
    efs = np.zeros(G)
    for g in range(G):
        s, e = indptr[g], indptr[g + 1]
        k, ef = _best_k(probs[s:e], beta2)
        ks[g] = k
        efs[g] = ef
    return ks, efs


def decode(pairs: pl.DataFrame, p_col="p", min_p=0.01, exclusive=True, beta=0.5, prior=1.0):
    """pairs: q, t, p. Returns (q, t) predicted matches.

    prior < 1 shrinks all probabilities (conservative mode for unseen domains)."""
    d = pairs.select("q", "t", pl.col(p_col).alias("p") * prior)
    if exclusive:
        d = d.filter(pl.col("p") == pl.col("p").max().over("t"))
        d = d.unique("t", keep="first")
    d = d.filter(pl.col("p") >= min_p).sort(["q", "p"], descending=[False, True])
    qs = d["q"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]]) if len(qs) else np.array([], dtype=np.int64)
    indptr = np.r_[starts, len(qs)].astype(np.int64)
    ks, _ = decode_all(indptr, d["p"].to_numpy().astype(np.float64), beta * beta)
    rank = d.with_columns(pl.int_range(pl.len()).over("q").alias("_r"))
    kmap = pl.DataFrame({"q": qs[starts] if len(qs) else [], "_k": ks})
    return rank.join(kmap, on="q").filter(pl.col("_r") < pl.col("_k")).select("q", "t", "p")


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, all_q) -> float:
    """pred/truth: (q, t) long frames. all_q: every S1 id in the evaluation set."""
    base = pl.DataFrame({"q": list(all_q)})
    tp = pred.join(truth, on=["q", "t"]).group_by("q").len().rename({"len": "tp"})
    k = pred.group_by("q").len().rename({"len": "k"})
    T = truth.group_by("q").len().rename({"len": "T"})
    m = base.join(tp, on="q", how="left").join(k, on="q", how="left").join(T, on="q", how="left").fill_null(0)
    f = m.with_columns(
        f=pl.when((pl.col("T") == 0) & (pl.col("k") == 0)).then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("T") + pl.col("k")).clip(1e-9, None)))
    return float(f["f"].mean())
